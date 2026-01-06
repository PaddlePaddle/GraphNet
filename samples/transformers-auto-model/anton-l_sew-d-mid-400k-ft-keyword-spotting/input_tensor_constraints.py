from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 80000}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, S0], "L_input_values_"),
    ([12], "L_self_modules_classifier_parameters_bias_"),
    ([12, 256], "L_self_modules_classifier_parameters_weight_"),
    ([256], "L_self_modules_projector_parameters_bias_"),
    ([256, 512], "L_self_modules_projector_parameters_weight_"),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_12_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_12_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_13_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_13_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_14_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_14_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_15_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_15_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_16_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_16_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_17_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_17_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_18_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_18_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_19_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_19_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_20_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_20_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_21_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_21_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_22_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_22_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_23_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_23_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512, 512],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_rel_embeddings_parameters_weight_",
    ),
    (
        [1, 1, 31],
        "L_self_modules_sew_d_modules_encoder_modules_pos_conv_embed_modules_conv_modules_parametrizations_modules_weight_parameters_original0_",
    ),
    (
        [512, 32, 31],
        "L_self_modules_sew_d_modules_encoder_modules_pos_conv_embed_modules_conv_modules_parametrizations_modules_weight_parameters_original1_",
    ),
    (
        [512],
        "L_self_modules_sew_d_modules_encoder_modules_pos_conv_embed_modules_conv_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_sew_d_modules_encoder_modules_upsample_modules_projection_parameters_bias_",
    ),
    (
        [1024, 512],
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
        [512, 512, 1],
        "L_self_modules_sew_d_modules_feature_extractor_modules_conv_layers_modules_10_modules_conv_parameters_weight_",
    ),
    (
        [512, 512, 2],
        "L_self_modules_sew_d_modules_feature_extractor_modules_conv_layers_modules_11_modules_conv_parameters_weight_",
    ),
    (
        [512, 512, 1],
        "L_self_modules_sew_d_modules_feature_extractor_modules_conv_layers_modules_12_modules_conv_parameters_weight_",
    ),
    (
        [128, 64, 3],
        "L_self_modules_sew_d_modules_feature_extractor_modules_conv_layers_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [128, 128, 1],
        "L_self_modules_sew_d_modules_feature_extractor_modules_conv_layers_modules_2_modules_conv_parameters_weight_",
    ),
    (
        [128, 128, 3],
        "L_self_modules_sew_d_modules_feature_extractor_modules_conv_layers_modules_3_modules_conv_parameters_weight_",
    ),
    (
        [128, 128, 1],
        "L_self_modules_sew_d_modules_feature_extractor_modules_conv_layers_modules_4_modules_conv_parameters_weight_",
    ),
    (
        [256, 128, 3],
        "L_self_modules_sew_d_modules_feature_extractor_modules_conv_layers_modules_5_modules_conv_parameters_weight_",
    ),
    (
        [256, 256, 1],
        "L_self_modules_sew_d_modules_feature_extractor_modules_conv_layers_modules_6_modules_conv_parameters_weight_",
    ),
    (
        [256, 256, 3],
        "L_self_modules_sew_d_modules_feature_extractor_modules_conv_layers_modules_7_modules_conv_parameters_weight_",
    ),
    (
        [256, 256, 1],
        "L_self_modules_sew_d_modules_feature_extractor_modules_conv_layers_modules_8_modules_conv_parameters_weight_",
    ),
    (
        [512, 256, 2],
        "L_self_modules_sew_d_modules_feature_extractor_modules_conv_layers_modules_9_modules_conv_parameters_weight_",
    ),
    ([512], "L_self_modules_sew_d_modules_layer_norm_parameters_bias_"),
    ([512], "L_self_modules_sew_d_modules_layer_norm_parameters_weight_"),
]
