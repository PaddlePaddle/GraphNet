from sympy import Symbol, Expr, Rel, Eq

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 80000}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, S1], "L_input_values_"),
    ([4], "L_self_modules_classifier_parameters_bias_"),
    ([4, 256], "L_self_modules_classifier_parameters_weight_"),
    ([256], "L_self_modules_projector_parameters_bias_"),
    ([256, 768], "L_self_modules_projector_parameters_weight_"),
    (
        [1, 9999, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_embed_positions_pe",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_conv_module_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_conv_module_modules_batch_norm_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_conv_module_modules_batch_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_conv_module_modules_batch_norm_parameters_weight_",
    ),
    (
        [768, 1, 31],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_conv_module_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_conv_module_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_conv_module_modules_layer_norm_parameters_weight_",
    ),
    (
        [1536, 768, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_conv_module_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [768, 768, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_conv_module_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_ffn1_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_ffn1_layer_norm_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_ffn1_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_ffn1_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_ffn1_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_ffn1_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_ffn2_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_ffn2_layer_norm_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_ffn2_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_ffn2_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_ffn2_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_ffn2_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [12, 64],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [12, 64],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_conv_module_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_conv_module_modules_batch_norm_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_conv_module_modules_batch_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_conv_module_modules_batch_norm_parameters_weight_",
    ),
    (
        [768, 1, 31],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_conv_module_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_conv_module_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_conv_module_modules_layer_norm_parameters_weight_",
    ),
    (
        [1536, 768, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_conv_module_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [768, 768, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_conv_module_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_ffn1_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_ffn1_layer_norm_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_ffn1_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_ffn1_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_ffn1_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_ffn1_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_ffn2_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_ffn2_layer_norm_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_ffn2_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_ffn2_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_ffn2_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_ffn2_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [12, 64],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [12, 64],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_10_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_conv_module_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_conv_module_modules_batch_norm_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_conv_module_modules_batch_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_conv_module_modules_batch_norm_parameters_weight_",
    ),
    (
        [768, 1, 31],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_conv_module_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_conv_module_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_conv_module_modules_layer_norm_parameters_weight_",
    ),
    (
        [1536, 768, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_conv_module_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [768, 768, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_conv_module_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_ffn1_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_ffn1_layer_norm_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_ffn1_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_ffn1_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_ffn1_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_ffn1_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_ffn2_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_ffn2_layer_norm_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_ffn2_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_ffn2_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_ffn2_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_ffn2_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [12, 64],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [12, 64],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_11_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_conv_module_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_conv_module_modules_batch_norm_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_conv_module_modules_batch_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_conv_module_modules_batch_norm_parameters_weight_",
    ),
    (
        [768, 1, 31],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_conv_module_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_conv_module_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_conv_module_modules_layer_norm_parameters_weight_",
    ),
    (
        [1536, 768, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_conv_module_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [768, 768, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_conv_module_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_ffn1_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_ffn1_layer_norm_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_ffn1_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_ffn1_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_ffn1_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_ffn1_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_ffn2_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_ffn2_layer_norm_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_ffn2_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_ffn2_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_ffn2_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_ffn2_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [12, 64],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [12, 64],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_conv_module_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_conv_module_modules_batch_norm_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_conv_module_modules_batch_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_conv_module_modules_batch_norm_parameters_weight_",
    ),
    (
        [768, 1, 31],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_conv_module_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_conv_module_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_conv_module_modules_layer_norm_parameters_weight_",
    ),
    (
        [1536, 768, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_conv_module_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [768, 768, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_conv_module_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_ffn1_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_ffn1_layer_norm_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_ffn1_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_ffn1_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_ffn1_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_ffn1_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_ffn2_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_ffn2_layer_norm_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_ffn2_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_ffn2_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_ffn2_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_ffn2_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [12, 64],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [12, 64],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_conv_module_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_conv_module_modules_batch_norm_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_conv_module_modules_batch_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_conv_module_modules_batch_norm_parameters_weight_",
    ),
    (
        [768, 1, 31],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_conv_module_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_conv_module_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_conv_module_modules_layer_norm_parameters_weight_",
    ),
    (
        [1536, 768, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_conv_module_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [768, 768, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_conv_module_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_ffn1_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_ffn1_layer_norm_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_ffn1_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_ffn1_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_ffn1_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_ffn1_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_ffn2_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_ffn2_layer_norm_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_ffn2_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_ffn2_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_ffn2_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_ffn2_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [12, 64],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [12, 64],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_conv_module_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_conv_module_modules_batch_norm_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_conv_module_modules_batch_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_conv_module_modules_batch_norm_parameters_weight_",
    ),
    (
        [768, 1, 31],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_conv_module_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_conv_module_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_conv_module_modules_layer_norm_parameters_weight_",
    ),
    (
        [1536, 768, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_conv_module_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [768, 768, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_conv_module_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_ffn1_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_ffn1_layer_norm_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_ffn1_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_ffn1_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_ffn1_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_ffn1_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_ffn2_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_ffn2_layer_norm_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_ffn2_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_ffn2_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_ffn2_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_ffn2_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [12, 64],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [12, 64],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_4_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_conv_module_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_conv_module_modules_batch_norm_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_conv_module_modules_batch_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_conv_module_modules_batch_norm_parameters_weight_",
    ),
    (
        [768, 1, 31],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_conv_module_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_conv_module_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_conv_module_modules_layer_norm_parameters_weight_",
    ),
    (
        [1536, 768, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_conv_module_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [768, 768, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_conv_module_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_ffn1_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_ffn1_layer_norm_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_ffn1_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_ffn1_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_ffn1_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_ffn1_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_ffn2_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_ffn2_layer_norm_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_ffn2_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_ffn2_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_ffn2_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_ffn2_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [12, 64],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [12, 64],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_5_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_conv_module_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_conv_module_modules_batch_norm_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_conv_module_modules_batch_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_conv_module_modules_batch_norm_parameters_weight_",
    ),
    (
        [768, 1, 31],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_conv_module_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_conv_module_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_conv_module_modules_layer_norm_parameters_weight_",
    ),
    (
        [1536, 768, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_conv_module_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [768, 768, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_conv_module_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_ffn1_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_ffn1_layer_norm_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_ffn1_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_ffn1_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_ffn1_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_ffn1_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_ffn2_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_ffn2_layer_norm_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_ffn2_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_ffn2_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_ffn2_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_ffn2_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [12, 64],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [12, 64],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_6_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_conv_module_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_conv_module_modules_batch_norm_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_conv_module_modules_batch_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_conv_module_modules_batch_norm_parameters_weight_",
    ),
    (
        [768, 1, 31],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_conv_module_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_conv_module_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_conv_module_modules_layer_norm_parameters_weight_",
    ),
    (
        [1536, 768, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_conv_module_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [768, 768, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_conv_module_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_ffn1_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_ffn1_layer_norm_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_ffn1_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_ffn1_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_ffn1_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_ffn1_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_ffn2_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_ffn2_layer_norm_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_ffn2_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_ffn2_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_ffn2_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_ffn2_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [12, 64],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [12, 64],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_7_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_conv_module_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_conv_module_modules_batch_norm_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_conv_module_modules_batch_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_conv_module_modules_batch_norm_parameters_weight_",
    ),
    (
        [768, 1, 31],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_conv_module_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_conv_module_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_conv_module_modules_layer_norm_parameters_weight_",
    ),
    (
        [1536, 768, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_conv_module_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [768, 768, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_conv_module_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_ffn1_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_ffn1_layer_norm_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_ffn1_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_ffn1_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_ffn1_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_ffn1_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_ffn2_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_ffn2_layer_norm_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_ffn2_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_ffn2_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_ffn2_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_ffn2_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [12, 64],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [12, 64],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_8_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_conv_module_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_conv_module_modules_batch_norm_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_conv_module_modules_batch_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_conv_module_modules_batch_norm_parameters_weight_",
    ),
    (
        [768, 1, 31],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_conv_module_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_conv_module_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_conv_module_modules_layer_norm_parameters_weight_",
    ),
    (
        [1536, 768, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_conv_module_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [768, 768, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_conv_module_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_ffn1_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_ffn1_layer_norm_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_ffn1_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_ffn1_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_ffn1_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_ffn1_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_ffn2_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_ffn2_layer_norm_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_ffn2_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_ffn2_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_ffn2_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_ffn2_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [12, 64],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [12, 64],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_9_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [512, 1, 10],
        "L_self_modules_wav2vec2_conformer_modules_feature_extractor_modules_conv_layers_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_wav2vec2_conformer_modules_feature_extractor_modules_conv_layers_modules_0_modules_layer_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_wav2vec2_conformer_modules_feature_extractor_modules_conv_layers_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [512, 512, 3],
        "L_self_modules_wav2vec2_conformer_modules_feature_extractor_modules_conv_layers_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_wav2vec2_conformer_modules_feature_extractor_modules_conv_layers_modules_1_modules_layer_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_wav2vec2_conformer_modules_feature_extractor_modules_conv_layers_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [512, 512, 3],
        "L_self_modules_wav2vec2_conformer_modules_feature_extractor_modules_conv_layers_modules_2_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_wav2vec2_conformer_modules_feature_extractor_modules_conv_layers_modules_2_modules_layer_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_wav2vec2_conformer_modules_feature_extractor_modules_conv_layers_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [512, 512, 3],
        "L_self_modules_wav2vec2_conformer_modules_feature_extractor_modules_conv_layers_modules_3_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_wav2vec2_conformer_modules_feature_extractor_modules_conv_layers_modules_3_modules_layer_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_wav2vec2_conformer_modules_feature_extractor_modules_conv_layers_modules_3_modules_layer_norm_parameters_weight_",
    ),
    (
        [512, 512, 3],
        "L_self_modules_wav2vec2_conformer_modules_feature_extractor_modules_conv_layers_modules_4_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_wav2vec2_conformer_modules_feature_extractor_modules_conv_layers_modules_4_modules_layer_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_wav2vec2_conformer_modules_feature_extractor_modules_conv_layers_modules_4_modules_layer_norm_parameters_weight_",
    ),
    (
        [512, 512, 2],
        "L_self_modules_wav2vec2_conformer_modules_feature_extractor_modules_conv_layers_modules_5_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_wav2vec2_conformer_modules_feature_extractor_modules_conv_layers_modules_5_modules_layer_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_wav2vec2_conformer_modules_feature_extractor_modules_conv_layers_modules_5_modules_layer_norm_parameters_weight_",
    ),
    (
        [512, 512, 2],
        "L_self_modules_wav2vec2_conformer_modules_feature_extractor_modules_conv_layers_modules_6_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_wav2vec2_conformer_modules_feature_extractor_modules_conv_layers_modules_6_modules_layer_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_wav2vec2_conformer_modules_feature_extractor_modules_conv_layers_modules_6_modules_layer_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_wav2vec2_conformer_modules_feature_projection_modules_layer_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_wav2vec2_conformer_modules_feature_projection_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_wav2vec2_conformer_modules_feature_projection_modules_projection_parameters_bias_",
    ),
    (
        [768, 512],
        "L_self_modules_wav2vec2_conformer_modules_feature_projection_modules_projection_parameters_weight_",
    ),
]
