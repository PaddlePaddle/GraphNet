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
    ([256, 16], "L_self_modules_projector_parameters_weight_"),
    (
        [1, 9999, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_embed_positions_pe",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_conv_module_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_conv_module_modules_batch_norm_buffers_running_var_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_conv_module_modules_batch_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_conv_module_modules_batch_norm_parameters_weight_",
    ),
    (
        [16, 1, 31],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_conv_module_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_conv_module_modules_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_conv_module_modules_layer_norm_parameters_weight_",
    ),
    (
        [32, 16, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_conv_module_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [16, 16, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_conv_module_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_ffn1_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_ffn1_layer_norm_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_ffn1_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [20, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_ffn1_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_ffn1_modules_output_dense_parameters_bias_",
    ),
    (
        [16, 20],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_ffn1_modules_output_dense_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_ffn2_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_ffn2_layer_norm_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_ffn2_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [20, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_ffn2_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_ffn2_modules_output_dense_parameters_bias_",
    ),
    (
        [16, 20],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_ffn2_modules_output_dense_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [16, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [2, 8],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [2, 8],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_0_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_conv_module_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_conv_module_modules_batch_norm_buffers_running_var_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_conv_module_modules_batch_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_conv_module_modules_batch_norm_parameters_weight_",
    ),
    (
        [16, 1, 31],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_conv_module_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_conv_module_modules_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_conv_module_modules_layer_norm_parameters_weight_",
    ),
    (
        [32, 16, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_conv_module_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [16, 16, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_conv_module_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_ffn1_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_ffn1_layer_norm_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_ffn1_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [20, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_ffn1_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_ffn1_modules_output_dense_parameters_bias_",
    ),
    (
        [16, 20],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_ffn1_modules_output_dense_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_ffn2_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_ffn2_layer_norm_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_ffn2_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [20, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_ffn2_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_ffn2_modules_output_dense_parameters_bias_",
    ),
    (
        [16, 20],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_ffn2_modules_output_dense_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [16, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [2, 8],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [2, 8],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_1_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_conv_module_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_conv_module_modules_batch_norm_buffers_running_var_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_conv_module_modules_batch_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_conv_module_modules_batch_norm_parameters_weight_",
    ),
    (
        [16, 1, 31],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_conv_module_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_conv_module_modules_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_conv_module_modules_layer_norm_parameters_weight_",
    ),
    (
        [32, 16, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_conv_module_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [16, 16, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_conv_module_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_ffn1_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_ffn1_layer_norm_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_ffn1_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [20, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_ffn1_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_ffn1_modules_output_dense_parameters_bias_",
    ),
    (
        [16, 20],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_ffn1_modules_output_dense_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_ffn2_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_ffn2_layer_norm_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_ffn2_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [20, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_ffn2_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_ffn2_modules_output_dense_parameters_bias_",
    ),
    (
        [16, 20],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_ffn2_modules_output_dense_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [16, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [2, 8],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [2, 8],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_2_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_conv_module_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_conv_module_modules_batch_norm_buffers_running_var_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_conv_module_modules_batch_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_conv_module_modules_batch_norm_parameters_weight_",
    ),
    (
        [16, 1, 31],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_conv_module_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_conv_module_modules_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_conv_module_modules_layer_norm_parameters_weight_",
    ),
    (
        [32, 16, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_conv_module_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [16, 16, 1],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_conv_module_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_ffn1_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_ffn1_layer_norm_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_ffn1_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [20, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_ffn1_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_ffn1_modules_output_dense_parameters_bias_",
    ),
    (
        [16, 20],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_ffn1_modules_output_dense_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_ffn2_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_ffn2_layer_norm_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_ffn2_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [20, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_ffn2_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_ffn2_modules_output_dense_parameters_bias_",
    ),
    (
        [16, 20],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_ffn2_modules_output_dense_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [16, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [2, 8],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [2, 8],
        "L_self_modules_wav2vec2_conformer_modules_encoder_modules_layers_modules_3_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [32, 1, 8],
        "L_self_modules_wav2vec2_conformer_modules_feature_extractor_modules_conv_layers_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_wav2vec2_conformer_modules_feature_extractor_modules_conv_layers_modules_0_modules_layer_norm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_wav2vec2_conformer_modules_feature_extractor_modules_conv_layers_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [32, 32, 8],
        "L_self_modules_wav2vec2_conformer_modules_feature_extractor_modules_conv_layers_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [32, 32, 8],
        "L_self_modules_wav2vec2_conformer_modules_feature_extractor_modules_conv_layers_modules_2_modules_conv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_wav2vec2_conformer_modules_feature_projection_modules_layer_norm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_wav2vec2_conformer_modules_feature_projection_modules_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_wav2vec2_conformer_modules_feature_projection_modules_projection_parameters_bias_",
    ),
    (
        [16, 32],
        "L_self_modules_wav2vec2_conformer_modules_feature_projection_modules_projection_parameters_weight_",
    ),
]
