import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_pixel_values_: torch.Tensor,
        L_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_parameters_cls_token_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_parameters_position_embeddings_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_33_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_33_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_33_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_33_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_33_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_33_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_33_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_33_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_33_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_33_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_34_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_34_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_34_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_34_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_34_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_34_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_34_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_34_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_34_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_34_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_35_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_35_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_35_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_35_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_35_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_35_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_35_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_35_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_35_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_35_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_36_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_36_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_36_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_36_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_36_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_36_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_36_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_36_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_36_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_36_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_37_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_37_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_37_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_37_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_37_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_37_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_37_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_37_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_37_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_37_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_38_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_38_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_38_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_38_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_38_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_38_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_38_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_38_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_38_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_38_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_39_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_39_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_39_modules_layer_scale1_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_39_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_39_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_39_modules_mlp_modules_weights_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_39_modules_mlp_modules_weights_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_39_modules_mlp_modules_weights_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_39_modules_mlp_modules_weights_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_39_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
        L_self_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_pixel_values_ = L_pixel_values_
        l_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_ = L_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_
        l_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_ = L_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_
        l_self_modules_embeddings_parameters_cls_token_ = (
            L_self_modules_embeddings_parameters_cls_token_
        )
        l_self_modules_embeddings_parameters_position_embeddings_ = (
            L_self_modules_embeddings_parameters_position_embeddings_
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_0_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_0_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_0_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_1_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_1_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_1_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_1_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_2_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_2_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_2_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_2_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_3_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_3_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_3_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_3_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_4_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_4_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_4_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_4_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_5_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_5_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_5_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_5_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_6_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_6_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_6_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_6_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_7_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_7_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_7_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_7_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_8_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_8_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_8_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_8_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_9_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_9_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_9_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_9_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_10_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_10_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_10_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_10_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_11_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_11_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_11_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_11_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_12_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_12_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_12_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_12_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_13_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_13_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_13_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_13_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_14_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_14_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_14_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_14_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_15_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_15_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_15_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_15_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_16_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_16_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_16_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_16_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_17_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_17_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_17_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_17_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_18_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_18_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_18_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_18_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_19_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_19_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_19_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_19_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_20_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_20_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_20_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_20_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_21_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_21_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_21_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_21_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_22_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_22_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_22_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_22_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_23_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_23_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_23_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_23_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_24_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_24_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_24_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_24_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_24_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_24_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_24_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_24_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_24_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_24_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_24_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_24_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_24_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_24_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_24_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_24_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_24_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_24_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_24_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_24_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_25_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_25_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_25_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_25_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_25_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_25_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_25_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_25_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_25_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_25_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_25_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_25_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_25_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_25_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_25_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_25_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_25_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_25_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_25_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_25_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_26_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_26_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_26_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_26_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_26_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_26_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_26_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_26_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_26_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_26_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_26_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_26_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_26_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_26_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_26_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_26_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_26_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_26_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_26_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_26_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_27_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_27_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_27_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_27_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_27_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_27_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_27_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_27_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_27_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_27_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_27_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_27_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_27_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_27_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_27_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_27_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_27_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_27_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_27_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_27_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_28_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_28_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_28_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_28_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_28_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_28_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_28_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_28_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_28_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_28_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_28_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_28_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_28_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_28_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_28_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_28_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_28_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_28_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_28_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_28_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_29_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_29_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_29_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_29_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_29_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_29_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_29_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_29_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_29_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_29_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_29_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_29_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_29_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_29_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_29_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_29_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_29_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_29_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_29_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_29_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_30_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_30_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_30_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_30_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_30_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_30_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_30_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_30_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_30_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_30_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_30_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_30_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_30_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_30_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_30_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_30_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_30_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_30_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_30_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_30_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_31_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_31_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_31_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_31_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_31_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_31_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_31_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_31_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_31_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_31_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_31_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_31_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_31_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_31_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_31_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_31_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_31_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_31_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_31_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_31_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_32_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_32_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_32_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_32_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_32_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_32_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_32_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_32_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_32_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_32_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_32_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_32_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_32_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_32_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_32_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_32_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_32_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_32_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_32_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_32_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_33_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_33_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_33_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_33_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_33_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_33_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_33_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_33_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_33_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_33_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_33_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_33_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_33_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_33_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_33_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_33_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_33_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_33_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_33_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_33_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_34_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_34_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_34_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_34_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_34_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_34_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_34_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_34_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_34_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_34_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_34_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_34_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_34_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_34_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_34_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_34_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_34_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_34_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_34_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_34_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_35_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_35_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_35_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_35_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_35_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_35_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_35_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_35_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_35_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_35_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_35_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_35_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_35_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_35_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_35_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_35_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_35_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_35_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_35_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_35_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_36_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_36_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_36_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_36_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_36_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_36_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_36_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_36_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_36_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_36_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_36_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_36_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_36_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_36_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_36_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_36_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_36_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_36_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_36_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_36_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_37_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_37_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_37_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_37_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_37_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_37_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_37_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_37_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_37_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_37_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_37_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_37_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_37_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_37_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_37_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_37_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_37_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_37_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_37_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_37_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_38_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_38_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_38_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_38_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_38_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_38_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_38_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_38_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_38_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_38_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_38_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_38_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_38_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_38_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_38_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_38_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_38_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_38_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_38_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_38_modules_layer_scale2_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_39_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_39_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_39_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_39_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_39_modules_layer_scale1_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_39_modules_layer_scale1_parameters_lambda1_
        l_self_modules_encoder_modules_layer_modules_39_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_39_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_39_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_39_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_39_modules_mlp_modules_weights_in_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_39_modules_mlp_modules_weights_in_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_39_modules_mlp_modules_weights_in_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_39_modules_mlp_modules_weights_in_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_39_modules_mlp_modules_weights_out_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_39_modules_mlp_modules_weights_out_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_39_modules_mlp_modules_weights_out_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_39_modules_mlp_modules_weights_out_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_39_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_39_modules_layer_scale2_parameters_lambda1_
        l_self_modules_layernorm_parameters_weight_ = (
            L_self_modules_layernorm_parameters_weight_
        )
        l_self_modules_layernorm_parameters_bias_ = (
            L_self_modules_layernorm_parameters_bias_
        )
        to = l_pixel_values_.to(dtype=torch.float32)
        l_pixel_values_ = None
        conv2d = torch.conv2d(
            to,
            l_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_,
            l_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_,
            (14, 14),
            (0, 0),
            (1, 1),
            1,
        )
        to = l_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_ = l_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_ = (None)
        flatten = conv2d.flatten(2)
        conv2d = None
        embeddings = flatten.transpose(1, 2)
        flatten = None
        cls_tokens = l_self_modules_embeddings_parameters_cls_token_.expand(1, -1, -1)
        l_self_modules_embeddings_parameters_cls_token_ = None
        embeddings_1 = torch.cat((cls_tokens, embeddings), dim=1)
        cls_tokens = embeddings = None
        class_pos_embed = l_self_modules_embeddings_parameters_position_embeddings_[
            (slice(None, None, None), slice(None, 1, None))
        ]
        patch_pos_embed = l_self_modules_embeddings_parameters_position_embeddings_[
            (slice(None, None, None), slice(1, None, None))
        ]
        l_self_modules_embeddings_parameters_position_embeddings_ = None
        patch_pos_embed_1 = patch_pos_embed.reshape(1, 37, 37, 1536)
        patch_pos_embed = None
        patch_pos_embed_2 = patch_pos_embed_1.permute(0, 3, 1, 2)
        patch_pos_embed_1 = None
        to_1 = patch_pos_embed_2.to(torch.float32)
        patch_pos_embed_2 = None
        interpolate = torch.nn.functional.interpolate(
            to_1, size=(16, 16), mode="bicubic", align_corners=False
        )
        to_1 = None
        patch_pos_embed_3 = interpolate.to(dtype=torch.float32)
        interpolate = None
        permute_1 = patch_pos_embed_3.permute(0, 2, 3, 1)
        patch_pos_embed_3 = None
        patch_pos_embed_4 = permute_1.view(1, -1, 1536)
        permute_1 = None
        cat_1 = torch.cat((class_pos_embed, patch_pos_embed_4), dim=1)
        class_pos_embed = patch_pos_embed_4 = None
        embeddings_2 = embeddings_1 + cat_1
        embeddings_1 = cat_1 = None
        embeddings_3 = torch.nn.functional.dropout(embeddings_2, 0.0, False, False)
        embeddings_2 = None
        layer_norm = torch.nn.functional.layer_norm(
            embeddings_3,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_norm1_parameters_bias_ = (None)
        linear = torch._C._nn.linear(
            layer_norm,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_1 = linear.view(1, -1, 24, 64)
        linear = None
        key_layer = view_1.transpose(1, 2)
        view_1 = None
        linear_1 = torch._C._nn.linear(
            layer_norm,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_2 = linear_1.view(1, -1, 24, 64)
        linear_1 = None
        value_layer = view_2.transpose(1, 2)
        view_2 = None
        linear_2 = torch._C._nn.linear(
            layer_norm,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_3 = linear_2.view(1, -1, 24, 64)
        linear_2 = None
        query_layer = view_3.transpose(1, 2)
        view_3 = None
        query = query_layer.contiguous()
        query_layer = None
        key = key_layer.contiguous()
        key_layer = None
        value = value_layer.contiguous()
        value_layer = None
        attn_output = torch._C._nn.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query = key = value = None
        transpose_4 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_4.contiguous()
        transpose_4 = None
        context_layer = attn_output_1.reshape((1, 257, 1536))
        attn_output_1 = None
        hidden_states = torch._C._nn.linear(
            context_layer,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_1 = torch.nn.functional.dropout(hidden_states, 0.0, False, False)
        hidden_states = None
        attention_output = (
            hidden_states_1
            * l_self_modules_encoder_modules_layer_modules_0_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_1 = l_self_modules_encoder_modules_layer_modules_0_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_2 = attention_output + embeddings_3
        attention_output = embeddings_3 = None
        layer_output = torch.nn.functional.layer_norm(
            hidden_states_2,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_norm2_parameters_bias_ = (None)
        hidden_state = torch._C._nn.linear(
            layer_output,
            l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output = l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk = hidden_state.chunk(2, dim=-1)
        hidden_state = None
        x1 = chunk[0]
        x2 = chunk[1]
        chunk = None
        silu = torch.nn.functional.silu(x1)
        x1 = None
        hidden = silu * x2
        silu = x2 = None
        layer_output_1 = torch._C._nn.linear(
            hidden,
            l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden = l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_2 = (
            layer_output_1
            * l_self_modules_encoder_modules_layer_modules_0_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_1 = l_self_modules_encoder_modules_layer_modules_0_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_3 = layer_output_2 + hidden_states_2
        layer_output_2 = hidden_states_2 = None
        layer_norm_2 = torch.nn.functional.layer_norm(
            layer_output_3,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_norm1_parameters_bias_ = (None)
        linear_6 = torch._C._nn.linear(
            layer_norm_2,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_4 = linear_6.view(1, -1, 24, 64)
        linear_6 = None
        key_layer_1 = view_4.transpose(1, 2)
        view_4 = None
        linear_7 = torch._C._nn.linear(
            layer_norm_2,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_5 = linear_7.view(1, -1, 24, 64)
        linear_7 = None
        value_layer_1 = view_5.transpose(1, 2)
        view_5 = None
        linear_8 = torch._C._nn.linear(
            layer_norm_2,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_2 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_6 = linear_8.view(1, -1, 24, 64)
        linear_8 = None
        query_layer_1 = view_6.transpose(1, 2)
        view_6 = None
        query_1 = query_layer_1.contiguous()
        query_layer_1 = None
        key_1 = key_layer_1.contiguous()
        key_layer_1 = None
        value_1 = value_layer_1.contiguous()
        value_layer_1 = None
        attn_output_2 = torch._C._nn.scaled_dot_product_attention(
            query_1,
            key_1,
            value_1,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_1 = key_1 = value_1 = None
        transpose_8 = attn_output_2.transpose(1, 2)
        attn_output_2 = None
        attn_output_3 = transpose_8.contiguous()
        transpose_8 = None
        context_layer_1 = attn_output_3.reshape((1, 257, 1536))
        attn_output_3 = None
        hidden_states_3 = torch._C._nn.linear(
            context_layer_1,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_1 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_4 = torch.nn.functional.dropout(
            hidden_states_3, 0.0, False, False
        )
        hidden_states_3 = None
        attention_output_1 = (
            hidden_states_4
            * l_self_modules_encoder_modules_layer_modules_1_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_4 = l_self_modules_encoder_modules_layer_modules_1_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_5 = attention_output_1 + layer_output_3
        attention_output_1 = layer_output_3 = None
        layer_output_4 = torch.nn.functional.layer_norm(
            hidden_states_5,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_norm2_parameters_bias_ = (None)
        hidden_state_1 = torch._C._nn.linear(
            layer_output_4,
            l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_4 = l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_1 = hidden_state_1.chunk(2, dim=-1)
        hidden_state_1 = None
        x1_1 = chunk_1[0]
        x2_1 = chunk_1[1]
        chunk_1 = None
        silu_1 = torch.nn.functional.silu(x1_1)
        x1_1 = None
        hidden_1 = silu_1 * x2_1
        silu_1 = x2_1 = None
        layer_output_5 = torch._C._nn.linear(
            hidden_1,
            l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_1 = l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_6 = (
            layer_output_5
            * l_self_modules_encoder_modules_layer_modules_1_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_5 = l_self_modules_encoder_modules_layer_modules_1_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_7 = layer_output_6 + hidden_states_5
        layer_output_6 = hidden_states_5 = None
        layer_norm_4 = torch.nn.functional.layer_norm(
            layer_output_7,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_norm1_parameters_bias_ = (None)
        linear_12 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_7 = linear_12.view(1, -1, 24, 64)
        linear_12 = None
        key_layer_2 = view_7.transpose(1, 2)
        view_7 = None
        linear_13 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_8 = linear_13.view(1, -1, 24, 64)
        linear_13 = None
        value_layer_2 = view_8.transpose(1, 2)
        view_8 = None
        linear_14 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_4 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_9 = linear_14.view(1, -1, 24, 64)
        linear_14 = None
        query_layer_2 = view_9.transpose(1, 2)
        view_9 = None
        query_2 = query_layer_2.contiguous()
        query_layer_2 = None
        key_2 = key_layer_2.contiguous()
        key_layer_2 = None
        value_2 = value_layer_2.contiguous()
        value_layer_2 = None
        attn_output_4 = torch._C._nn.scaled_dot_product_attention(
            query_2,
            key_2,
            value_2,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_2 = key_2 = value_2 = None
        transpose_12 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_12.contiguous()
        transpose_12 = None
        context_layer_2 = attn_output_5.reshape((1, 257, 1536))
        attn_output_5 = None
        hidden_states_6 = torch._C._nn.linear(
            context_layer_2,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_2 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_7 = torch.nn.functional.dropout(
            hidden_states_6, 0.0, False, False
        )
        hidden_states_6 = None
        attention_output_2 = (
            hidden_states_7
            * l_self_modules_encoder_modules_layer_modules_2_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_7 = l_self_modules_encoder_modules_layer_modules_2_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_8 = attention_output_2 + layer_output_7
        attention_output_2 = layer_output_7 = None
        layer_output_8 = torch.nn.functional.layer_norm(
            hidden_states_8,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_norm2_parameters_bias_ = (None)
        hidden_state_2 = torch._C._nn.linear(
            layer_output_8,
            l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_8 = l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_2 = hidden_state_2.chunk(2, dim=-1)
        hidden_state_2 = None
        x1_2 = chunk_2[0]
        x2_2 = chunk_2[1]
        chunk_2 = None
        silu_2 = torch.nn.functional.silu(x1_2)
        x1_2 = None
        hidden_2 = silu_2 * x2_2
        silu_2 = x2_2 = None
        layer_output_9 = torch._C._nn.linear(
            hidden_2,
            l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_2 = l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_10 = (
            layer_output_9
            * l_self_modules_encoder_modules_layer_modules_2_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_9 = l_self_modules_encoder_modules_layer_modules_2_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_11 = layer_output_10 + hidden_states_8
        layer_output_10 = hidden_states_8 = None
        layer_norm_6 = torch.nn.functional.layer_norm(
            layer_output_11,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_norm1_parameters_bias_ = (None)
        linear_18 = torch._C._nn.linear(
            layer_norm_6,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_10 = linear_18.view(1, -1, 24, 64)
        linear_18 = None
        key_layer_3 = view_10.transpose(1, 2)
        view_10 = None
        linear_19 = torch._C._nn.linear(
            layer_norm_6,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_11 = linear_19.view(1, -1, 24, 64)
        linear_19 = None
        value_layer_3 = view_11.transpose(1, 2)
        view_11 = None
        linear_20 = torch._C._nn.linear(
            layer_norm_6,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_6 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_12 = linear_20.view(1, -1, 24, 64)
        linear_20 = None
        query_layer_3 = view_12.transpose(1, 2)
        view_12 = None
        query_3 = query_layer_3.contiguous()
        query_layer_3 = None
        key_3 = key_layer_3.contiguous()
        key_layer_3 = None
        value_3 = value_layer_3.contiguous()
        value_layer_3 = None
        attn_output_6 = torch._C._nn.scaled_dot_product_attention(
            query_3,
            key_3,
            value_3,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_3 = key_3 = value_3 = None
        transpose_16 = attn_output_6.transpose(1, 2)
        attn_output_6 = None
        attn_output_7 = transpose_16.contiguous()
        transpose_16 = None
        context_layer_3 = attn_output_7.reshape((1, 257, 1536))
        attn_output_7 = None
        hidden_states_9 = torch._C._nn.linear(
            context_layer_3,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_3 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_10 = torch.nn.functional.dropout(
            hidden_states_9, 0.0, False, False
        )
        hidden_states_9 = None
        attention_output_3 = (
            hidden_states_10
            * l_self_modules_encoder_modules_layer_modules_3_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_10 = l_self_modules_encoder_modules_layer_modules_3_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_11 = attention_output_3 + layer_output_11
        attention_output_3 = layer_output_11 = None
        layer_output_12 = torch.nn.functional.layer_norm(
            hidden_states_11,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_norm2_parameters_bias_ = (None)
        hidden_state_3 = torch._C._nn.linear(
            layer_output_12,
            l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_12 = l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_3 = hidden_state_3.chunk(2, dim=-1)
        hidden_state_3 = None
        x1_3 = chunk_3[0]
        x2_3 = chunk_3[1]
        chunk_3 = None
        silu_3 = torch.nn.functional.silu(x1_3)
        x1_3 = None
        hidden_3 = silu_3 * x2_3
        silu_3 = x2_3 = None
        layer_output_13 = torch._C._nn.linear(
            hidden_3,
            l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_3 = l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_14 = (
            layer_output_13
            * l_self_modules_encoder_modules_layer_modules_3_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_13 = l_self_modules_encoder_modules_layer_modules_3_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_15 = layer_output_14 + hidden_states_11
        layer_output_14 = hidden_states_11 = None
        layer_norm_8 = torch.nn.functional.layer_norm(
            layer_output_15,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_norm1_parameters_bias_ = (None)
        linear_24 = torch._C._nn.linear(
            layer_norm_8,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_13 = linear_24.view(1, -1, 24, 64)
        linear_24 = None
        key_layer_4 = view_13.transpose(1, 2)
        view_13 = None
        linear_25 = torch._C._nn.linear(
            layer_norm_8,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_14 = linear_25.view(1, -1, 24, 64)
        linear_25 = None
        value_layer_4 = view_14.transpose(1, 2)
        view_14 = None
        linear_26 = torch._C._nn.linear(
            layer_norm_8,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_8 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_15 = linear_26.view(1, -1, 24, 64)
        linear_26 = None
        query_layer_4 = view_15.transpose(1, 2)
        view_15 = None
        query_4 = query_layer_4.contiguous()
        query_layer_4 = None
        key_4 = key_layer_4.contiguous()
        key_layer_4 = None
        value_4 = value_layer_4.contiguous()
        value_layer_4 = None
        attn_output_8 = torch._C._nn.scaled_dot_product_attention(
            query_4,
            key_4,
            value_4,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_4 = key_4 = value_4 = None
        transpose_20 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_20.contiguous()
        transpose_20 = None
        context_layer_4 = attn_output_9.reshape((1, 257, 1536))
        attn_output_9 = None
        hidden_states_12 = torch._C._nn.linear(
            context_layer_4,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_4 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_13 = torch.nn.functional.dropout(
            hidden_states_12, 0.0, False, False
        )
        hidden_states_12 = None
        attention_output_4 = (
            hidden_states_13
            * l_self_modules_encoder_modules_layer_modules_4_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_13 = l_self_modules_encoder_modules_layer_modules_4_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_14 = attention_output_4 + layer_output_15
        attention_output_4 = layer_output_15 = None
        layer_output_16 = torch.nn.functional.layer_norm(
            hidden_states_14,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_norm2_parameters_bias_ = (None)
        hidden_state_4 = torch._C._nn.linear(
            layer_output_16,
            l_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_16 = l_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_4 = hidden_state_4.chunk(2, dim=-1)
        hidden_state_4 = None
        x1_4 = chunk_4[0]
        x2_4 = chunk_4[1]
        chunk_4 = None
        silu_4 = torch.nn.functional.silu(x1_4)
        x1_4 = None
        hidden_4 = silu_4 * x2_4
        silu_4 = x2_4 = None
        layer_output_17 = torch._C._nn.linear(
            hidden_4,
            l_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_4 = l_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_18 = (
            layer_output_17
            * l_self_modules_encoder_modules_layer_modules_4_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_17 = l_self_modules_encoder_modules_layer_modules_4_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_19 = layer_output_18 + hidden_states_14
        layer_output_18 = hidden_states_14 = None
        layer_norm_10 = torch.nn.functional.layer_norm(
            layer_output_19,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_norm1_parameters_bias_ = (None)
        linear_30 = torch._C._nn.linear(
            layer_norm_10,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_16 = linear_30.view(1, -1, 24, 64)
        linear_30 = None
        key_layer_5 = view_16.transpose(1, 2)
        view_16 = None
        linear_31 = torch._C._nn.linear(
            layer_norm_10,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_17 = linear_31.view(1, -1, 24, 64)
        linear_31 = None
        value_layer_5 = view_17.transpose(1, 2)
        view_17 = None
        linear_32 = torch._C._nn.linear(
            layer_norm_10,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_10 = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_18 = linear_32.view(1, -1, 24, 64)
        linear_32 = None
        query_layer_5 = view_18.transpose(1, 2)
        view_18 = None
        query_5 = query_layer_5.contiguous()
        query_layer_5 = None
        key_5 = key_layer_5.contiguous()
        key_layer_5 = None
        value_5 = value_layer_5.contiguous()
        value_layer_5 = None
        attn_output_10 = torch._C._nn.scaled_dot_product_attention(
            query_5,
            key_5,
            value_5,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_5 = key_5 = value_5 = None
        transpose_24 = attn_output_10.transpose(1, 2)
        attn_output_10 = None
        attn_output_11 = transpose_24.contiguous()
        transpose_24 = None
        context_layer_5 = attn_output_11.reshape((1, 257, 1536))
        attn_output_11 = None
        hidden_states_15 = torch._C._nn.linear(
            context_layer_5,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_5 = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_16 = torch.nn.functional.dropout(
            hidden_states_15, 0.0, False, False
        )
        hidden_states_15 = None
        attention_output_5 = (
            hidden_states_16
            * l_self_modules_encoder_modules_layer_modules_5_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_16 = l_self_modules_encoder_modules_layer_modules_5_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_17 = attention_output_5 + layer_output_19
        attention_output_5 = layer_output_19 = None
        layer_output_20 = torch.nn.functional.layer_norm(
            hidden_states_17,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_norm2_parameters_bias_ = (None)
        hidden_state_5 = torch._C._nn.linear(
            layer_output_20,
            l_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_20 = l_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_5 = hidden_state_5.chunk(2, dim=-1)
        hidden_state_5 = None
        x1_5 = chunk_5[0]
        x2_5 = chunk_5[1]
        chunk_5 = None
        silu_5 = torch.nn.functional.silu(x1_5)
        x1_5 = None
        hidden_5 = silu_5 * x2_5
        silu_5 = x2_5 = None
        layer_output_21 = torch._C._nn.linear(
            hidden_5,
            l_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_5 = l_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_22 = (
            layer_output_21
            * l_self_modules_encoder_modules_layer_modules_5_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_21 = l_self_modules_encoder_modules_layer_modules_5_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_23 = layer_output_22 + hidden_states_17
        layer_output_22 = hidden_states_17 = None
        layer_norm_12 = torch.nn.functional.layer_norm(
            layer_output_23,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_norm1_parameters_bias_ = (None)
        linear_36 = torch._C._nn.linear(
            layer_norm_12,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_19 = linear_36.view(1, -1, 24, 64)
        linear_36 = None
        key_layer_6 = view_19.transpose(1, 2)
        view_19 = None
        linear_37 = torch._C._nn.linear(
            layer_norm_12,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_20 = linear_37.view(1, -1, 24, 64)
        linear_37 = None
        value_layer_6 = view_20.transpose(1, 2)
        view_20 = None
        linear_38 = torch._C._nn.linear(
            layer_norm_12,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_12 = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_21 = linear_38.view(1, -1, 24, 64)
        linear_38 = None
        query_layer_6 = view_21.transpose(1, 2)
        view_21 = None
        query_6 = query_layer_6.contiguous()
        query_layer_6 = None
        key_6 = key_layer_6.contiguous()
        key_layer_6 = None
        value_6 = value_layer_6.contiguous()
        value_layer_6 = None
        attn_output_12 = torch._C._nn.scaled_dot_product_attention(
            query_6,
            key_6,
            value_6,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_6 = key_6 = value_6 = None
        transpose_28 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_28.contiguous()
        transpose_28 = None
        context_layer_6 = attn_output_13.reshape((1, 257, 1536))
        attn_output_13 = None
        hidden_states_18 = torch._C._nn.linear(
            context_layer_6,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_6 = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_19 = torch.nn.functional.dropout(
            hidden_states_18, 0.0, False, False
        )
        hidden_states_18 = None
        attention_output_6 = (
            hidden_states_19
            * l_self_modules_encoder_modules_layer_modules_6_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_19 = l_self_modules_encoder_modules_layer_modules_6_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_20 = attention_output_6 + layer_output_23
        attention_output_6 = layer_output_23 = None
        layer_output_24 = torch.nn.functional.layer_norm(
            hidden_states_20,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_norm2_parameters_bias_ = (None)
        hidden_state_6 = torch._C._nn.linear(
            layer_output_24,
            l_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_24 = l_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_6 = hidden_state_6.chunk(2, dim=-1)
        hidden_state_6 = None
        x1_6 = chunk_6[0]
        x2_6 = chunk_6[1]
        chunk_6 = None
        silu_6 = torch.nn.functional.silu(x1_6)
        x1_6 = None
        hidden_6 = silu_6 * x2_6
        silu_6 = x2_6 = None
        layer_output_25 = torch._C._nn.linear(
            hidden_6,
            l_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_6 = l_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_26 = (
            layer_output_25
            * l_self_modules_encoder_modules_layer_modules_6_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_25 = l_self_modules_encoder_modules_layer_modules_6_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_27 = layer_output_26 + hidden_states_20
        layer_output_26 = hidden_states_20 = None
        layer_norm_14 = torch.nn.functional.layer_norm(
            layer_output_27,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_norm1_parameters_bias_ = (None)
        linear_42 = torch._C._nn.linear(
            layer_norm_14,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_22 = linear_42.view(1, -1, 24, 64)
        linear_42 = None
        key_layer_7 = view_22.transpose(1, 2)
        view_22 = None
        linear_43 = torch._C._nn.linear(
            layer_norm_14,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_23 = linear_43.view(1, -1, 24, 64)
        linear_43 = None
        value_layer_7 = view_23.transpose(1, 2)
        view_23 = None
        linear_44 = torch._C._nn.linear(
            layer_norm_14,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_14 = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_24 = linear_44.view(1, -1, 24, 64)
        linear_44 = None
        query_layer_7 = view_24.transpose(1, 2)
        view_24 = None
        query_7 = query_layer_7.contiguous()
        query_layer_7 = None
        key_7 = key_layer_7.contiguous()
        key_layer_7 = None
        value_7 = value_layer_7.contiguous()
        value_layer_7 = None
        attn_output_14 = torch._C._nn.scaled_dot_product_attention(
            query_7,
            key_7,
            value_7,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_7 = key_7 = value_7 = None
        transpose_32 = attn_output_14.transpose(1, 2)
        attn_output_14 = None
        attn_output_15 = transpose_32.contiguous()
        transpose_32 = None
        context_layer_7 = attn_output_15.reshape((1, 257, 1536))
        attn_output_15 = None
        hidden_states_21 = torch._C._nn.linear(
            context_layer_7,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_7 = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_22 = torch.nn.functional.dropout(
            hidden_states_21, 0.0, False, False
        )
        hidden_states_21 = None
        attention_output_7 = (
            hidden_states_22
            * l_self_modules_encoder_modules_layer_modules_7_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_22 = l_self_modules_encoder_modules_layer_modules_7_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_23 = attention_output_7 + layer_output_27
        attention_output_7 = layer_output_27 = None
        layer_output_28 = torch.nn.functional.layer_norm(
            hidden_states_23,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_norm2_parameters_bias_ = (None)
        hidden_state_7 = torch._C._nn.linear(
            layer_output_28,
            l_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_28 = l_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_7 = hidden_state_7.chunk(2, dim=-1)
        hidden_state_7 = None
        x1_7 = chunk_7[0]
        x2_7 = chunk_7[1]
        chunk_7 = None
        silu_7 = torch.nn.functional.silu(x1_7)
        x1_7 = None
        hidden_7 = silu_7 * x2_7
        silu_7 = x2_7 = None
        layer_output_29 = torch._C._nn.linear(
            hidden_7,
            l_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_7 = l_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_30 = (
            layer_output_29
            * l_self_modules_encoder_modules_layer_modules_7_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_29 = l_self_modules_encoder_modules_layer_modules_7_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_31 = layer_output_30 + hidden_states_23
        layer_output_30 = hidden_states_23 = None
        layer_norm_16 = torch.nn.functional.layer_norm(
            layer_output_31,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_norm1_parameters_bias_ = (None)
        linear_48 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_25 = linear_48.view(1, -1, 24, 64)
        linear_48 = None
        key_layer_8 = view_25.transpose(1, 2)
        view_25 = None
        linear_49 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_26 = linear_49.view(1, -1, 24, 64)
        linear_49 = None
        value_layer_8 = view_26.transpose(1, 2)
        view_26 = None
        linear_50 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_16 = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_27 = linear_50.view(1, -1, 24, 64)
        linear_50 = None
        query_layer_8 = view_27.transpose(1, 2)
        view_27 = None
        query_8 = query_layer_8.contiguous()
        query_layer_8 = None
        key_8 = key_layer_8.contiguous()
        key_layer_8 = None
        value_8 = value_layer_8.contiguous()
        value_layer_8 = None
        attn_output_16 = torch._C._nn.scaled_dot_product_attention(
            query_8,
            key_8,
            value_8,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_8 = key_8 = value_8 = None
        transpose_36 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_17 = transpose_36.contiguous()
        transpose_36 = None
        context_layer_8 = attn_output_17.reshape((1, 257, 1536))
        attn_output_17 = None
        hidden_states_24 = torch._C._nn.linear(
            context_layer_8,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_8 = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_25 = torch.nn.functional.dropout(
            hidden_states_24, 0.0, False, False
        )
        hidden_states_24 = None
        attention_output_8 = (
            hidden_states_25
            * l_self_modules_encoder_modules_layer_modules_8_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_25 = l_self_modules_encoder_modules_layer_modules_8_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_26 = attention_output_8 + layer_output_31
        attention_output_8 = layer_output_31 = None
        layer_output_32 = torch.nn.functional.layer_norm(
            hidden_states_26,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_norm2_parameters_bias_ = (None)
        hidden_state_8 = torch._C._nn.linear(
            layer_output_32,
            l_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_32 = l_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_8 = hidden_state_8.chunk(2, dim=-1)
        hidden_state_8 = None
        x1_8 = chunk_8[0]
        x2_8 = chunk_8[1]
        chunk_8 = None
        silu_8 = torch.nn.functional.silu(x1_8)
        x1_8 = None
        hidden_8 = silu_8 * x2_8
        silu_8 = x2_8 = None
        layer_output_33 = torch._C._nn.linear(
            hidden_8,
            l_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_8 = l_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_34 = (
            layer_output_33
            * l_self_modules_encoder_modules_layer_modules_8_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_33 = l_self_modules_encoder_modules_layer_modules_8_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_35 = layer_output_34 + hidden_states_26
        layer_output_34 = hidden_states_26 = None
        layer_norm_18 = torch.nn.functional.layer_norm(
            layer_output_35,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_norm1_parameters_bias_ = (None)
        linear_54 = torch._C._nn.linear(
            layer_norm_18,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_28 = linear_54.view(1, -1, 24, 64)
        linear_54 = None
        key_layer_9 = view_28.transpose(1, 2)
        view_28 = None
        linear_55 = torch._C._nn.linear(
            layer_norm_18,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_29 = linear_55.view(1, -1, 24, 64)
        linear_55 = None
        value_layer_9 = view_29.transpose(1, 2)
        view_29 = None
        linear_56 = torch._C._nn.linear(
            layer_norm_18,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_18 = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_30 = linear_56.view(1, -1, 24, 64)
        linear_56 = None
        query_layer_9 = view_30.transpose(1, 2)
        view_30 = None
        query_9 = query_layer_9.contiguous()
        query_layer_9 = None
        key_9 = key_layer_9.contiguous()
        key_layer_9 = None
        value_9 = value_layer_9.contiguous()
        value_layer_9 = None
        attn_output_18 = torch._C._nn.scaled_dot_product_attention(
            query_9,
            key_9,
            value_9,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_9 = key_9 = value_9 = None
        transpose_40 = attn_output_18.transpose(1, 2)
        attn_output_18 = None
        attn_output_19 = transpose_40.contiguous()
        transpose_40 = None
        context_layer_9 = attn_output_19.reshape((1, 257, 1536))
        attn_output_19 = None
        hidden_states_27 = torch._C._nn.linear(
            context_layer_9,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_9 = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_28 = torch.nn.functional.dropout(
            hidden_states_27, 0.0, False, False
        )
        hidden_states_27 = None
        attention_output_9 = (
            hidden_states_28
            * l_self_modules_encoder_modules_layer_modules_9_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_28 = l_self_modules_encoder_modules_layer_modules_9_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_29 = attention_output_9 + layer_output_35
        attention_output_9 = layer_output_35 = None
        layer_output_36 = torch.nn.functional.layer_norm(
            hidden_states_29,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_norm2_parameters_bias_ = (None)
        hidden_state_9 = torch._C._nn.linear(
            layer_output_36,
            l_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_36 = l_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_9 = hidden_state_9.chunk(2, dim=-1)
        hidden_state_9 = None
        x1_9 = chunk_9[0]
        x2_9 = chunk_9[1]
        chunk_9 = None
        silu_9 = torch.nn.functional.silu(x1_9)
        x1_9 = None
        hidden_9 = silu_9 * x2_9
        silu_9 = x2_9 = None
        layer_output_37 = torch._C._nn.linear(
            hidden_9,
            l_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_9 = l_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_38 = (
            layer_output_37
            * l_self_modules_encoder_modules_layer_modules_9_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_37 = l_self_modules_encoder_modules_layer_modules_9_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_39 = layer_output_38 + hidden_states_29
        layer_output_38 = hidden_states_29 = None
        layer_norm_20 = torch.nn.functional.layer_norm(
            layer_output_39,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_norm1_parameters_bias_ = (None)
        linear_60 = torch._C._nn.linear(
            layer_norm_20,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_31 = linear_60.view(1, -1, 24, 64)
        linear_60 = None
        key_layer_10 = view_31.transpose(1, 2)
        view_31 = None
        linear_61 = torch._C._nn.linear(
            layer_norm_20,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_32 = linear_61.view(1, -1, 24, 64)
        linear_61 = None
        value_layer_10 = view_32.transpose(1, 2)
        view_32 = None
        linear_62 = torch._C._nn.linear(
            layer_norm_20,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_20 = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_33 = linear_62.view(1, -1, 24, 64)
        linear_62 = None
        query_layer_10 = view_33.transpose(1, 2)
        view_33 = None
        query_10 = query_layer_10.contiguous()
        query_layer_10 = None
        key_10 = key_layer_10.contiguous()
        key_layer_10 = None
        value_10 = value_layer_10.contiguous()
        value_layer_10 = None
        attn_output_20 = torch._C._nn.scaled_dot_product_attention(
            query_10,
            key_10,
            value_10,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_10 = key_10 = value_10 = None
        transpose_44 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_44.contiguous()
        transpose_44 = None
        context_layer_10 = attn_output_21.reshape((1, 257, 1536))
        attn_output_21 = None
        hidden_states_30 = torch._C._nn.linear(
            context_layer_10,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_10 = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_31 = torch.nn.functional.dropout(
            hidden_states_30, 0.0, False, False
        )
        hidden_states_30 = None
        attention_output_10 = (
            hidden_states_31
            * l_self_modules_encoder_modules_layer_modules_10_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_31 = l_self_modules_encoder_modules_layer_modules_10_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_32 = attention_output_10 + layer_output_39
        attention_output_10 = layer_output_39 = None
        layer_output_40 = torch.nn.functional.layer_norm(
            hidden_states_32,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_norm2_parameters_bias_ = (None)
        hidden_state_10 = torch._C._nn.linear(
            layer_output_40,
            l_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_40 = l_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_10 = hidden_state_10.chunk(2, dim=-1)
        hidden_state_10 = None
        x1_10 = chunk_10[0]
        x2_10 = chunk_10[1]
        chunk_10 = None
        silu_10 = torch.nn.functional.silu(x1_10)
        x1_10 = None
        hidden_10 = silu_10 * x2_10
        silu_10 = x2_10 = None
        layer_output_41 = torch._C._nn.linear(
            hidden_10,
            l_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_10 = l_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_42 = (
            layer_output_41
            * l_self_modules_encoder_modules_layer_modules_10_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_41 = l_self_modules_encoder_modules_layer_modules_10_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_43 = layer_output_42 + hidden_states_32
        layer_output_42 = hidden_states_32 = None
        layer_norm_22 = torch.nn.functional.layer_norm(
            layer_output_43,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_norm1_parameters_bias_ = (None)
        linear_66 = torch._C._nn.linear(
            layer_norm_22,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_34 = linear_66.view(1, -1, 24, 64)
        linear_66 = None
        key_layer_11 = view_34.transpose(1, 2)
        view_34 = None
        linear_67 = torch._C._nn.linear(
            layer_norm_22,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_35 = linear_67.view(1, -1, 24, 64)
        linear_67 = None
        value_layer_11 = view_35.transpose(1, 2)
        view_35 = None
        linear_68 = torch._C._nn.linear(
            layer_norm_22,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_22 = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_36 = linear_68.view(1, -1, 24, 64)
        linear_68 = None
        query_layer_11 = view_36.transpose(1, 2)
        view_36 = None
        query_11 = query_layer_11.contiguous()
        query_layer_11 = None
        key_11 = key_layer_11.contiguous()
        key_layer_11 = None
        value_11 = value_layer_11.contiguous()
        value_layer_11 = None
        attn_output_22 = torch._C._nn.scaled_dot_product_attention(
            query_11,
            key_11,
            value_11,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_11 = key_11 = value_11 = None
        transpose_48 = attn_output_22.transpose(1, 2)
        attn_output_22 = None
        attn_output_23 = transpose_48.contiguous()
        transpose_48 = None
        context_layer_11 = attn_output_23.reshape((1, 257, 1536))
        attn_output_23 = None
        hidden_states_33 = torch._C._nn.linear(
            context_layer_11,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_11 = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_34 = torch.nn.functional.dropout(
            hidden_states_33, 0.0, False, False
        )
        hidden_states_33 = None
        attention_output_11 = (
            hidden_states_34
            * l_self_modules_encoder_modules_layer_modules_11_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_34 = l_self_modules_encoder_modules_layer_modules_11_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_35 = attention_output_11 + layer_output_43
        attention_output_11 = layer_output_43 = None
        layer_output_44 = torch.nn.functional.layer_norm(
            hidden_states_35,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_norm2_parameters_bias_ = (None)
        hidden_state_11 = torch._C._nn.linear(
            layer_output_44,
            l_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_44 = l_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_11 = hidden_state_11.chunk(2, dim=-1)
        hidden_state_11 = None
        x1_11 = chunk_11[0]
        x2_11 = chunk_11[1]
        chunk_11 = None
        silu_11 = torch.nn.functional.silu(x1_11)
        x1_11 = None
        hidden_11 = silu_11 * x2_11
        silu_11 = x2_11 = None
        layer_output_45 = torch._C._nn.linear(
            hidden_11,
            l_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_11 = l_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_46 = (
            layer_output_45
            * l_self_modules_encoder_modules_layer_modules_11_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_45 = l_self_modules_encoder_modules_layer_modules_11_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_47 = layer_output_46 + hidden_states_35
        layer_output_46 = hidden_states_35 = None
        layer_norm_24 = torch.nn.functional.layer_norm(
            layer_output_47,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_12_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_12_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_norm1_parameters_bias_ = (None)
        linear_72 = torch._C._nn.linear(
            layer_norm_24,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_37 = linear_72.view(1, -1, 24, 64)
        linear_72 = None
        key_layer_12 = view_37.transpose(1, 2)
        view_37 = None
        linear_73 = torch._C._nn.linear(
            layer_norm_24,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_38 = linear_73.view(1, -1, 24, 64)
        linear_73 = None
        value_layer_12 = view_38.transpose(1, 2)
        view_38 = None
        linear_74 = torch._C._nn.linear(
            layer_norm_24,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_24 = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_39 = linear_74.view(1, -1, 24, 64)
        linear_74 = None
        query_layer_12 = view_39.transpose(1, 2)
        view_39 = None
        query_12 = query_layer_12.contiguous()
        query_layer_12 = None
        key_12 = key_layer_12.contiguous()
        key_layer_12 = None
        value_12 = value_layer_12.contiguous()
        value_layer_12 = None
        attn_output_24 = torch._C._nn.scaled_dot_product_attention(
            query_12,
            key_12,
            value_12,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_12 = key_12 = value_12 = None
        transpose_52 = attn_output_24.transpose(1, 2)
        attn_output_24 = None
        attn_output_25 = transpose_52.contiguous()
        transpose_52 = None
        context_layer_12 = attn_output_25.reshape((1, 257, 1536))
        attn_output_25 = None
        hidden_states_36 = torch._C._nn.linear(
            context_layer_12,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_12 = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_37 = torch.nn.functional.dropout(
            hidden_states_36, 0.0, False, False
        )
        hidden_states_36 = None
        attention_output_12 = (
            hidden_states_37
            * l_self_modules_encoder_modules_layer_modules_12_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_37 = l_self_modules_encoder_modules_layer_modules_12_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_38 = attention_output_12 + layer_output_47
        attention_output_12 = layer_output_47 = None
        layer_output_48 = torch.nn.functional.layer_norm(
            hidden_states_38,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_12_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_12_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_norm2_parameters_bias_ = (None)
        hidden_state_12 = torch._C._nn.linear(
            layer_output_48,
            l_self_modules_encoder_modules_layer_modules_12_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_48 = l_self_modules_encoder_modules_layer_modules_12_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_12 = hidden_state_12.chunk(2, dim=-1)
        hidden_state_12 = None
        x1_12 = chunk_12[0]
        x2_12 = chunk_12[1]
        chunk_12 = None
        silu_12 = torch.nn.functional.silu(x1_12)
        x1_12 = None
        hidden_12 = silu_12 * x2_12
        silu_12 = x2_12 = None
        layer_output_49 = torch._C._nn.linear(
            hidden_12,
            l_self_modules_encoder_modules_layer_modules_12_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_12 = l_self_modules_encoder_modules_layer_modules_12_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_50 = (
            layer_output_49
            * l_self_modules_encoder_modules_layer_modules_12_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_49 = l_self_modules_encoder_modules_layer_modules_12_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_51 = layer_output_50 + hidden_states_38
        layer_output_50 = hidden_states_38 = None
        layer_norm_26 = torch.nn.functional.layer_norm(
            layer_output_51,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_13_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_13_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_norm1_parameters_bias_ = (None)
        linear_78 = torch._C._nn.linear(
            layer_norm_26,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_40 = linear_78.view(1, -1, 24, 64)
        linear_78 = None
        key_layer_13 = view_40.transpose(1, 2)
        view_40 = None
        linear_79 = torch._C._nn.linear(
            layer_norm_26,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_41 = linear_79.view(1, -1, 24, 64)
        linear_79 = None
        value_layer_13 = view_41.transpose(1, 2)
        view_41 = None
        linear_80 = torch._C._nn.linear(
            layer_norm_26,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_26 = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_42 = linear_80.view(1, -1, 24, 64)
        linear_80 = None
        query_layer_13 = view_42.transpose(1, 2)
        view_42 = None
        query_13 = query_layer_13.contiguous()
        query_layer_13 = None
        key_13 = key_layer_13.contiguous()
        key_layer_13 = None
        value_13 = value_layer_13.contiguous()
        value_layer_13 = None
        attn_output_26 = torch._C._nn.scaled_dot_product_attention(
            query_13,
            key_13,
            value_13,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_13 = key_13 = value_13 = None
        transpose_56 = attn_output_26.transpose(1, 2)
        attn_output_26 = None
        attn_output_27 = transpose_56.contiguous()
        transpose_56 = None
        context_layer_13 = attn_output_27.reshape((1, 257, 1536))
        attn_output_27 = None
        hidden_states_39 = torch._C._nn.linear(
            context_layer_13,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_13 = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_40 = torch.nn.functional.dropout(
            hidden_states_39, 0.0, False, False
        )
        hidden_states_39 = None
        attention_output_13 = (
            hidden_states_40
            * l_self_modules_encoder_modules_layer_modules_13_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_40 = l_self_modules_encoder_modules_layer_modules_13_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_41 = attention_output_13 + layer_output_51
        attention_output_13 = layer_output_51 = None
        layer_output_52 = torch.nn.functional.layer_norm(
            hidden_states_41,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_13_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_13_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_norm2_parameters_bias_ = (None)
        hidden_state_13 = torch._C._nn.linear(
            layer_output_52,
            l_self_modules_encoder_modules_layer_modules_13_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_52 = l_self_modules_encoder_modules_layer_modules_13_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_13 = hidden_state_13.chunk(2, dim=-1)
        hidden_state_13 = None
        x1_13 = chunk_13[0]
        x2_13 = chunk_13[1]
        chunk_13 = None
        silu_13 = torch.nn.functional.silu(x1_13)
        x1_13 = None
        hidden_13 = silu_13 * x2_13
        silu_13 = x2_13 = None
        layer_output_53 = torch._C._nn.linear(
            hidden_13,
            l_self_modules_encoder_modules_layer_modules_13_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_13 = l_self_modules_encoder_modules_layer_modules_13_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_54 = (
            layer_output_53
            * l_self_modules_encoder_modules_layer_modules_13_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_53 = l_self_modules_encoder_modules_layer_modules_13_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_55 = layer_output_54 + hidden_states_41
        layer_output_54 = hidden_states_41 = None
        layer_norm_28 = torch.nn.functional.layer_norm(
            layer_output_55,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_14_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_14_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_norm1_parameters_bias_ = (None)
        linear_84 = torch._C._nn.linear(
            layer_norm_28,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_43 = linear_84.view(1, -1, 24, 64)
        linear_84 = None
        key_layer_14 = view_43.transpose(1, 2)
        view_43 = None
        linear_85 = torch._C._nn.linear(
            layer_norm_28,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_44 = linear_85.view(1, -1, 24, 64)
        linear_85 = None
        value_layer_14 = view_44.transpose(1, 2)
        view_44 = None
        linear_86 = torch._C._nn.linear(
            layer_norm_28,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_28 = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_45 = linear_86.view(1, -1, 24, 64)
        linear_86 = None
        query_layer_14 = view_45.transpose(1, 2)
        view_45 = None
        query_14 = query_layer_14.contiguous()
        query_layer_14 = None
        key_14 = key_layer_14.contiguous()
        key_layer_14 = None
        value_14 = value_layer_14.contiguous()
        value_layer_14 = None
        attn_output_28 = torch._C._nn.scaled_dot_product_attention(
            query_14,
            key_14,
            value_14,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_14 = key_14 = value_14 = None
        transpose_60 = attn_output_28.transpose(1, 2)
        attn_output_28 = None
        attn_output_29 = transpose_60.contiguous()
        transpose_60 = None
        context_layer_14 = attn_output_29.reshape((1, 257, 1536))
        attn_output_29 = None
        hidden_states_42 = torch._C._nn.linear(
            context_layer_14,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_14 = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_43 = torch.nn.functional.dropout(
            hidden_states_42, 0.0, False, False
        )
        hidden_states_42 = None
        attention_output_14 = (
            hidden_states_43
            * l_self_modules_encoder_modules_layer_modules_14_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_43 = l_self_modules_encoder_modules_layer_modules_14_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_44 = attention_output_14 + layer_output_55
        attention_output_14 = layer_output_55 = None
        layer_output_56 = torch.nn.functional.layer_norm(
            hidden_states_44,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_14_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_14_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_norm2_parameters_bias_ = (None)
        hidden_state_14 = torch._C._nn.linear(
            layer_output_56,
            l_self_modules_encoder_modules_layer_modules_14_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_56 = l_self_modules_encoder_modules_layer_modules_14_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_14 = hidden_state_14.chunk(2, dim=-1)
        hidden_state_14 = None
        x1_14 = chunk_14[0]
        x2_14 = chunk_14[1]
        chunk_14 = None
        silu_14 = torch.nn.functional.silu(x1_14)
        x1_14 = None
        hidden_14 = silu_14 * x2_14
        silu_14 = x2_14 = None
        layer_output_57 = torch._C._nn.linear(
            hidden_14,
            l_self_modules_encoder_modules_layer_modules_14_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_14 = l_self_modules_encoder_modules_layer_modules_14_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_58 = (
            layer_output_57
            * l_self_modules_encoder_modules_layer_modules_14_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_57 = l_self_modules_encoder_modules_layer_modules_14_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_59 = layer_output_58 + hidden_states_44
        layer_output_58 = hidden_states_44 = None
        layer_norm_30 = torch.nn.functional.layer_norm(
            layer_output_59,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_15_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_15_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_norm1_parameters_bias_ = (None)
        linear_90 = torch._C._nn.linear(
            layer_norm_30,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_46 = linear_90.view(1, -1, 24, 64)
        linear_90 = None
        key_layer_15 = view_46.transpose(1, 2)
        view_46 = None
        linear_91 = torch._C._nn.linear(
            layer_norm_30,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_47 = linear_91.view(1, -1, 24, 64)
        linear_91 = None
        value_layer_15 = view_47.transpose(1, 2)
        view_47 = None
        linear_92 = torch._C._nn.linear(
            layer_norm_30,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_30 = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_48 = linear_92.view(1, -1, 24, 64)
        linear_92 = None
        query_layer_15 = view_48.transpose(1, 2)
        view_48 = None
        query_15 = query_layer_15.contiguous()
        query_layer_15 = None
        key_15 = key_layer_15.contiguous()
        key_layer_15 = None
        value_15 = value_layer_15.contiguous()
        value_layer_15 = None
        attn_output_30 = torch._C._nn.scaled_dot_product_attention(
            query_15,
            key_15,
            value_15,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_15 = key_15 = value_15 = None
        transpose_64 = attn_output_30.transpose(1, 2)
        attn_output_30 = None
        attn_output_31 = transpose_64.contiguous()
        transpose_64 = None
        context_layer_15 = attn_output_31.reshape((1, 257, 1536))
        attn_output_31 = None
        hidden_states_45 = torch._C._nn.linear(
            context_layer_15,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_15 = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_46 = torch.nn.functional.dropout(
            hidden_states_45, 0.0, False, False
        )
        hidden_states_45 = None
        attention_output_15 = (
            hidden_states_46
            * l_self_modules_encoder_modules_layer_modules_15_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_46 = l_self_modules_encoder_modules_layer_modules_15_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_47 = attention_output_15 + layer_output_59
        attention_output_15 = layer_output_59 = None
        layer_output_60 = torch.nn.functional.layer_norm(
            hidden_states_47,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_15_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_15_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_norm2_parameters_bias_ = (None)
        hidden_state_15 = torch._C._nn.linear(
            layer_output_60,
            l_self_modules_encoder_modules_layer_modules_15_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_60 = l_self_modules_encoder_modules_layer_modules_15_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_15 = hidden_state_15.chunk(2, dim=-1)
        hidden_state_15 = None
        x1_15 = chunk_15[0]
        x2_15 = chunk_15[1]
        chunk_15 = None
        silu_15 = torch.nn.functional.silu(x1_15)
        x1_15 = None
        hidden_15 = silu_15 * x2_15
        silu_15 = x2_15 = None
        layer_output_61 = torch._C._nn.linear(
            hidden_15,
            l_self_modules_encoder_modules_layer_modules_15_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_15 = l_self_modules_encoder_modules_layer_modules_15_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_62 = (
            layer_output_61
            * l_self_modules_encoder_modules_layer_modules_15_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_61 = l_self_modules_encoder_modules_layer_modules_15_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_63 = layer_output_62 + hidden_states_47
        layer_output_62 = hidden_states_47 = None
        layer_norm_32 = torch.nn.functional.layer_norm(
            layer_output_63,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_16_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_16_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_norm1_parameters_bias_ = (None)
        linear_96 = torch._C._nn.linear(
            layer_norm_32,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_49 = linear_96.view(1, -1, 24, 64)
        linear_96 = None
        key_layer_16 = view_49.transpose(1, 2)
        view_49 = None
        linear_97 = torch._C._nn.linear(
            layer_norm_32,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_50 = linear_97.view(1, -1, 24, 64)
        linear_97 = None
        value_layer_16 = view_50.transpose(1, 2)
        view_50 = None
        linear_98 = torch._C._nn.linear(
            layer_norm_32,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_32 = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_51 = linear_98.view(1, -1, 24, 64)
        linear_98 = None
        query_layer_16 = view_51.transpose(1, 2)
        view_51 = None
        query_16 = query_layer_16.contiguous()
        query_layer_16 = None
        key_16 = key_layer_16.contiguous()
        key_layer_16 = None
        value_16 = value_layer_16.contiguous()
        value_layer_16 = None
        attn_output_32 = torch._C._nn.scaled_dot_product_attention(
            query_16,
            key_16,
            value_16,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_16 = key_16 = value_16 = None
        transpose_68 = attn_output_32.transpose(1, 2)
        attn_output_32 = None
        attn_output_33 = transpose_68.contiguous()
        transpose_68 = None
        context_layer_16 = attn_output_33.reshape((1, 257, 1536))
        attn_output_33 = None
        hidden_states_48 = torch._C._nn.linear(
            context_layer_16,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_16 = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_49 = torch.nn.functional.dropout(
            hidden_states_48, 0.0, False, False
        )
        hidden_states_48 = None
        attention_output_16 = (
            hidden_states_49
            * l_self_modules_encoder_modules_layer_modules_16_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_49 = l_self_modules_encoder_modules_layer_modules_16_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_50 = attention_output_16 + layer_output_63
        attention_output_16 = layer_output_63 = None
        layer_output_64 = torch.nn.functional.layer_norm(
            hidden_states_50,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_16_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_16_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_norm2_parameters_bias_ = (None)
        hidden_state_16 = torch._C._nn.linear(
            layer_output_64,
            l_self_modules_encoder_modules_layer_modules_16_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_64 = l_self_modules_encoder_modules_layer_modules_16_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_16 = hidden_state_16.chunk(2, dim=-1)
        hidden_state_16 = None
        x1_16 = chunk_16[0]
        x2_16 = chunk_16[1]
        chunk_16 = None
        silu_16 = torch.nn.functional.silu(x1_16)
        x1_16 = None
        hidden_16 = silu_16 * x2_16
        silu_16 = x2_16 = None
        layer_output_65 = torch._C._nn.linear(
            hidden_16,
            l_self_modules_encoder_modules_layer_modules_16_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_16 = l_self_modules_encoder_modules_layer_modules_16_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_66 = (
            layer_output_65
            * l_self_modules_encoder_modules_layer_modules_16_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_65 = l_self_modules_encoder_modules_layer_modules_16_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_67 = layer_output_66 + hidden_states_50
        layer_output_66 = hidden_states_50 = None
        layer_norm_34 = torch.nn.functional.layer_norm(
            layer_output_67,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_17_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_17_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_norm1_parameters_bias_ = (None)
        linear_102 = torch._C._nn.linear(
            layer_norm_34,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_52 = linear_102.view(1, -1, 24, 64)
        linear_102 = None
        key_layer_17 = view_52.transpose(1, 2)
        view_52 = None
        linear_103 = torch._C._nn.linear(
            layer_norm_34,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_53 = linear_103.view(1, -1, 24, 64)
        linear_103 = None
        value_layer_17 = view_53.transpose(1, 2)
        view_53 = None
        linear_104 = torch._C._nn.linear(
            layer_norm_34,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_34 = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_54 = linear_104.view(1, -1, 24, 64)
        linear_104 = None
        query_layer_17 = view_54.transpose(1, 2)
        view_54 = None
        query_17 = query_layer_17.contiguous()
        query_layer_17 = None
        key_17 = key_layer_17.contiguous()
        key_layer_17 = None
        value_17 = value_layer_17.contiguous()
        value_layer_17 = None
        attn_output_34 = torch._C._nn.scaled_dot_product_attention(
            query_17,
            key_17,
            value_17,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_17 = key_17 = value_17 = None
        transpose_72 = attn_output_34.transpose(1, 2)
        attn_output_34 = None
        attn_output_35 = transpose_72.contiguous()
        transpose_72 = None
        context_layer_17 = attn_output_35.reshape((1, 257, 1536))
        attn_output_35 = None
        hidden_states_51 = torch._C._nn.linear(
            context_layer_17,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_17 = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_52 = torch.nn.functional.dropout(
            hidden_states_51, 0.0, False, False
        )
        hidden_states_51 = None
        attention_output_17 = (
            hidden_states_52
            * l_self_modules_encoder_modules_layer_modules_17_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_52 = l_self_modules_encoder_modules_layer_modules_17_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_53 = attention_output_17 + layer_output_67
        attention_output_17 = layer_output_67 = None
        layer_output_68 = torch.nn.functional.layer_norm(
            hidden_states_53,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_17_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_17_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_norm2_parameters_bias_ = (None)
        hidden_state_17 = torch._C._nn.linear(
            layer_output_68,
            l_self_modules_encoder_modules_layer_modules_17_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_68 = l_self_modules_encoder_modules_layer_modules_17_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_17 = hidden_state_17.chunk(2, dim=-1)
        hidden_state_17 = None
        x1_17 = chunk_17[0]
        x2_17 = chunk_17[1]
        chunk_17 = None
        silu_17 = torch.nn.functional.silu(x1_17)
        x1_17 = None
        hidden_17 = silu_17 * x2_17
        silu_17 = x2_17 = None
        layer_output_69 = torch._C._nn.linear(
            hidden_17,
            l_self_modules_encoder_modules_layer_modules_17_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_17 = l_self_modules_encoder_modules_layer_modules_17_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_70 = (
            layer_output_69
            * l_self_modules_encoder_modules_layer_modules_17_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_69 = l_self_modules_encoder_modules_layer_modules_17_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_71 = layer_output_70 + hidden_states_53
        layer_output_70 = hidden_states_53 = None
        layer_norm_36 = torch.nn.functional.layer_norm(
            layer_output_71,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_18_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_18_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_norm1_parameters_bias_ = (None)
        linear_108 = torch._C._nn.linear(
            layer_norm_36,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_55 = linear_108.view(1, -1, 24, 64)
        linear_108 = None
        key_layer_18 = view_55.transpose(1, 2)
        view_55 = None
        linear_109 = torch._C._nn.linear(
            layer_norm_36,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_56 = linear_109.view(1, -1, 24, 64)
        linear_109 = None
        value_layer_18 = view_56.transpose(1, 2)
        view_56 = None
        linear_110 = torch._C._nn.linear(
            layer_norm_36,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_36 = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_57 = linear_110.view(1, -1, 24, 64)
        linear_110 = None
        query_layer_18 = view_57.transpose(1, 2)
        view_57 = None
        query_18 = query_layer_18.contiguous()
        query_layer_18 = None
        key_18 = key_layer_18.contiguous()
        key_layer_18 = None
        value_18 = value_layer_18.contiguous()
        value_layer_18 = None
        attn_output_36 = torch._C._nn.scaled_dot_product_attention(
            query_18,
            key_18,
            value_18,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_18 = key_18 = value_18 = None
        transpose_76 = attn_output_36.transpose(1, 2)
        attn_output_36 = None
        attn_output_37 = transpose_76.contiguous()
        transpose_76 = None
        context_layer_18 = attn_output_37.reshape((1, 257, 1536))
        attn_output_37 = None
        hidden_states_54 = torch._C._nn.linear(
            context_layer_18,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_18 = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_55 = torch.nn.functional.dropout(
            hidden_states_54, 0.0, False, False
        )
        hidden_states_54 = None
        attention_output_18 = (
            hidden_states_55
            * l_self_modules_encoder_modules_layer_modules_18_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_55 = l_self_modules_encoder_modules_layer_modules_18_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_56 = attention_output_18 + layer_output_71
        attention_output_18 = layer_output_71 = None
        layer_output_72 = torch.nn.functional.layer_norm(
            hidden_states_56,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_18_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_18_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_norm2_parameters_bias_ = (None)
        hidden_state_18 = torch._C._nn.linear(
            layer_output_72,
            l_self_modules_encoder_modules_layer_modules_18_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_72 = l_self_modules_encoder_modules_layer_modules_18_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_18 = hidden_state_18.chunk(2, dim=-1)
        hidden_state_18 = None
        x1_18 = chunk_18[0]
        x2_18 = chunk_18[1]
        chunk_18 = None
        silu_18 = torch.nn.functional.silu(x1_18)
        x1_18 = None
        hidden_18 = silu_18 * x2_18
        silu_18 = x2_18 = None
        layer_output_73 = torch._C._nn.linear(
            hidden_18,
            l_self_modules_encoder_modules_layer_modules_18_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_18 = l_self_modules_encoder_modules_layer_modules_18_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_74 = (
            layer_output_73
            * l_self_modules_encoder_modules_layer_modules_18_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_73 = l_self_modules_encoder_modules_layer_modules_18_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_75 = layer_output_74 + hidden_states_56
        layer_output_74 = hidden_states_56 = None
        layer_norm_38 = torch.nn.functional.layer_norm(
            layer_output_75,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_19_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_19_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_norm1_parameters_bias_ = (None)
        linear_114 = torch._C._nn.linear(
            layer_norm_38,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_58 = linear_114.view(1, -1, 24, 64)
        linear_114 = None
        key_layer_19 = view_58.transpose(1, 2)
        view_58 = None
        linear_115 = torch._C._nn.linear(
            layer_norm_38,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_59 = linear_115.view(1, -1, 24, 64)
        linear_115 = None
        value_layer_19 = view_59.transpose(1, 2)
        view_59 = None
        linear_116 = torch._C._nn.linear(
            layer_norm_38,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_38 = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_60 = linear_116.view(1, -1, 24, 64)
        linear_116 = None
        query_layer_19 = view_60.transpose(1, 2)
        view_60 = None
        query_19 = query_layer_19.contiguous()
        query_layer_19 = None
        key_19 = key_layer_19.contiguous()
        key_layer_19 = None
        value_19 = value_layer_19.contiguous()
        value_layer_19 = None
        attn_output_38 = torch._C._nn.scaled_dot_product_attention(
            query_19,
            key_19,
            value_19,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_19 = key_19 = value_19 = None
        transpose_80 = attn_output_38.transpose(1, 2)
        attn_output_38 = None
        attn_output_39 = transpose_80.contiguous()
        transpose_80 = None
        context_layer_19 = attn_output_39.reshape((1, 257, 1536))
        attn_output_39 = None
        hidden_states_57 = torch._C._nn.linear(
            context_layer_19,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_19 = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_58 = torch.nn.functional.dropout(
            hidden_states_57, 0.0, False, False
        )
        hidden_states_57 = None
        attention_output_19 = (
            hidden_states_58
            * l_self_modules_encoder_modules_layer_modules_19_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_58 = l_self_modules_encoder_modules_layer_modules_19_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_59 = attention_output_19 + layer_output_75
        attention_output_19 = layer_output_75 = None
        layer_output_76 = torch.nn.functional.layer_norm(
            hidden_states_59,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_19_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_19_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_norm2_parameters_bias_ = (None)
        hidden_state_19 = torch._C._nn.linear(
            layer_output_76,
            l_self_modules_encoder_modules_layer_modules_19_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_76 = l_self_modules_encoder_modules_layer_modules_19_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_19 = hidden_state_19.chunk(2, dim=-1)
        hidden_state_19 = None
        x1_19 = chunk_19[0]
        x2_19 = chunk_19[1]
        chunk_19 = None
        silu_19 = torch.nn.functional.silu(x1_19)
        x1_19 = None
        hidden_19 = silu_19 * x2_19
        silu_19 = x2_19 = None
        layer_output_77 = torch._C._nn.linear(
            hidden_19,
            l_self_modules_encoder_modules_layer_modules_19_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_19 = l_self_modules_encoder_modules_layer_modules_19_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_78 = (
            layer_output_77
            * l_self_modules_encoder_modules_layer_modules_19_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_77 = l_self_modules_encoder_modules_layer_modules_19_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_79 = layer_output_78 + hidden_states_59
        layer_output_78 = hidden_states_59 = None
        layer_norm_40 = torch.nn.functional.layer_norm(
            layer_output_79,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_20_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_20_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_norm1_parameters_bias_ = (None)
        linear_120 = torch._C._nn.linear(
            layer_norm_40,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_61 = linear_120.view(1, -1, 24, 64)
        linear_120 = None
        key_layer_20 = view_61.transpose(1, 2)
        view_61 = None
        linear_121 = torch._C._nn.linear(
            layer_norm_40,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_62 = linear_121.view(1, -1, 24, 64)
        linear_121 = None
        value_layer_20 = view_62.transpose(1, 2)
        view_62 = None
        linear_122 = torch._C._nn.linear(
            layer_norm_40,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_40 = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_63 = linear_122.view(1, -1, 24, 64)
        linear_122 = None
        query_layer_20 = view_63.transpose(1, 2)
        view_63 = None
        query_20 = query_layer_20.contiguous()
        query_layer_20 = None
        key_20 = key_layer_20.contiguous()
        key_layer_20 = None
        value_20 = value_layer_20.contiguous()
        value_layer_20 = None
        attn_output_40 = torch._C._nn.scaled_dot_product_attention(
            query_20,
            key_20,
            value_20,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_20 = key_20 = value_20 = None
        transpose_84 = attn_output_40.transpose(1, 2)
        attn_output_40 = None
        attn_output_41 = transpose_84.contiguous()
        transpose_84 = None
        context_layer_20 = attn_output_41.reshape((1, 257, 1536))
        attn_output_41 = None
        hidden_states_60 = torch._C._nn.linear(
            context_layer_20,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_20 = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_61 = torch.nn.functional.dropout(
            hidden_states_60, 0.0, False, False
        )
        hidden_states_60 = None
        attention_output_20 = (
            hidden_states_61
            * l_self_modules_encoder_modules_layer_modules_20_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_61 = l_self_modules_encoder_modules_layer_modules_20_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_62 = attention_output_20 + layer_output_79
        attention_output_20 = layer_output_79 = None
        layer_output_80 = torch.nn.functional.layer_norm(
            hidden_states_62,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_20_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_20_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_norm2_parameters_bias_ = (None)
        hidden_state_20 = torch._C._nn.linear(
            layer_output_80,
            l_self_modules_encoder_modules_layer_modules_20_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_80 = l_self_modules_encoder_modules_layer_modules_20_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_20 = hidden_state_20.chunk(2, dim=-1)
        hidden_state_20 = None
        x1_20 = chunk_20[0]
        x2_20 = chunk_20[1]
        chunk_20 = None
        silu_20 = torch.nn.functional.silu(x1_20)
        x1_20 = None
        hidden_20 = silu_20 * x2_20
        silu_20 = x2_20 = None
        layer_output_81 = torch._C._nn.linear(
            hidden_20,
            l_self_modules_encoder_modules_layer_modules_20_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_20 = l_self_modules_encoder_modules_layer_modules_20_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_82 = (
            layer_output_81
            * l_self_modules_encoder_modules_layer_modules_20_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_81 = l_self_modules_encoder_modules_layer_modules_20_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_83 = layer_output_82 + hidden_states_62
        layer_output_82 = hidden_states_62 = None
        layer_norm_42 = torch.nn.functional.layer_norm(
            layer_output_83,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_21_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_21_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_norm1_parameters_bias_ = (None)
        linear_126 = torch._C._nn.linear(
            layer_norm_42,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_64 = linear_126.view(1, -1, 24, 64)
        linear_126 = None
        key_layer_21 = view_64.transpose(1, 2)
        view_64 = None
        linear_127 = torch._C._nn.linear(
            layer_norm_42,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_65 = linear_127.view(1, -1, 24, 64)
        linear_127 = None
        value_layer_21 = view_65.transpose(1, 2)
        view_65 = None
        linear_128 = torch._C._nn.linear(
            layer_norm_42,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_42 = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_66 = linear_128.view(1, -1, 24, 64)
        linear_128 = None
        query_layer_21 = view_66.transpose(1, 2)
        view_66 = None
        query_21 = query_layer_21.contiguous()
        query_layer_21 = None
        key_21 = key_layer_21.contiguous()
        key_layer_21 = None
        value_21 = value_layer_21.contiguous()
        value_layer_21 = None
        attn_output_42 = torch._C._nn.scaled_dot_product_attention(
            query_21,
            key_21,
            value_21,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_21 = key_21 = value_21 = None
        transpose_88 = attn_output_42.transpose(1, 2)
        attn_output_42 = None
        attn_output_43 = transpose_88.contiguous()
        transpose_88 = None
        context_layer_21 = attn_output_43.reshape((1, 257, 1536))
        attn_output_43 = None
        hidden_states_63 = torch._C._nn.linear(
            context_layer_21,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_21 = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_64 = torch.nn.functional.dropout(
            hidden_states_63, 0.0, False, False
        )
        hidden_states_63 = None
        attention_output_21 = (
            hidden_states_64
            * l_self_modules_encoder_modules_layer_modules_21_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_64 = l_self_modules_encoder_modules_layer_modules_21_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_65 = attention_output_21 + layer_output_83
        attention_output_21 = layer_output_83 = None
        layer_output_84 = torch.nn.functional.layer_norm(
            hidden_states_65,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_21_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_21_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_norm2_parameters_bias_ = (None)
        hidden_state_21 = torch._C._nn.linear(
            layer_output_84,
            l_self_modules_encoder_modules_layer_modules_21_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_84 = l_self_modules_encoder_modules_layer_modules_21_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_21 = hidden_state_21.chunk(2, dim=-1)
        hidden_state_21 = None
        x1_21 = chunk_21[0]
        x2_21 = chunk_21[1]
        chunk_21 = None
        silu_21 = torch.nn.functional.silu(x1_21)
        x1_21 = None
        hidden_21 = silu_21 * x2_21
        silu_21 = x2_21 = None
        layer_output_85 = torch._C._nn.linear(
            hidden_21,
            l_self_modules_encoder_modules_layer_modules_21_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_21 = l_self_modules_encoder_modules_layer_modules_21_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_86 = (
            layer_output_85
            * l_self_modules_encoder_modules_layer_modules_21_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_85 = l_self_modules_encoder_modules_layer_modules_21_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_87 = layer_output_86 + hidden_states_65
        layer_output_86 = hidden_states_65 = None
        layer_norm_44 = torch.nn.functional.layer_norm(
            layer_output_87,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_22_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_22_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_norm1_parameters_bias_ = (None)
        linear_132 = torch._C._nn.linear(
            layer_norm_44,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_67 = linear_132.view(1, -1, 24, 64)
        linear_132 = None
        key_layer_22 = view_67.transpose(1, 2)
        view_67 = None
        linear_133 = torch._C._nn.linear(
            layer_norm_44,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_68 = linear_133.view(1, -1, 24, 64)
        linear_133 = None
        value_layer_22 = view_68.transpose(1, 2)
        view_68 = None
        linear_134 = torch._C._nn.linear(
            layer_norm_44,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_44 = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_69 = linear_134.view(1, -1, 24, 64)
        linear_134 = None
        query_layer_22 = view_69.transpose(1, 2)
        view_69 = None
        query_22 = query_layer_22.contiguous()
        query_layer_22 = None
        key_22 = key_layer_22.contiguous()
        key_layer_22 = None
        value_22 = value_layer_22.contiguous()
        value_layer_22 = None
        attn_output_44 = torch._C._nn.scaled_dot_product_attention(
            query_22,
            key_22,
            value_22,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_22 = key_22 = value_22 = None
        transpose_92 = attn_output_44.transpose(1, 2)
        attn_output_44 = None
        attn_output_45 = transpose_92.contiguous()
        transpose_92 = None
        context_layer_22 = attn_output_45.reshape((1, 257, 1536))
        attn_output_45 = None
        hidden_states_66 = torch._C._nn.linear(
            context_layer_22,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_22 = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_67 = torch.nn.functional.dropout(
            hidden_states_66, 0.0, False, False
        )
        hidden_states_66 = None
        attention_output_22 = (
            hidden_states_67
            * l_self_modules_encoder_modules_layer_modules_22_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_67 = l_self_modules_encoder_modules_layer_modules_22_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_68 = attention_output_22 + layer_output_87
        attention_output_22 = layer_output_87 = None
        layer_output_88 = torch.nn.functional.layer_norm(
            hidden_states_68,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_22_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_22_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_norm2_parameters_bias_ = (None)
        hidden_state_22 = torch._C._nn.linear(
            layer_output_88,
            l_self_modules_encoder_modules_layer_modules_22_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_88 = l_self_modules_encoder_modules_layer_modules_22_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_22 = hidden_state_22.chunk(2, dim=-1)
        hidden_state_22 = None
        x1_22 = chunk_22[0]
        x2_22 = chunk_22[1]
        chunk_22 = None
        silu_22 = torch.nn.functional.silu(x1_22)
        x1_22 = None
        hidden_22 = silu_22 * x2_22
        silu_22 = x2_22 = None
        layer_output_89 = torch._C._nn.linear(
            hidden_22,
            l_self_modules_encoder_modules_layer_modules_22_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_22 = l_self_modules_encoder_modules_layer_modules_22_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_90 = (
            layer_output_89
            * l_self_modules_encoder_modules_layer_modules_22_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_89 = l_self_modules_encoder_modules_layer_modules_22_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_91 = layer_output_90 + hidden_states_68
        layer_output_90 = hidden_states_68 = None
        layer_norm_46 = torch.nn.functional.layer_norm(
            layer_output_91,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_23_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_23_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_norm1_parameters_bias_ = (None)
        linear_138 = torch._C._nn.linear(
            layer_norm_46,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_70 = linear_138.view(1, -1, 24, 64)
        linear_138 = None
        key_layer_23 = view_70.transpose(1, 2)
        view_70 = None
        linear_139 = torch._C._nn.linear(
            layer_norm_46,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_71 = linear_139.view(1, -1, 24, 64)
        linear_139 = None
        value_layer_23 = view_71.transpose(1, 2)
        view_71 = None
        linear_140 = torch._C._nn.linear(
            layer_norm_46,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_46 = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_72 = linear_140.view(1, -1, 24, 64)
        linear_140 = None
        query_layer_23 = view_72.transpose(1, 2)
        view_72 = None
        query_23 = query_layer_23.contiguous()
        query_layer_23 = None
        key_23 = key_layer_23.contiguous()
        key_layer_23 = None
        value_23 = value_layer_23.contiguous()
        value_layer_23 = None
        attn_output_46 = torch._C._nn.scaled_dot_product_attention(
            query_23,
            key_23,
            value_23,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_23 = key_23 = value_23 = None
        transpose_96 = attn_output_46.transpose(1, 2)
        attn_output_46 = None
        attn_output_47 = transpose_96.contiguous()
        transpose_96 = None
        context_layer_23 = attn_output_47.reshape((1, 257, 1536))
        attn_output_47 = None
        hidden_states_69 = torch._C._nn.linear(
            context_layer_23,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_23 = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_70 = torch.nn.functional.dropout(
            hidden_states_69, 0.0, False, False
        )
        hidden_states_69 = None
        attention_output_23 = (
            hidden_states_70
            * l_self_modules_encoder_modules_layer_modules_23_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_70 = l_self_modules_encoder_modules_layer_modules_23_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_71 = attention_output_23 + layer_output_91
        attention_output_23 = layer_output_91 = None
        layer_output_92 = torch.nn.functional.layer_norm(
            hidden_states_71,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_23_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_23_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_norm2_parameters_bias_ = (None)
        hidden_state_23 = torch._C._nn.linear(
            layer_output_92,
            l_self_modules_encoder_modules_layer_modules_23_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_92 = l_self_modules_encoder_modules_layer_modules_23_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_23 = hidden_state_23.chunk(2, dim=-1)
        hidden_state_23 = None
        x1_23 = chunk_23[0]
        x2_23 = chunk_23[1]
        chunk_23 = None
        silu_23 = torch.nn.functional.silu(x1_23)
        x1_23 = None
        hidden_23 = silu_23 * x2_23
        silu_23 = x2_23 = None
        layer_output_93 = torch._C._nn.linear(
            hidden_23,
            l_self_modules_encoder_modules_layer_modules_23_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_23 = l_self_modules_encoder_modules_layer_modules_23_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_94 = (
            layer_output_93
            * l_self_modules_encoder_modules_layer_modules_23_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_93 = l_self_modules_encoder_modules_layer_modules_23_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_95 = layer_output_94 + hidden_states_71
        layer_output_94 = hidden_states_71 = None
        layer_norm_48 = torch.nn.functional.layer_norm(
            layer_output_95,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_24_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_24_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_24_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_24_modules_norm1_parameters_bias_ = (None)
        linear_144 = torch._C._nn.linear(
            layer_norm_48,
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_73 = linear_144.view(1, -1, 24, 64)
        linear_144 = None
        key_layer_24 = view_73.transpose(1, 2)
        view_73 = None
        linear_145 = torch._C._nn.linear(
            layer_norm_48,
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_74 = linear_145.view(1, -1, 24, 64)
        linear_145 = None
        value_layer_24 = view_74.transpose(1, 2)
        view_74 = None
        linear_146 = torch._C._nn.linear(
            layer_norm_48,
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_48 = l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_75 = linear_146.view(1, -1, 24, 64)
        linear_146 = None
        query_layer_24 = view_75.transpose(1, 2)
        view_75 = None
        query_24 = query_layer_24.contiguous()
        query_layer_24 = None
        key_24 = key_layer_24.contiguous()
        key_layer_24 = None
        value_24 = value_layer_24.contiguous()
        value_layer_24 = None
        attn_output_48 = torch._C._nn.scaled_dot_product_attention(
            query_24,
            key_24,
            value_24,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_24 = key_24 = value_24 = None
        transpose_100 = attn_output_48.transpose(1, 2)
        attn_output_48 = None
        attn_output_49 = transpose_100.contiguous()
        transpose_100 = None
        context_layer_24 = attn_output_49.reshape((1, 257, 1536))
        attn_output_49 = None
        hidden_states_72 = torch._C._nn.linear(
            context_layer_24,
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_24 = l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_73 = torch.nn.functional.dropout(
            hidden_states_72, 0.0, False, False
        )
        hidden_states_72 = None
        attention_output_24 = (
            hidden_states_73
            * l_self_modules_encoder_modules_layer_modules_24_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_73 = l_self_modules_encoder_modules_layer_modules_24_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_74 = attention_output_24 + layer_output_95
        attention_output_24 = layer_output_95 = None
        layer_output_96 = torch.nn.functional.layer_norm(
            hidden_states_74,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_24_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_24_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_24_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_24_modules_norm2_parameters_bias_ = (None)
        hidden_state_24 = torch._C._nn.linear(
            layer_output_96,
            l_self_modules_encoder_modules_layer_modules_24_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_24_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_96 = l_self_modules_encoder_modules_layer_modules_24_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_24_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_24 = hidden_state_24.chunk(2, dim=-1)
        hidden_state_24 = None
        x1_24 = chunk_24[0]
        x2_24 = chunk_24[1]
        chunk_24 = None
        silu_24 = torch.nn.functional.silu(x1_24)
        x1_24 = None
        hidden_24 = silu_24 * x2_24
        silu_24 = x2_24 = None
        layer_output_97 = torch._C._nn.linear(
            hidden_24,
            l_self_modules_encoder_modules_layer_modules_24_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_24_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_24 = l_self_modules_encoder_modules_layer_modules_24_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_24_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_98 = (
            layer_output_97
            * l_self_modules_encoder_modules_layer_modules_24_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_97 = l_self_modules_encoder_modules_layer_modules_24_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_99 = layer_output_98 + hidden_states_74
        layer_output_98 = hidden_states_74 = None
        layer_norm_50 = torch.nn.functional.layer_norm(
            layer_output_99,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_25_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_25_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_25_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_25_modules_norm1_parameters_bias_ = (None)
        linear_150 = torch._C._nn.linear(
            layer_norm_50,
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_76 = linear_150.view(1, -1, 24, 64)
        linear_150 = None
        key_layer_25 = view_76.transpose(1, 2)
        view_76 = None
        linear_151 = torch._C._nn.linear(
            layer_norm_50,
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_77 = linear_151.view(1, -1, 24, 64)
        linear_151 = None
        value_layer_25 = view_77.transpose(1, 2)
        view_77 = None
        linear_152 = torch._C._nn.linear(
            layer_norm_50,
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_50 = l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_78 = linear_152.view(1, -1, 24, 64)
        linear_152 = None
        query_layer_25 = view_78.transpose(1, 2)
        view_78 = None
        query_25 = query_layer_25.contiguous()
        query_layer_25 = None
        key_25 = key_layer_25.contiguous()
        key_layer_25 = None
        value_25 = value_layer_25.contiguous()
        value_layer_25 = None
        attn_output_50 = torch._C._nn.scaled_dot_product_attention(
            query_25,
            key_25,
            value_25,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_25 = key_25 = value_25 = None
        transpose_104 = attn_output_50.transpose(1, 2)
        attn_output_50 = None
        attn_output_51 = transpose_104.contiguous()
        transpose_104 = None
        context_layer_25 = attn_output_51.reshape((1, 257, 1536))
        attn_output_51 = None
        hidden_states_75 = torch._C._nn.linear(
            context_layer_25,
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_25 = l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_76 = torch.nn.functional.dropout(
            hidden_states_75, 0.0, False, False
        )
        hidden_states_75 = None
        attention_output_25 = (
            hidden_states_76
            * l_self_modules_encoder_modules_layer_modules_25_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_76 = l_self_modules_encoder_modules_layer_modules_25_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_77 = attention_output_25 + layer_output_99
        attention_output_25 = layer_output_99 = None
        layer_output_100 = torch.nn.functional.layer_norm(
            hidden_states_77,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_25_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_25_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_25_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_25_modules_norm2_parameters_bias_ = (None)
        hidden_state_25 = torch._C._nn.linear(
            layer_output_100,
            l_self_modules_encoder_modules_layer_modules_25_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_25_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_100 = l_self_modules_encoder_modules_layer_modules_25_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_25_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_25 = hidden_state_25.chunk(2, dim=-1)
        hidden_state_25 = None
        x1_25 = chunk_25[0]
        x2_25 = chunk_25[1]
        chunk_25 = None
        silu_25 = torch.nn.functional.silu(x1_25)
        x1_25 = None
        hidden_25 = silu_25 * x2_25
        silu_25 = x2_25 = None
        layer_output_101 = torch._C._nn.linear(
            hidden_25,
            l_self_modules_encoder_modules_layer_modules_25_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_25_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_25 = l_self_modules_encoder_modules_layer_modules_25_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_25_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_102 = (
            layer_output_101
            * l_self_modules_encoder_modules_layer_modules_25_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_101 = l_self_modules_encoder_modules_layer_modules_25_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_103 = layer_output_102 + hidden_states_77
        layer_output_102 = hidden_states_77 = None
        layer_norm_52 = torch.nn.functional.layer_norm(
            layer_output_103,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_26_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_26_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_26_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_26_modules_norm1_parameters_bias_ = (None)
        linear_156 = torch._C._nn.linear(
            layer_norm_52,
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_79 = linear_156.view(1, -1, 24, 64)
        linear_156 = None
        key_layer_26 = view_79.transpose(1, 2)
        view_79 = None
        linear_157 = torch._C._nn.linear(
            layer_norm_52,
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_80 = linear_157.view(1, -1, 24, 64)
        linear_157 = None
        value_layer_26 = view_80.transpose(1, 2)
        view_80 = None
        linear_158 = torch._C._nn.linear(
            layer_norm_52,
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_52 = l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_81 = linear_158.view(1, -1, 24, 64)
        linear_158 = None
        query_layer_26 = view_81.transpose(1, 2)
        view_81 = None
        query_26 = query_layer_26.contiguous()
        query_layer_26 = None
        key_26 = key_layer_26.contiguous()
        key_layer_26 = None
        value_26 = value_layer_26.contiguous()
        value_layer_26 = None
        attn_output_52 = torch._C._nn.scaled_dot_product_attention(
            query_26,
            key_26,
            value_26,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_26 = key_26 = value_26 = None
        transpose_108 = attn_output_52.transpose(1, 2)
        attn_output_52 = None
        attn_output_53 = transpose_108.contiguous()
        transpose_108 = None
        context_layer_26 = attn_output_53.reshape((1, 257, 1536))
        attn_output_53 = None
        hidden_states_78 = torch._C._nn.linear(
            context_layer_26,
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_26 = l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_79 = torch.nn.functional.dropout(
            hidden_states_78, 0.0, False, False
        )
        hidden_states_78 = None
        attention_output_26 = (
            hidden_states_79
            * l_self_modules_encoder_modules_layer_modules_26_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_79 = l_self_modules_encoder_modules_layer_modules_26_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_80 = attention_output_26 + layer_output_103
        attention_output_26 = layer_output_103 = None
        layer_output_104 = torch.nn.functional.layer_norm(
            hidden_states_80,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_26_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_26_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_26_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_26_modules_norm2_parameters_bias_ = (None)
        hidden_state_26 = torch._C._nn.linear(
            layer_output_104,
            l_self_modules_encoder_modules_layer_modules_26_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_26_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_104 = l_self_modules_encoder_modules_layer_modules_26_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_26_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_26 = hidden_state_26.chunk(2, dim=-1)
        hidden_state_26 = None
        x1_26 = chunk_26[0]
        x2_26 = chunk_26[1]
        chunk_26 = None
        silu_26 = torch.nn.functional.silu(x1_26)
        x1_26 = None
        hidden_26 = silu_26 * x2_26
        silu_26 = x2_26 = None
        layer_output_105 = torch._C._nn.linear(
            hidden_26,
            l_self_modules_encoder_modules_layer_modules_26_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_26_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_26 = l_self_modules_encoder_modules_layer_modules_26_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_26_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_106 = (
            layer_output_105
            * l_self_modules_encoder_modules_layer_modules_26_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_105 = l_self_modules_encoder_modules_layer_modules_26_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_107 = layer_output_106 + hidden_states_80
        layer_output_106 = hidden_states_80 = None
        layer_norm_54 = torch.nn.functional.layer_norm(
            layer_output_107,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_27_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_27_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_27_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_27_modules_norm1_parameters_bias_ = (None)
        linear_162 = torch._C._nn.linear(
            layer_norm_54,
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_82 = linear_162.view(1, -1, 24, 64)
        linear_162 = None
        key_layer_27 = view_82.transpose(1, 2)
        view_82 = None
        linear_163 = torch._C._nn.linear(
            layer_norm_54,
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_83 = linear_163.view(1, -1, 24, 64)
        linear_163 = None
        value_layer_27 = view_83.transpose(1, 2)
        view_83 = None
        linear_164 = torch._C._nn.linear(
            layer_norm_54,
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_54 = l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_84 = linear_164.view(1, -1, 24, 64)
        linear_164 = None
        query_layer_27 = view_84.transpose(1, 2)
        view_84 = None
        query_27 = query_layer_27.contiguous()
        query_layer_27 = None
        key_27 = key_layer_27.contiguous()
        key_layer_27 = None
        value_27 = value_layer_27.contiguous()
        value_layer_27 = None
        attn_output_54 = torch._C._nn.scaled_dot_product_attention(
            query_27,
            key_27,
            value_27,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_27 = key_27 = value_27 = None
        transpose_112 = attn_output_54.transpose(1, 2)
        attn_output_54 = None
        attn_output_55 = transpose_112.contiguous()
        transpose_112 = None
        context_layer_27 = attn_output_55.reshape((1, 257, 1536))
        attn_output_55 = None
        hidden_states_81 = torch._C._nn.linear(
            context_layer_27,
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_27 = l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_82 = torch.nn.functional.dropout(
            hidden_states_81, 0.0, False, False
        )
        hidden_states_81 = None
        attention_output_27 = (
            hidden_states_82
            * l_self_modules_encoder_modules_layer_modules_27_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_82 = l_self_modules_encoder_modules_layer_modules_27_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_83 = attention_output_27 + layer_output_107
        attention_output_27 = layer_output_107 = None
        layer_output_108 = torch.nn.functional.layer_norm(
            hidden_states_83,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_27_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_27_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_27_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_27_modules_norm2_parameters_bias_ = (None)
        hidden_state_27 = torch._C._nn.linear(
            layer_output_108,
            l_self_modules_encoder_modules_layer_modules_27_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_27_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_108 = l_self_modules_encoder_modules_layer_modules_27_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_27_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_27 = hidden_state_27.chunk(2, dim=-1)
        hidden_state_27 = None
        x1_27 = chunk_27[0]
        x2_27 = chunk_27[1]
        chunk_27 = None
        silu_27 = torch.nn.functional.silu(x1_27)
        x1_27 = None
        hidden_27 = silu_27 * x2_27
        silu_27 = x2_27 = None
        layer_output_109 = torch._C._nn.linear(
            hidden_27,
            l_self_modules_encoder_modules_layer_modules_27_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_27_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_27 = l_self_modules_encoder_modules_layer_modules_27_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_27_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_110 = (
            layer_output_109
            * l_self_modules_encoder_modules_layer_modules_27_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_109 = l_self_modules_encoder_modules_layer_modules_27_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_111 = layer_output_110 + hidden_states_83
        layer_output_110 = hidden_states_83 = None
        layer_norm_56 = torch.nn.functional.layer_norm(
            layer_output_111,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_28_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_28_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_28_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_28_modules_norm1_parameters_bias_ = (None)
        linear_168 = torch._C._nn.linear(
            layer_norm_56,
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_85 = linear_168.view(1, -1, 24, 64)
        linear_168 = None
        key_layer_28 = view_85.transpose(1, 2)
        view_85 = None
        linear_169 = torch._C._nn.linear(
            layer_norm_56,
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_86 = linear_169.view(1, -1, 24, 64)
        linear_169 = None
        value_layer_28 = view_86.transpose(1, 2)
        view_86 = None
        linear_170 = torch._C._nn.linear(
            layer_norm_56,
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_56 = l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_87 = linear_170.view(1, -1, 24, 64)
        linear_170 = None
        query_layer_28 = view_87.transpose(1, 2)
        view_87 = None
        query_28 = query_layer_28.contiguous()
        query_layer_28 = None
        key_28 = key_layer_28.contiguous()
        key_layer_28 = None
        value_28 = value_layer_28.contiguous()
        value_layer_28 = None
        attn_output_56 = torch._C._nn.scaled_dot_product_attention(
            query_28,
            key_28,
            value_28,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_28 = key_28 = value_28 = None
        transpose_116 = attn_output_56.transpose(1, 2)
        attn_output_56 = None
        attn_output_57 = transpose_116.contiguous()
        transpose_116 = None
        context_layer_28 = attn_output_57.reshape((1, 257, 1536))
        attn_output_57 = None
        hidden_states_84 = torch._C._nn.linear(
            context_layer_28,
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_28 = l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_85 = torch.nn.functional.dropout(
            hidden_states_84, 0.0, False, False
        )
        hidden_states_84 = None
        attention_output_28 = (
            hidden_states_85
            * l_self_modules_encoder_modules_layer_modules_28_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_85 = l_self_modules_encoder_modules_layer_modules_28_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_86 = attention_output_28 + layer_output_111
        attention_output_28 = layer_output_111 = None
        layer_output_112 = torch.nn.functional.layer_norm(
            hidden_states_86,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_28_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_28_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_28_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_28_modules_norm2_parameters_bias_ = (None)
        hidden_state_28 = torch._C._nn.linear(
            layer_output_112,
            l_self_modules_encoder_modules_layer_modules_28_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_28_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_112 = l_self_modules_encoder_modules_layer_modules_28_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_28_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_28 = hidden_state_28.chunk(2, dim=-1)
        hidden_state_28 = None
        x1_28 = chunk_28[0]
        x2_28 = chunk_28[1]
        chunk_28 = None
        silu_28 = torch.nn.functional.silu(x1_28)
        x1_28 = None
        hidden_28 = silu_28 * x2_28
        silu_28 = x2_28 = None
        layer_output_113 = torch._C._nn.linear(
            hidden_28,
            l_self_modules_encoder_modules_layer_modules_28_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_28_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_28 = l_self_modules_encoder_modules_layer_modules_28_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_28_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_114 = (
            layer_output_113
            * l_self_modules_encoder_modules_layer_modules_28_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_113 = l_self_modules_encoder_modules_layer_modules_28_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_115 = layer_output_114 + hidden_states_86
        layer_output_114 = hidden_states_86 = None
        layer_norm_58 = torch.nn.functional.layer_norm(
            layer_output_115,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_29_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_29_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_29_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_29_modules_norm1_parameters_bias_ = (None)
        linear_174 = torch._C._nn.linear(
            layer_norm_58,
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_88 = linear_174.view(1, -1, 24, 64)
        linear_174 = None
        key_layer_29 = view_88.transpose(1, 2)
        view_88 = None
        linear_175 = torch._C._nn.linear(
            layer_norm_58,
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_89 = linear_175.view(1, -1, 24, 64)
        linear_175 = None
        value_layer_29 = view_89.transpose(1, 2)
        view_89 = None
        linear_176 = torch._C._nn.linear(
            layer_norm_58,
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_58 = l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_90 = linear_176.view(1, -1, 24, 64)
        linear_176 = None
        query_layer_29 = view_90.transpose(1, 2)
        view_90 = None
        query_29 = query_layer_29.contiguous()
        query_layer_29 = None
        key_29 = key_layer_29.contiguous()
        key_layer_29 = None
        value_29 = value_layer_29.contiguous()
        value_layer_29 = None
        attn_output_58 = torch._C._nn.scaled_dot_product_attention(
            query_29,
            key_29,
            value_29,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_29 = key_29 = value_29 = None
        transpose_120 = attn_output_58.transpose(1, 2)
        attn_output_58 = None
        attn_output_59 = transpose_120.contiguous()
        transpose_120 = None
        context_layer_29 = attn_output_59.reshape((1, 257, 1536))
        attn_output_59 = None
        hidden_states_87 = torch._C._nn.linear(
            context_layer_29,
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_29 = l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_88 = torch.nn.functional.dropout(
            hidden_states_87, 0.0, False, False
        )
        hidden_states_87 = None
        attention_output_29 = (
            hidden_states_88
            * l_self_modules_encoder_modules_layer_modules_29_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_88 = l_self_modules_encoder_modules_layer_modules_29_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_89 = attention_output_29 + layer_output_115
        attention_output_29 = layer_output_115 = None
        layer_output_116 = torch.nn.functional.layer_norm(
            hidden_states_89,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_29_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_29_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_29_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_29_modules_norm2_parameters_bias_ = (None)
        hidden_state_29 = torch._C._nn.linear(
            layer_output_116,
            l_self_modules_encoder_modules_layer_modules_29_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_29_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_116 = l_self_modules_encoder_modules_layer_modules_29_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_29_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_29 = hidden_state_29.chunk(2, dim=-1)
        hidden_state_29 = None
        x1_29 = chunk_29[0]
        x2_29 = chunk_29[1]
        chunk_29 = None
        silu_29 = torch.nn.functional.silu(x1_29)
        x1_29 = None
        hidden_29 = silu_29 * x2_29
        silu_29 = x2_29 = None
        layer_output_117 = torch._C._nn.linear(
            hidden_29,
            l_self_modules_encoder_modules_layer_modules_29_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_29_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_29 = l_self_modules_encoder_modules_layer_modules_29_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_29_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_118 = (
            layer_output_117
            * l_self_modules_encoder_modules_layer_modules_29_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_117 = l_self_modules_encoder_modules_layer_modules_29_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_119 = layer_output_118 + hidden_states_89
        layer_output_118 = hidden_states_89 = None
        layer_norm_60 = torch.nn.functional.layer_norm(
            layer_output_119,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_30_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_30_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_30_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_30_modules_norm1_parameters_bias_ = (None)
        linear_180 = torch._C._nn.linear(
            layer_norm_60,
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_91 = linear_180.view(1, -1, 24, 64)
        linear_180 = None
        key_layer_30 = view_91.transpose(1, 2)
        view_91 = None
        linear_181 = torch._C._nn.linear(
            layer_norm_60,
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_92 = linear_181.view(1, -1, 24, 64)
        linear_181 = None
        value_layer_30 = view_92.transpose(1, 2)
        view_92 = None
        linear_182 = torch._C._nn.linear(
            layer_norm_60,
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_60 = l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_93 = linear_182.view(1, -1, 24, 64)
        linear_182 = None
        query_layer_30 = view_93.transpose(1, 2)
        view_93 = None
        query_30 = query_layer_30.contiguous()
        query_layer_30 = None
        key_30 = key_layer_30.contiguous()
        key_layer_30 = None
        value_30 = value_layer_30.contiguous()
        value_layer_30 = None
        attn_output_60 = torch._C._nn.scaled_dot_product_attention(
            query_30,
            key_30,
            value_30,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_30 = key_30 = value_30 = None
        transpose_124 = attn_output_60.transpose(1, 2)
        attn_output_60 = None
        attn_output_61 = transpose_124.contiguous()
        transpose_124 = None
        context_layer_30 = attn_output_61.reshape((1, 257, 1536))
        attn_output_61 = None
        hidden_states_90 = torch._C._nn.linear(
            context_layer_30,
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_30 = l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_91 = torch.nn.functional.dropout(
            hidden_states_90, 0.0, False, False
        )
        hidden_states_90 = None
        attention_output_30 = (
            hidden_states_91
            * l_self_modules_encoder_modules_layer_modules_30_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_91 = l_self_modules_encoder_modules_layer_modules_30_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_92 = attention_output_30 + layer_output_119
        attention_output_30 = layer_output_119 = None
        layer_output_120 = torch.nn.functional.layer_norm(
            hidden_states_92,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_30_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_30_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_30_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_30_modules_norm2_parameters_bias_ = (None)
        hidden_state_30 = torch._C._nn.linear(
            layer_output_120,
            l_self_modules_encoder_modules_layer_modules_30_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_30_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_120 = l_self_modules_encoder_modules_layer_modules_30_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_30_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_30 = hidden_state_30.chunk(2, dim=-1)
        hidden_state_30 = None
        x1_30 = chunk_30[0]
        x2_30 = chunk_30[1]
        chunk_30 = None
        silu_30 = torch.nn.functional.silu(x1_30)
        x1_30 = None
        hidden_30 = silu_30 * x2_30
        silu_30 = x2_30 = None
        layer_output_121 = torch._C._nn.linear(
            hidden_30,
            l_self_modules_encoder_modules_layer_modules_30_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_30_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_30 = l_self_modules_encoder_modules_layer_modules_30_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_30_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_122 = (
            layer_output_121
            * l_self_modules_encoder_modules_layer_modules_30_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_121 = l_self_modules_encoder_modules_layer_modules_30_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_123 = layer_output_122 + hidden_states_92
        layer_output_122 = hidden_states_92 = None
        layer_norm_62 = torch.nn.functional.layer_norm(
            layer_output_123,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_31_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_31_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_31_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_31_modules_norm1_parameters_bias_ = (None)
        linear_186 = torch._C._nn.linear(
            layer_norm_62,
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_94 = linear_186.view(1, -1, 24, 64)
        linear_186 = None
        key_layer_31 = view_94.transpose(1, 2)
        view_94 = None
        linear_187 = torch._C._nn.linear(
            layer_norm_62,
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_95 = linear_187.view(1, -1, 24, 64)
        linear_187 = None
        value_layer_31 = view_95.transpose(1, 2)
        view_95 = None
        linear_188 = torch._C._nn.linear(
            layer_norm_62,
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_62 = l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_96 = linear_188.view(1, -1, 24, 64)
        linear_188 = None
        query_layer_31 = view_96.transpose(1, 2)
        view_96 = None
        query_31 = query_layer_31.contiguous()
        query_layer_31 = None
        key_31 = key_layer_31.contiguous()
        key_layer_31 = None
        value_31 = value_layer_31.contiguous()
        value_layer_31 = None
        attn_output_62 = torch._C._nn.scaled_dot_product_attention(
            query_31,
            key_31,
            value_31,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_31 = key_31 = value_31 = None
        transpose_128 = attn_output_62.transpose(1, 2)
        attn_output_62 = None
        attn_output_63 = transpose_128.contiguous()
        transpose_128 = None
        context_layer_31 = attn_output_63.reshape((1, 257, 1536))
        attn_output_63 = None
        hidden_states_93 = torch._C._nn.linear(
            context_layer_31,
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_31 = l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_94 = torch.nn.functional.dropout(
            hidden_states_93, 0.0, False, False
        )
        hidden_states_93 = None
        attention_output_31 = (
            hidden_states_94
            * l_self_modules_encoder_modules_layer_modules_31_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_94 = l_self_modules_encoder_modules_layer_modules_31_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_95 = attention_output_31 + layer_output_123
        attention_output_31 = layer_output_123 = None
        layer_output_124 = torch.nn.functional.layer_norm(
            hidden_states_95,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_31_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_31_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_31_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_31_modules_norm2_parameters_bias_ = (None)
        hidden_state_31 = torch._C._nn.linear(
            layer_output_124,
            l_self_modules_encoder_modules_layer_modules_31_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_31_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_124 = l_self_modules_encoder_modules_layer_modules_31_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_31_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_31 = hidden_state_31.chunk(2, dim=-1)
        hidden_state_31 = None
        x1_31 = chunk_31[0]
        x2_31 = chunk_31[1]
        chunk_31 = None
        silu_31 = torch.nn.functional.silu(x1_31)
        x1_31 = None
        hidden_31 = silu_31 * x2_31
        silu_31 = x2_31 = None
        layer_output_125 = torch._C._nn.linear(
            hidden_31,
            l_self_modules_encoder_modules_layer_modules_31_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_31_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_31 = l_self_modules_encoder_modules_layer_modules_31_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_31_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_126 = (
            layer_output_125
            * l_self_modules_encoder_modules_layer_modules_31_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_125 = l_self_modules_encoder_modules_layer_modules_31_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_127 = layer_output_126 + hidden_states_95
        layer_output_126 = hidden_states_95 = None
        layer_norm_64 = torch.nn.functional.layer_norm(
            layer_output_127,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_32_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_32_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_32_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_32_modules_norm1_parameters_bias_ = (None)
        linear_192 = torch._C._nn.linear(
            layer_norm_64,
            l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_97 = linear_192.view(1, -1, 24, 64)
        linear_192 = None
        key_layer_32 = view_97.transpose(1, 2)
        view_97 = None
        linear_193 = torch._C._nn.linear(
            layer_norm_64,
            l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_98 = linear_193.view(1, -1, 24, 64)
        linear_193 = None
        value_layer_32 = view_98.transpose(1, 2)
        view_98 = None
        linear_194 = torch._C._nn.linear(
            layer_norm_64,
            l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_64 = l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_99 = linear_194.view(1, -1, 24, 64)
        linear_194 = None
        query_layer_32 = view_99.transpose(1, 2)
        view_99 = None
        query_32 = query_layer_32.contiguous()
        query_layer_32 = None
        key_32 = key_layer_32.contiguous()
        key_layer_32 = None
        value_32 = value_layer_32.contiguous()
        value_layer_32 = None
        attn_output_64 = torch._C._nn.scaled_dot_product_attention(
            query_32,
            key_32,
            value_32,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_32 = key_32 = value_32 = None
        transpose_132 = attn_output_64.transpose(1, 2)
        attn_output_64 = None
        attn_output_65 = transpose_132.contiguous()
        transpose_132 = None
        context_layer_32 = attn_output_65.reshape((1, 257, 1536))
        attn_output_65 = None
        hidden_states_96 = torch._C._nn.linear(
            context_layer_32,
            l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_32 = l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_97 = torch.nn.functional.dropout(
            hidden_states_96, 0.0, False, False
        )
        hidden_states_96 = None
        attention_output_32 = (
            hidden_states_97
            * l_self_modules_encoder_modules_layer_modules_32_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_97 = l_self_modules_encoder_modules_layer_modules_32_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_98 = attention_output_32 + layer_output_127
        attention_output_32 = layer_output_127 = None
        layer_output_128 = torch.nn.functional.layer_norm(
            hidden_states_98,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_32_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_32_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_32_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_32_modules_norm2_parameters_bias_ = (None)
        hidden_state_32 = torch._C._nn.linear(
            layer_output_128,
            l_self_modules_encoder_modules_layer_modules_32_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_32_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_128 = l_self_modules_encoder_modules_layer_modules_32_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_32_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_32 = hidden_state_32.chunk(2, dim=-1)
        hidden_state_32 = None
        x1_32 = chunk_32[0]
        x2_32 = chunk_32[1]
        chunk_32 = None
        silu_32 = torch.nn.functional.silu(x1_32)
        x1_32 = None
        hidden_32 = silu_32 * x2_32
        silu_32 = x2_32 = None
        layer_output_129 = torch._C._nn.linear(
            hidden_32,
            l_self_modules_encoder_modules_layer_modules_32_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_32_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_32 = l_self_modules_encoder_modules_layer_modules_32_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_32_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_130 = (
            layer_output_129
            * l_self_modules_encoder_modules_layer_modules_32_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_129 = l_self_modules_encoder_modules_layer_modules_32_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_131 = layer_output_130 + hidden_states_98
        layer_output_130 = hidden_states_98 = None
        layer_norm_66 = torch.nn.functional.layer_norm(
            layer_output_131,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_33_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_33_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_33_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_33_modules_norm1_parameters_bias_ = (None)
        linear_198 = torch._C._nn.linear(
            layer_norm_66,
            l_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_100 = linear_198.view(1, -1, 24, 64)
        linear_198 = None
        key_layer_33 = view_100.transpose(1, 2)
        view_100 = None
        linear_199 = torch._C._nn.linear(
            layer_norm_66,
            l_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_101 = linear_199.view(1, -1, 24, 64)
        linear_199 = None
        value_layer_33 = view_101.transpose(1, 2)
        view_101 = None
        linear_200 = torch._C._nn.linear(
            layer_norm_66,
            l_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_66 = l_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_102 = linear_200.view(1, -1, 24, 64)
        linear_200 = None
        query_layer_33 = view_102.transpose(1, 2)
        view_102 = None
        query_33 = query_layer_33.contiguous()
        query_layer_33 = None
        key_33 = key_layer_33.contiguous()
        key_layer_33 = None
        value_33 = value_layer_33.contiguous()
        value_layer_33 = None
        attn_output_66 = torch._C._nn.scaled_dot_product_attention(
            query_33,
            key_33,
            value_33,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_33 = key_33 = value_33 = None
        transpose_136 = attn_output_66.transpose(1, 2)
        attn_output_66 = None
        attn_output_67 = transpose_136.contiguous()
        transpose_136 = None
        context_layer_33 = attn_output_67.reshape((1, 257, 1536))
        attn_output_67 = None
        hidden_states_99 = torch._C._nn.linear(
            context_layer_33,
            l_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_33 = l_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_33_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_100 = torch.nn.functional.dropout(
            hidden_states_99, 0.0, False, False
        )
        hidden_states_99 = None
        attention_output_33 = (
            hidden_states_100
            * l_self_modules_encoder_modules_layer_modules_33_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_100 = l_self_modules_encoder_modules_layer_modules_33_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_101 = attention_output_33 + layer_output_131
        attention_output_33 = layer_output_131 = None
        layer_output_132 = torch.nn.functional.layer_norm(
            hidden_states_101,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_33_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_33_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_33_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_33_modules_norm2_parameters_bias_ = (None)
        hidden_state_33 = torch._C._nn.linear(
            layer_output_132,
            l_self_modules_encoder_modules_layer_modules_33_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_33_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_132 = l_self_modules_encoder_modules_layer_modules_33_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_33_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_33 = hidden_state_33.chunk(2, dim=-1)
        hidden_state_33 = None
        x1_33 = chunk_33[0]
        x2_33 = chunk_33[1]
        chunk_33 = None
        silu_33 = torch.nn.functional.silu(x1_33)
        x1_33 = None
        hidden_33 = silu_33 * x2_33
        silu_33 = x2_33 = None
        layer_output_133 = torch._C._nn.linear(
            hidden_33,
            l_self_modules_encoder_modules_layer_modules_33_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_33_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_33 = l_self_modules_encoder_modules_layer_modules_33_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_33_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_134 = (
            layer_output_133
            * l_self_modules_encoder_modules_layer_modules_33_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_133 = l_self_modules_encoder_modules_layer_modules_33_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_135 = layer_output_134 + hidden_states_101
        layer_output_134 = hidden_states_101 = None
        layer_norm_68 = torch.nn.functional.layer_norm(
            layer_output_135,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_34_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_34_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_34_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_34_modules_norm1_parameters_bias_ = (None)
        linear_204 = torch._C._nn.linear(
            layer_norm_68,
            l_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_103 = linear_204.view(1, -1, 24, 64)
        linear_204 = None
        key_layer_34 = view_103.transpose(1, 2)
        view_103 = None
        linear_205 = torch._C._nn.linear(
            layer_norm_68,
            l_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_104 = linear_205.view(1, -1, 24, 64)
        linear_205 = None
        value_layer_34 = view_104.transpose(1, 2)
        view_104 = None
        linear_206 = torch._C._nn.linear(
            layer_norm_68,
            l_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_68 = l_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_105 = linear_206.view(1, -1, 24, 64)
        linear_206 = None
        query_layer_34 = view_105.transpose(1, 2)
        view_105 = None
        query_34 = query_layer_34.contiguous()
        query_layer_34 = None
        key_34 = key_layer_34.contiguous()
        key_layer_34 = None
        value_34 = value_layer_34.contiguous()
        value_layer_34 = None
        attn_output_68 = torch._C._nn.scaled_dot_product_attention(
            query_34,
            key_34,
            value_34,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_34 = key_34 = value_34 = None
        transpose_140 = attn_output_68.transpose(1, 2)
        attn_output_68 = None
        attn_output_69 = transpose_140.contiguous()
        transpose_140 = None
        context_layer_34 = attn_output_69.reshape((1, 257, 1536))
        attn_output_69 = None
        hidden_states_102 = torch._C._nn.linear(
            context_layer_34,
            l_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_34 = l_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_34_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_103 = torch.nn.functional.dropout(
            hidden_states_102, 0.0, False, False
        )
        hidden_states_102 = None
        attention_output_34 = (
            hidden_states_103
            * l_self_modules_encoder_modules_layer_modules_34_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_103 = l_self_modules_encoder_modules_layer_modules_34_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_104 = attention_output_34 + layer_output_135
        attention_output_34 = layer_output_135 = None
        layer_output_136 = torch.nn.functional.layer_norm(
            hidden_states_104,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_34_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_34_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_34_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_34_modules_norm2_parameters_bias_ = (None)
        hidden_state_34 = torch._C._nn.linear(
            layer_output_136,
            l_self_modules_encoder_modules_layer_modules_34_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_34_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_136 = l_self_modules_encoder_modules_layer_modules_34_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_34_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_34 = hidden_state_34.chunk(2, dim=-1)
        hidden_state_34 = None
        x1_34 = chunk_34[0]
        x2_34 = chunk_34[1]
        chunk_34 = None
        silu_34 = torch.nn.functional.silu(x1_34)
        x1_34 = None
        hidden_34 = silu_34 * x2_34
        silu_34 = x2_34 = None
        layer_output_137 = torch._C._nn.linear(
            hidden_34,
            l_self_modules_encoder_modules_layer_modules_34_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_34_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_34 = l_self_modules_encoder_modules_layer_modules_34_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_34_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_138 = (
            layer_output_137
            * l_self_modules_encoder_modules_layer_modules_34_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_137 = l_self_modules_encoder_modules_layer_modules_34_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_139 = layer_output_138 + hidden_states_104
        layer_output_138 = hidden_states_104 = None
        layer_norm_70 = torch.nn.functional.layer_norm(
            layer_output_139,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_35_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_35_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_35_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_35_modules_norm1_parameters_bias_ = (None)
        linear_210 = torch._C._nn.linear(
            layer_norm_70,
            l_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_106 = linear_210.view(1, -1, 24, 64)
        linear_210 = None
        key_layer_35 = view_106.transpose(1, 2)
        view_106 = None
        linear_211 = torch._C._nn.linear(
            layer_norm_70,
            l_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_107 = linear_211.view(1, -1, 24, 64)
        linear_211 = None
        value_layer_35 = view_107.transpose(1, 2)
        view_107 = None
        linear_212 = torch._C._nn.linear(
            layer_norm_70,
            l_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_70 = l_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_108 = linear_212.view(1, -1, 24, 64)
        linear_212 = None
        query_layer_35 = view_108.transpose(1, 2)
        view_108 = None
        query_35 = query_layer_35.contiguous()
        query_layer_35 = None
        key_35 = key_layer_35.contiguous()
        key_layer_35 = None
        value_35 = value_layer_35.contiguous()
        value_layer_35 = None
        attn_output_70 = torch._C._nn.scaled_dot_product_attention(
            query_35,
            key_35,
            value_35,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_35 = key_35 = value_35 = None
        transpose_144 = attn_output_70.transpose(1, 2)
        attn_output_70 = None
        attn_output_71 = transpose_144.contiguous()
        transpose_144 = None
        context_layer_35 = attn_output_71.reshape((1, 257, 1536))
        attn_output_71 = None
        hidden_states_105 = torch._C._nn.linear(
            context_layer_35,
            l_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_35 = l_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_35_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_106 = torch.nn.functional.dropout(
            hidden_states_105, 0.0, False, False
        )
        hidden_states_105 = None
        attention_output_35 = (
            hidden_states_106
            * l_self_modules_encoder_modules_layer_modules_35_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_106 = l_self_modules_encoder_modules_layer_modules_35_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_107 = attention_output_35 + layer_output_139
        attention_output_35 = layer_output_139 = None
        layer_output_140 = torch.nn.functional.layer_norm(
            hidden_states_107,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_35_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_35_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_35_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_35_modules_norm2_parameters_bias_ = (None)
        hidden_state_35 = torch._C._nn.linear(
            layer_output_140,
            l_self_modules_encoder_modules_layer_modules_35_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_35_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_140 = l_self_modules_encoder_modules_layer_modules_35_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_35_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_35 = hidden_state_35.chunk(2, dim=-1)
        hidden_state_35 = None
        x1_35 = chunk_35[0]
        x2_35 = chunk_35[1]
        chunk_35 = None
        silu_35 = torch.nn.functional.silu(x1_35)
        x1_35 = None
        hidden_35 = silu_35 * x2_35
        silu_35 = x2_35 = None
        layer_output_141 = torch._C._nn.linear(
            hidden_35,
            l_self_modules_encoder_modules_layer_modules_35_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_35_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_35 = l_self_modules_encoder_modules_layer_modules_35_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_35_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_142 = (
            layer_output_141
            * l_self_modules_encoder_modules_layer_modules_35_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_141 = l_self_modules_encoder_modules_layer_modules_35_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_143 = layer_output_142 + hidden_states_107
        layer_output_142 = hidden_states_107 = None
        layer_norm_72 = torch.nn.functional.layer_norm(
            layer_output_143,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_36_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_36_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_36_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_36_modules_norm1_parameters_bias_ = (None)
        linear_216 = torch._C._nn.linear(
            layer_norm_72,
            l_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_109 = linear_216.view(1, -1, 24, 64)
        linear_216 = None
        key_layer_36 = view_109.transpose(1, 2)
        view_109 = None
        linear_217 = torch._C._nn.linear(
            layer_norm_72,
            l_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_110 = linear_217.view(1, -1, 24, 64)
        linear_217 = None
        value_layer_36 = view_110.transpose(1, 2)
        view_110 = None
        linear_218 = torch._C._nn.linear(
            layer_norm_72,
            l_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_72 = l_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_111 = linear_218.view(1, -1, 24, 64)
        linear_218 = None
        query_layer_36 = view_111.transpose(1, 2)
        view_111 = None
        query_36 = query_layer_36.contiguous()
        query_layer_36 = None
        key_36 = key_layer_36.contiguous()
        key_layer_36 = None
        value_36 = value_layer_36.contiguous()
        value_layer_36 = None
        attn_output_72 = torch._C._nn.scaled_dot_product_attention(
            query_36,
            key_36,
            value_36,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_36 = key_36 = value_36 = None
        transpose_148 = attn_output_72.transpose(1, 2)
        attn_output_72 = None
        attn_output_73 = transpose_148.contiguous()
        transpose_148 = None
        context_layer_36 = attn_output_73.reshape((1, 257, 1536))
        attn_output_73 = None
        hidden_states_108 = torch._C._nn.linear(
            context_layer_36,
            l_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_36 = l_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_36_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_109 = torch.nn.functional.dropout(
            hidden_states_108, 0.0, False, False
        )
        hidden_states_108 = None
        attention_output_36 = (
            hidden_states_109
            * l_self_modules_encoder_modules_layer_modules_36_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_109 = l_self_modules_encoder_modules_layer_modules_36_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_110 = attention_output_36 + layer_output_143
        attention_output_36 = layer_output_143 = None
        layer_output_144 = torch.nn.functional.layer_norm(
            hidden_states_110,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_36_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_36_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_36_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_36_modules_norm2_parameters_bias_ = (None)
        hidden_state_36 = torch._C._nn.linear(
            layer_output_144,
            l_self_modules_encoder_modules_layer_modules_36_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_36_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_144 = l_self_modules_encoder_modules_layer_modules_36_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_36_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_36 = hidden_state_36.chunk(2, dim=-1)
        hidden_state_36 = None
        x1_36 = chunk_36[0]
        x2_36 = chunk_36[1]
        chunk_36 = None
        silu_36 = torch.nn.functional.silu(x1_36)
        x1_36 = None
        hidden_36 = silu_36 * x2_36
        silu_36 = x2_36 = None
        layer_output_145 = torch._C._nn.linear(
            hidden_36,
            l_self_modules_encoder_modules_layer_modules_36_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_36_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_36 = l_self_modules_encoder_modules_layer_modules_36_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_36_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_146 = (
            layer_output_145
            * l_self_modules_encoder_modules_layer_modules_36_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_145 = l_self_modules_encoder_modules_layer_modules_36_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_147 = layer_output_146 + hidden_states_110
        layer_output_146 = hidden_states_110 = None
        layer_norm_74 = torch.nn.functional.layer_norm(
            layer_output_147,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_37_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_37_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_37_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_37_modules_norm1_parameters_bias_ = (None)
        linear_222 = torch._C._nn.linear(
            layer_norm_74,
            l_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_112 = linear_222.view(1, -1, 24, 64)
        linear_222 = None
        key_layer_37 = view_112.transpose(1, 2)
        view_112 = None
        linear_223 = torch._C._nn.linear(
            layer_norm_74,
            l_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_113 = linear_223.view(1, -1, 24, 64)
        linear_223 = None
        value_layer_37 = view_113.transpose(1, 2)
        view_113 = None
        linear_224 = torch._C._nn.linear(
            layer_norm_74,
            l_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_74 = l_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_114 = linear_224.view(1, -1, 24, 64)
        linear_224 = None
        query_layer_37 = view_114.transpose(1, 2)
        view_114 = None
        query_37 = query_layer_37.contiguous()
        query_layer_37 = None
        key_37 = key_layer_37.contiguous()
        key_layer_37 = None
        value_37 = value_layer_37.contiguous()
        value_layer_37 = None
        attn_output_74 = torch._C._nn.scaled_dot_product_attention(
            query_37,
            key_37,
            value_37,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_37 = key_37 = value_37 = None
        transpose_152 = attn_output_74.transpose(1, 2)
        attn_output_74 = None
        attn_output_75 = transpose_152.contiguous()
        transpose_152 = None
        context_layer_37 = attn_output_75.reshape((1, 257, 1536))
        attn_output_75 = None
        hidden_states_111 = torch._C._nn.linear(
            context_layer_37,
            l_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_37 = l_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_37_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_112 = torch.nn.functional.dropout(
            hidden_states_111, 0.0, False, False
        )
        hidden_states_111 = None
        attention_output_37 = (
            hidden_states_112
            * l_self_modules_encoder_modules_layer_modules_37_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_112 = l_self_modules_encoder_modules_layer_modules_37_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_113 = attention_output_37 + layer_output_147
        attention_output_37 = layer_output_147 = None
        layer_output_148 = torch.nn.functional.layer_norm(
            hidden_states_113,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_37_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_37_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_37_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_37_modules_norm2_parameters_bias_ = (None)
        hidden_state_37 = torch._C._nn.linear(
            layer_output_148,
            l_self_modules_encoder_modules_layer_modules_37_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_37_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_148 = l_self_modules_encoder_modules_layer_modules_37_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_37_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_37 = hidden_state_37.chunk(2, dim=-1)
        hidden_state_37 = None
        x1_37 = chunk_37[0]
        x2_37 = chunk_37[1]
        chunk_37 = None
        silu_37 = torch.nn.functional.silu(x1_37)
        x1_37 = None
        hidden_37 = silu_37 * x2_37
        silu_37 = x2_37 = None
        layer_output_149 = torch._C._nn.linear(
            hidden_37,
            l_self_modules_encoder_modules_layer_modules_37_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_37_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_37 = l_self_modules_encoder_modules_layer_modules_37_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_37_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_150 = (
            layer_output_149
            * l_self_modules_encoder_modules_layer_modules_37_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_149 = l_self_modules_encoder_modules_layer_modules_37_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_151 = layer_output_150 + hidden_states_113
        layer_output_150 = hidden_states_113 = None
        layer_norm_76 = torch.nn.functional.layer_norm(
            layer_output_151,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_38_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_38_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_38_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_38_modules_norm1_parameters_bias_ = (None)
        linear_228 = torch._C._nn.linear(
            layer_norm_76,
            l_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_115 = linear_228.view(1, -1, 24, 64)
        linear_228 = None
        key_layer_38 = view_115.transpose(1, 2)
        view_115 = None
        linear_229 = torch._C._nn.linear(
            layer_norm_76,
            l_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_116 = linear_229.view(1, -1, 24, 64)
        linear_229 = None
        value_layer_38 = view_116.transpose(1, 2)
        view_116 = None
        linear_230 = torch._C._nn.linear(
            layer_norm_76,
            l_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_76 = l_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_117 = linear_230.view(1, -1, 24, 64)
        linear_230 = None
        query_layer_38 = view_117.transpose(1, 2)
        view_117 = None
        query_38 = query_layer_38.contiguous()
        query_layer_38 = None
        key_38 = key_layer_38.contiguous()
        key_layer_38 = None
        value_38 = value_layer_38.contiguous()
        value_layer_38 = None
        attn_output_76 = torch._C._nn.scaled_dot_product_attention(
            query_38,
            key_38,
            value_38,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_38 = key_38 = value_38 = None
        transpose_156 = attn_output_76.transpose(1, 2)
        attn_output_76 = None
        attn_output_77 = transpose_156.contiguous()
        transpose_156 = None
        context_layer_38 = attn_output_77.reshape((1, 257, 1536))
        attn_output_77 = None
        hidden_states_114 = torch._C._nn.linear(
            context_layer_38,
            l_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_38 = l_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_38_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_115 = torch.nn.functional.dropout(
            hidden_states_114, 0.0, False, False
        )
        hidden_states_114 = None
        attention_output_38 = (
            hidden_states_115
            * l_self_modules_encoder_modules_layer_modules_38_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_115 = l_self_modules_encoder_modules_layer_modules_38_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_116 = attention_output_38 + layer_output_151
        attention_output_38 = layer_output_151 = None
        layer_output_152 = torch.nn.functional.layer_norm(
            hidden_states_116,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_38_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_38_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_38_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_38_modules_norm2_parameters_bias_ = (None)
        hidden_state_38 = torch._C._nn.linear(
            layer_output_152,
            l_self_modules_encoder_modules_layer_modules_38_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_38_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_152 = l_self_modules_encoder_modules_layer_modules_38_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_38_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_38 = hidden_state_38.chunk(2, dim=-1)
        hidden_state_38 = None
        x1_38 = chunk_38[0]
        x2_38 = chunk_38[1]
        chunk_38 = None
        silu_38 = torch.nn.functional.silu(x1_38)
        x1_38 = None
        hidden_38 = silu_38 * x2_38
        silu_38 = x2_38 = None
        layer_output_153 = torch._C._nn.linear(
            hidden_38,
            l_self_modules_encoder_modules_layer_modules_38_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_38_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_38 = l_self_modules_encoder_modules_layer_modules_38_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_38_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_154 = (
            layer_output_153
            * l_self_modules_encoder_modules_layer_modules_38_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_153 = l_self_modules_encoder_modules_layer_modules_38_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_155 = layer_output_154 + hidden_states_116
        layer_output_154 = hidden_states_116 = None
        layer_norm_78 = torch.nn.functional.layer_norm(
            layer_output_155,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_39_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_39_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_39_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_39_modules_norm1_parameters_bias_ = (None)
        linear_234 = torch._C._nn.linear(
            layer_norm_78,
            l_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_118 = linear_234.view(1, -1, 24, 64)
        linear_234 = None
        key_layer_39 = view_118.transpose(1, 2)
        view_118 = None
        linear_235 = torch._C._nn.linear(
            layer_norm_78,
            l_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_119 = linear_235.view(1, -1, 24, 64)
        linear_235 = None
        value_layer_39 = view_119.transpose(1, 2)
        view_119 = None
        linear_236 = torch._C._nn.linear(
            layer_norm_78,
            l_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_78 = l_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_120 = linear_236.view(1, -1, 24, 64)
        linear_236 = None
        query_layer_39 = view_120.transpose(1, 2)
        view_120 = None
        query_39 = query_layer_39.contiguous()
        query_layer_39 = None
        key_39 = key_layer_39.contiguous()
        key_layer_39 = None
        value_39 = value_layer_39.contiguous()
        value_layer_39 = None
        attn_output_78 = torch._C._nn.scaled_dot_product_attention(
            query_39,
            key_39,
            value_39,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_39 = key_39 = value_39 = None
        transpose_160 = attn_output_78.transpose(1, 2)
        attn_output_78 = None
        attn_output_79 = transpose_160.contiguous()
        transpose_160 = None
        context_layer_39 = attn_output_79.reshape((1, 257, 1536))
        attn_output_79 = None
        hidden_states_117 = torch._C._nn.linear(
            context_layer_39,
            l_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_39 = l_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_39_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_118 = torch.nn.functional.dropout(
            hidden_states_117, 0.0, False, False
        )
        hidden_states_117 = None
        attention_output_39 = (
            hidden_states_118
            * l_self_modules_encoder_modules_layer_modules_39_modules_layer_scale1_parameters_lambda1_
        )
        hidden_states_118 = l_self_modules_encoder_modules_layer_modules_39_modules_layer_scale1_parameters_lambda1_ = (None)
        hidden_states_119 = attention_output_39 + layer_output_155
        attention_output_39 = layer_output_155 = None
        layer_output_156 = torch.nn.functional.layer_norm(
            hidden_states_119,
            (1536,),
            l_self_modules_encoder_modules_layer_modules_39_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_39_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_39_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_39_modules_norm2_parameters_bias_ = (None)
        hidden_state_39 = torch._C._nn.linear(
            layer_output_156,
            l_self_modules_encoder_modules_layer_modules_39_modules_mlp_modules_weights_in_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_39_modules_mlp_modules_weights_in_parameters_bias_,
        )
        layer_output_156 = l_self_modules_encoder_modules_layer_modules_39_modules_mlp_modules_weights_in_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_39_modules_mlp_modules_weights_in_parameters_bias_ = (None)
        chunk_39 = hidden_state_39.chunk(2, dim=-1)
        hidden_state_39 = None
        x1_39 = chunk_39[0]
        x2_39 = chunk_39[1]
        chunk_39 = None
        silu_39 = torch.nn.functional.silu(x1_39)
        x1_39 = None
        hidden_39 = silu_39 * x2_39
        silu_39 = x2_39 = None
        layer_output_157 = torch._C._nn.linear(
            hidden_39,
            l_self_modules_encoder_modules_layer_modules_39_modules_mlp_modules_weights_out_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_39_modules_mlp_modules_weights_out_parameters_bias_,
        )
        hidden_39 = l_self_modules_encoder_modules_layer_modules_39_modules_mlp_modules_weights_out_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_39_modules_mlp_modules_weights_out_parameters_bias_ = (None)
        layer_output_158 = (
            layer_output_157
            * l_self_modules_encoder_modules_layer_modules_39_modules_layer_scale2_parameters_lambda1_
        )
        layer_output_157 = l_self_modules_encoder_modules_layer_modules_39_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_159 = layer_output_158 + hidden_states_119
        layer_output_158 = hidden_states_119 = None
        sequence_output = torch.nn.functional.layer_norm(
            layer_output_159,
            (1536,),
            l_self_modules_layernorm_parameters_weight_,
            l_self_modules_layernorm_parameters_bias_,
            1e-06,
        )
        layer_output_159 = (
            l_self_modules_layernorm_parameters_weight_
        ) = l_self_modules_layernorm_parameters_bias_ = None
        pooled_output = sequence_output[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        return (sequence_output, pooled_output)
