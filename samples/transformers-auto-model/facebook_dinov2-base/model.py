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
        L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_layer_scale2_parameters_lambda1_: torch.nn.parameter.Parameter,
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
        l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_fc2_parameters_bias_
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
        l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_fc2_parameters_bias_
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
        l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_fc2_parameters_bias_
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
        l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_fc2_parameters_bias_
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
        l_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_fc2_parameters_bias_
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
        l_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_fc2_parameters_bias_
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
        l_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_fc2_parameters_bias_
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
        l_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_fc2_parameters_bias_
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
        l_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_fc2_parameters_bias_
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
        l_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_fc2_parameters_bias_
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
        l_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_fc2_parameters_bias_
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
        l_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_layer_scale2_parameters_lambda1_ = L_self_modules_encoder_modules_layer_modules_11_modules_layer_scale2_parameters_lambda1_
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
        patch_pos_embed_1 = patch_pos_embed.reshape(1, 37, 37, 768)
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
        patch_pos_embed_4 = permute_1.view(1, -1, 768)
        permute_1 = None
        cat_1 = torch.cat((class_pos_embed, patch_pos_embed_4), dim=1)
        class_pos_embed = patch_pos_embed_4 = None
        embeddings_2 = embeddings_1 + cat_1
        embeddings_1 = cat_1 = None
        embeddings_3 = torch.nn.functional.dropout(embeddings_2, 0.0, False, False)
        embeddings_2 = None
        layer_norm = torch.nn.functional.layer_norm(
            embeddings_3,
            (768,),
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
        view_1 = linear.view(1, -1, 12, 64)
        linear = None
        key_layer = view_1.transpose(1, 2)
        view_1 = None
        linear_1 = torch._C._nn.linear(
            layer_norm,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_2 = linear_1.view(1, -1, 12, 64)
        linear_1 = None
        value_layer = view_2.transpose(1, 2)
        view_2 = None
        linear_2 = torch._C._nn.linear(
            layer_norm,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_3 = linear_2.view(1, -1, 12, 64)
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
        context_layer = attn_output_1.reshape((1, 257, 768))
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
            (768,),
            l_self_modules_encoder_modules_layer_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_norm2_parameters_bias_ = (None)
        hidden_state = torch._C._nn.linear(
            layer_output,
            l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_output = l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        hidden_state_1 = torch._C._nn.gelu(hidden_state)
        hidden_state = None
        hidden_state_2 = torch._C._nn.linear(
            hidden_state_1,
            l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_state_1 = l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        layer_output_1 = (
            hidden_state_2
            * l_self_modules_encoder_modules_layer_modules_0_modules_layer_scale2_parameters_lambda1_
        )
        hidden_state_2 = l_self_modules_encoder_modules_layer_modules_0_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_2 = layer_output_1 + hidden_states_2
        layer_output_1 = hidden_states_2 = None
        layer_norm_2 = torch.nn.functional.layer_norm(
            layer_output_2,
            (768,),
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
        view_4 = linear_6.view(1, -1, 12, 64)
        linear_6 = None
        key_layer_1 = view_4.transpose(1, 2)
        view_4 = None
        linear_7 = torch._C._nn.linear(
            layer_norm_2,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_5 = linear_7.view(1, -1, 12, 64)
        linear_7 = None
        value_layer_1 = view_5.transpose(1, 2)
        view_5 = None
        linear_8 = torch._C._nn.linear(
            layer_norm_2,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_2 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_6 = linear_8.view(1, -1, 12, 64)
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
        context_layer_1 = attn_output_3.reshape((1, 257, 768))
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
        hidden_states_5 = attention_output_1 + layer_output_2
        attention_output_1 = layer_output_2 = None
        layer_output_3 = torch.nn.functional.layer_norm(
            hidden_states_5,
            (768,),
            l_self_modules_encoder_modules_layer_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_norm2_parameters_bias_ = (None)
        hidden_state_3 = torch._C._nn.linear(
            layer_output_3,
            l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_output_3 = l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        hidden_state_4 = torch._C._nn.gelu(hidden_state_3)
        hidden_state_3 = None
        hidden_state_5 = torch._C._nn.linear(
            hidden_state_4,
            l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_state_4 = l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        layer_output_4 = (
            hidden_state_5
            * l_self_modules_encoder_modules_layer_modules_1_modules_layer_scale2_parameters_lambda1_
        )
        hidden_state_5 = l_self_modules_encoder_modules_layer_modules_1_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_5 = layer_output_4 + hidden_states_5
        layer_output_4 = hidden_states_5 = None
        layer_norm_4 = torch.nn.functional.layer_norm(
            layer_output_5,
            (768,),
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
        view_7 = linear_12.view(1, -1, 12, 64)
        linear_12 = None
        key_layer_2 = view_7.transpose(1, 2)
        view_7 = None
        linear_13 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_8 = linear_13.view(1, -1, 12, 64)
        linear_13 = None
        value_layer_2 = view_8.transpose(1, 2)
        view_8 = None
        linear_14 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_4 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_9 = linear_14.view(1, -1, 12, 64)
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
        context_layer_2 = attn_output_5.reshape((1, 257, 768))
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
        hidden_states_8 = attention_output_2 + layer_output_5
        attention_output_2 = layer_output_5 = None
        layer_output_6 = torch.nn.functional.layer_norm(
            hidden_states_8,
            (768,),
            l_self_modules_encoder_modules_layer_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_norm2_parameters_bias_ = (None)
        hidden_state_6 = torch._C._nn.linear(
            layer_output_6,
            l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_output_6 = l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        hidden_state_7 = torch._C._nn.gelu(hidden_state_6)
        hidden_state_6 = None
        hidden_state_8 = torch._C._nn.linear(
            hidden_state_7,
            l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_state_7 = l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        layer_output_7 = (
            hidden_state_8
            * l_self_modules_encoder_modules_layer_modules_2_modules_layer_scale2_parameters_lambda1_
        )
        hidden_state_8 = l_self_modules_encoder_modules_layer_modules_2_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_8 = layer_output_7 + hidden_states_8
        layer_output_7 = hidden_states_8 = None
        layer_norm_6 = torch.nn.functional.layer_norm(
            layer_output_8,
            (768,),
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
        view_10 = linear_18.view(1, -1, 12, 64)
        linear_18 = None
        key_layer_3 = view_10.transpose(1, 2)
        view_10 = None
        linear_19 = torch._C._nn.linear(
            layer_norm_6,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_11 = linear_19.view(1, -1, 12, 64)
        linear_19 = None
        value_layer_3 = view_11.transpose(1, 2)
        view_11 = None
        linear_20 = torch._C._nn.linear(
            layer_norm_6,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_6 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_12 = linear_20.view(1, -1, 12, 64)
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
        context_layer_3 = attn_output_7.reshape((1, 257, 768))
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
        hidden_states_11 = attention_output_3 + layer_output_8
        attention_output_3 = layer_output_8 = None
        layer_output_9 = torch.nn.functional.layer_norm(
            hidden_states_11,
            (768,),
            l_self_modules_encoder_modules_layer_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_norm2_parameters_bias_ = (None)
        hidden_state_9 = torch._C._nn.linear(
            layer_output_9,
            l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_output_9 = l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        hidden_state_10 = torch._C._nn.gelu(hidden_state_9)
        hidden_state_9 = None
        hidden_state_11 = torch._C._nn.linear(
            hidden_state_10,
            l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_state_10 = l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        layer_output_10 = (
            hidden_state_11
            * l_self_modules_encoder_modules_layer_modules_3_modules_layer_scale2_parameters_lambda1_
        )
        hidden_state_11 = l_self_modules_encoder_modules_layer_modules_3_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_11 = layer_output_10 + hidden_states_11
        layer_output_10 = hidden_states_11 = None
        layer_norm_8 = torch.nn.functional.layer_norm(
            layer_output_11,
            (768,),
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
        view_13 = linear_24.view(1, -1, 12, 64)
        linear_24 = None
        key_layer_4 = view_13.transpose(1, 2)
        view_13 = None
        linear_25 = torch._C._nn.linear(
            layer_norm_8,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_14 = linear_25.view(1, -1, 12, 64)
        linear_25 = None
        value_layer_4 = view_14.transpose(1, 2)
        view_14 = None
        linear_26 = torch._C._nn.linear(
            layer_norm_8,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_8 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_15 = linear_26.view(1, -1, 12, 64)
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
        context_layer_4 = attn_output_9.reshape((1, 257, 768))
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
        hidden_states_14 = attention_output_4 + layer_output_11
        attention_output_4 = layer_output_11 = None
        layer_output_12 = torch.nn.functional.layer_norm(
            hidden_states_14,
            (768,),
            l_self_modules_encoder_modules_layer_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_norm2_parameters_bias_ = (None)
        hidden_state_12 = torch._C._nn.linear(
            layer_output_12,
            l_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_output_12 = l_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        hidden_state_13 = torch._C._nn.gelu(hidden_state_12)
        hidden_state_12 = None
        hidden_state_14 = torch._C._nn.linear(
            hidden_state_13,
            l_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_state_13 = l_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        layer_output_13 = (
            hidden_state_14
            * l_self_modules_encoder_modules_layer_modules_4_modules_layer_scale2_parameters_lambda1_
        )
        hidden_state_14 = l_self_modules_encoder_modules_layer_modules_4_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_14 = layer_output_13 + hidden_states_14
        layer_output_13 = hidden_states_14 = None
        layer_norm_10 = torch.nn.functional.layer_norm(
            layer_output_14,
            (768,),
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
        view_16 = linear_30.view(1, -1, 12, 64)
        linear_30 = None
        key_layer_5 = view_16.transpose(1, 2)
        view_16 = None
        linear_31 = torch._C._nn.linear(
            layer_norm_10,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_17 = linear_31.view(1, -1, 12, 64)
        linear_31 = None
        value_layer_5 = view_17.transpose(1, 2)
        view_17 = None
        linear_32 = torch._C._nn.linear(
            layer_norm_10,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_10 = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_18 = linear_32.view(1, -1, 12, 64)
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
        context_layer_5 = attn_output_11.reshape((1, 257, 768))
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
        hidden_states_17 = attention_output_5 + layer_output_14
        attention_output_5 = layer_output_14 = None
        layer_output_15 = torch.nn.functional.layer_norm(
            hidden_states_17,
            (768,),
            l_self_modules_encoder_modules_layer_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_norm2_parameters_bias_ = (None)
        hidden_state_15 = torch._C._nn.linear(
            layer_output_15,
            l_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_output_15 = l_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        hidden_state_16 = torch._C._nn.gelu(hidden_state_15)
        hidden_state_15 = None
        hidden_state_17 = torch._C._nn.linear(
            hidden_state_16,
            l_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_state_16 = l_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        layer_output_16 = (
            hidden_state_17
            * l_self_modules_encoder_modules_layer_modules_5_modules_layer_scale2_parameters_lambda1_
        )
        hidden_state_17 = l_self_modules_encoder_modules_layer_modules_5_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_17 = layer_output_16 + hidden_states_17
        layer_output_16 = hidden_states_17 = None
        layer_norm_12 = torch.nn.functional.layer_norm(
            layer_output_17,
            (768,),
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
        view_19 = linear_36.view(1, -1, 12, 64)
        linear_36 = None
        key_layer_6 = view_19.transpose(1, 2)
        view_19 = None
        linear_37 = torch._C._nn.linear(
            layer_norm_12,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_20 = linear_37.view(1, -1, 12, 64)
        linear_37 = None
        value_layer_6 = view_20.transpose(1, 2)
        view_20 = None
        linear_38 = torch._C._nn.linear(
            layer_norm_12,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_12 = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_21 = linear_38.view(1, -1, 12, 64)
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
        context_layer_6 = attn_output_13.reshape((1, 257, 768))
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
        hidden_states_20 = attention_output_6 + layer_output_17
        attention_output_6 = layer_output_17 = None
        layer_output_18 = torch.nn.functional.layer_norm(
            hidden_states_20,
            (768,),
            l_self_modules_encoder_modules_layer_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_norm2_parameters_bias_ = (None)
        hidden_state_18 = torch._C._nn.linear(
            layer_output_18,
            l_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_output_18 = l_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (None)
        hidden_state_19 = torch._C._nn.gelu(hidden_state_18)
        hidden_state_18 = None
        hidden_state_20 = torch._C._nn.linear(
            hidden_state_19,
            l_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_state_19 = l_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (None)
        layer_output_19 = (
            hidden_state_20
            * l_self_modules_encoder_modules_layer_modules_6_modules_layer_scale2_parameters_lambda1_
        )
        hidden_state_20 = l_self_modules_encoder_modules_layer_modules_6_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_20 = layer_output_19 + hidden_states_20
        layer_output_19 = hidden_states_20 = None
        layer_norm_14 = torch.nn.functional.layer_norm(
            layer_output_20,
            (768,),
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
        view_22 = linear_42.view(1, -1, 12, 64)
        linear_42 = None
        key_layer_7 = view_22.transpose(1, 2)
        view_22 = None
        linear_43 = torch._C._nn.linear(
            layer_norm_14,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_23 = linear_43.view(1, -1, 12, 64)
        linear_43 = None
        value_layer_7 = view_23.transpose(1, 2)
        view_23 = None
        linear_44 = torch._C._nn.linear(
            layer_norm_14,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_14 = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_24 = linear_44.view(1, -1, 12, 64)
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
        context_layer_7 = attn_output_15.reshape((1, 257, 768))
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
        hidden_states_23 = attention_output_7 + layer_output_20
        attention_output_7 = layer_output_20 = None
        layer_output_21 = torch.nn.functional.layer_norm(
            hidden_states_23,
            (768,),
            l_self_modules_encoder_modules_layer_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_norm2_parameters_bias_ = (None)
        hidden_state_21 = torch._C._nn.linear(
            layer_output_21,
            l_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_output_21 = l_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (None)
        hidden_state_22 = torch._C._nn.gelu(hidden_state_21)
        hidden_state_21 = None
        hidden_state_23 = torch._C._nn.linear(
            hidden_state_22,
            l_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_state_22 = l_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (None)
        layer_output_22 = (
            hidden_state_23
            * l_self_modules_encoder_modules_layer_modules_7_modules_layer_scale2_parameters_lambda1_
        )
        hidden_state_23 = l_self_modules_encoder_modules_layer_modules_7_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_23 = layer_output_22 + hidden_states_23
        layer_output_22 = hidden_states_23 = None
        layer_norm_16 = torch.nn.functional.layer_norm(
            layer_output_23,
            (768,),
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
        view_25 = linear_48.view(1, -1, 12, 64)
        linear_48 = None
        key_layer_8 = view_25.transpose(1, 2)
        view_25 = None
        linear_49 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_26 = linear_49.view(1, -1, 12, 64)
        linear_49 = None
        value_layer_8 = view_26.transpose(1, 2)
        view_26 = None
        linear_50 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_16 = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_27 = linear_50.view(1, -1, 12, 64)
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
        context_layer_8 = attn_output_17.reshape((1, 257, 768))
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
        hidden_states_26 = attention_output_8 + layer_output_23
        attention_output_8 = layer_output_23 = None
        layer_output_24 = torch.nn.functional.layer_norm(
            hidden_states_26,
            (768,),
            l_self_modules_encoder_modules_layer_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_norm2_parameters_bias_ = (None)
        hidden_state_24 = torch._C._nn.linear(
            layer_output_24,
            l_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_output_24 = l_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_fc1_parameters_bias_ = (None)
        hidden_state_25 = torch._C._nn.gelu(hidden_state_24)
        hidden_state_24 = None
        hidden_state_26 = torch._C._nn.linear(
            hidden_state_25,
            l_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_state_25 = l_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_mlp_modules_fc2_parameters_bias_ = (None)
        layer_output_25 = (
            hidden_state_26
            * l_self_modules_encoder_modules_layer_modules_8_modules_layer_scale2_parameters_lambda1_
        )
        hidden_state_26 = l_self_modules_encoder_modules_layer_modules_8_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_26 = layer_output_25 + hidden_states_26
        layer_output_25 = hidden_states_26 = None
        layer_norm_18 = torch.nn.functional.layer_norm(
            layer_output_26,
            (768,),
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
        view_28 = linear_54.view(1, -1, 12, 64)
        linear_54 = None
        key_layer_9 = view_28.transpose(1, 2)
        view_28 = None
        linear_55 = torch._C._nn.linear(
            layer_norm_18,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_29 = linear_55.view(1, -1, 12, 64)
        linear_55 = None
        value_layer_9 = view_29.transpose(1, 2)
        view_29 = None
        linear_56 = torch._C._nn.linear(
            layer_norm_18,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_18 = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_30 = linear_56.view(1, -1, 12, 64)
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
        context_layer_9 = attn_output_19.reshape((1, 257, 768))
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
        hidden_states_29 = attention_output_9 + layer_output_26
        attention_output_9 = layer_output_26 = None
        layer_output_27 = torch.nn.functional.layer_norm(
            hidden_states_29,
            (768,),
            l_self_modules_encoder_modules_layer_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_norm2_parameters_bias_ = (None)
        hidden_state_27 = torch._C._nn.linear(
            layer_output_27,
            l_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_output_27 = l_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_fc1_parameters_bias_ = (None)
        hidden_state_28 = torch._C._nn.gelu(hidden_state_27)
        hidden_state_27 = None
        hidden_state_29 = torch._C._nn.linear(
            hidden_state_28,
            l_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_state_28 = l_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_mlp_modules_fc2_parameters_bias_ = (None)
        layer_output_28 = (
            hidden_state_29
            * l_self_modules_encoder_modules_layer_modules_9_modules_layer_scale2_parameters_lambda1_
        )
        hidden_state_29 = l_self_modules_encoder_modules_layer_modules_9_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_29 = layer_output_28 + hidden_states_29
        layer_output_28 = hidden_states_29 = None
        layer_norm_20 = torch.nn.functional.layer_norm(
            layer_output_29,
            (768,),
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
        view_31 = linear_60.view(1, -1, 12, 64)
        linear_60 = None
        key_layer_10 = view_31.transpose(1, 2)
        view_31 = None
        linear_61 = torch._C._nn.linear(
            layer_norm_20,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_32 = linear_61.view(1, -1, 12, 64)
        linear_61 = None
        value_layer_10 = view_32.transpose(1, 2)
        view_32 = None
        linear_62 = torch._C._nn.linear(
            layer_norm_20,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_20 = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_33 = linear_62.view(1, -1, 12, 64)
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
        context_layer_10 = attn_output_21.reshape((1, 257, 768))
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
        hidden_states_32 = attention_output_10 + layer_output_29
        attention_output_10 = layer_output_29 = None
        layer_output_30 = torch.nn.functional.layer_norm(
            hidden_states_32,
            (768,),
            l_self_modules_encoder_modules_layer_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_norm2_parameters_bias_ = (None)
        hidden_state_30 = torch._C._nn.linear(
            layer_output_30,
            l_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_output_30 = l_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_fc1_parameters_bias_ = (None)
        hidden_state_31 = torch._C._nn.gelu(hidden_state_30)
        hidden_state_30 = None
        hidden_state_32 = torch._C._nn.linear(
            hidden_state_31,
            l_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_state_31 = l_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_mlp_modules_fc2_parameters_bias_ = (None)
        layer_output_31 = (
            hidden_state_32
            * l_self_modules_encoder_modules_layer_modules_10_modules_layer_scale2_parameters_lambda1_
        )
        hidden_state_32 = l_self_modules_encoder_modules_layer_modules_10_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_32 = layer_output_31 + hidden_states_32
        layer_output_31 = hidden_states_32 = None
        layer_norm_22 = torch.nn.functional.layer_norm(
            layer_output_32,
            (768,),
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
        view_34 = linear_66.view(1, -1, 12, 64)
        linear_66 = None
        key_layer_11 = view_34.transpose(1, 2)
        view_34 = None
        linear_67 = torch._C._nn.linear(
            layer_norm_22,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_35 = linear_67.view(1, -1, 12, 64)
        linear_67 = None
        value_layer_11 = view_35.transpose(1, 2)
        view_35 = None
        linear_68 = torch._C._nn.linear(
            layer_norm_22,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_22 = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_36 = linear_68.view(1, -1, 12, 64)
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
        context_layer_11 = attn_output_23.reshape((1, 257, 768))
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
        hidden_states_35 = attention_output_11 + layer_output_32
        attention_output_11 = layer_output_32 = None
        layer_output_33 = torch.nn.functional.layer_norm(
            hidden_states_35,
            (768,),
            l_self_modules_encoder_modules_layer_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_norm2_parameters_bias_ = (None)
        hidden_state_33 = torch._C._nn.linear(
            layer_output_33,
            l_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_output_33 = l_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_fc1_parameters_bias_ = (None)
        hidden_state_34 = torch._C._nn.gelu(hidden_state_33)
        hidden_state_33 = None
        hidden_state_35 = torch._C._nn.linear(
            hidden_state_34,
            l_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_state_34 = l_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_mlp_modules_fc2_parameters_bias_ = (None)
        layer_output_34 = (
            hidden_state_35
            * l_self_modules_encoder_modules_layer_modules_11_modules_layer_scale2_parameters_lambda1_
        )
        hidden_state_35 = l_self_modules_encoder_modules_layer_modules_11_modules_layer_scale2_parameters_lambda1_ = (None)
        layer_output_35 = layer_output_34 + hidden_states_35
        layer_output_34 = hidden_states_35 = None
        sequence_output = torch.nn.functional.layer_norm(
            layer_output_35,
            (768,),
            l_self_modules_layernorm_parameters_weight_,
            l_self_modules_layernorm_parameters_bias_,
            1e-06,
        )
        layer_output_35 = (
            l_self_modules_layernorm_parameters_weight_
        ) = l_self_modules_layernorm_parameters_bias_ = None
        pooled_output = sequence_output[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        return (sequence_output, pooled_output)
