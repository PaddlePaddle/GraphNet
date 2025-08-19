import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_pixel_values_: torch.Tensor,
        L_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_parameters_cls_token_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_parameters_lambda_1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_parameters_lambda_2_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_parameters_lambda_1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_parameters_lambda_2_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_parameters_lambda_1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_parameters_lambda_2_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_parameters_lambda_1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_parameters_lambda_2_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_parameters_lambda_1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_parameters_lambda_2_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_parameters_lambda_1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_parameters_lambda_2_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_parameters_lambda_1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_parameters_lambda_2_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_parameters_lambda_1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_parameters_lambda_2_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_parameters_lambda_1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_parameters_lambda_2_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_parameters_lambda_1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_parameters_lambda_2_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_parameters_lambda_1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_parameters_lambda_2_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_parameters_lambda_1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_parameters_lambda_2_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_parameters_lambda_1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_parameters_lambda_2_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_parameters_lambda_1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_parameters_lambda_2_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_parameters_lambda_1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_parameters_lambda_2_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_parameters_lambda_1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_parameters_lambda_2_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_parameters_lambda_1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_parameters_lambda_2_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_parameters_lambda_1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_parameters_lambda_2_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_parameters_lambda_1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_parameters_lambda_2_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_parameters_lambda_1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_parameters_lambda_2_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_parameters_lambda_1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_parameters_lambda_2_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_parameters_lambda_1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_parameters_lambda_2_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_parameters_lambda_1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_parameters_lambda_2_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_parameters_lambda_1_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_parameters_lambda_2_: torch.nn.parameter.Parameter,
        L_self_modules_pooler_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_pooler_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_pixel_values_ = L_pixel_values_
        l_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_ = L_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_
        l_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_ = L_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_
        l_self_modules_embeddings_parameters_cls_token_ = (
            L_self_modules_embeddings_parameters_cls_token_
        )
        l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_ = L_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_
        l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_parameters_lambda_1_ = (
            L_self_modules_encoder_modules_layer_modules_0_parameters_lambda_1_
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_parameters_lambda_2_ = (
            L_self_modules_encoder_modules_layer_modules_0_parameters_lambda_2_
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_parameters_lambda_1_ = (
            L_self_modules_encoder_modules_layer_modules_1_parameters_lambda_1_
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_parameters_lambda_2_ = (
            L_self_modules_encoder_modules_layer_modules_1_parameters_lambda_2_
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_parameters_lambda_1_ = (
            L_self_modules_encoder_modules_layer_modules_2_parameters_lambda_1_
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_parameters_lambda_2_ = (
            L_self_modules_encoder_modules_layer_modules_2_parameters_lambda_2_
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_parameters_lambda_1_ = (
            L_self_modules_encoder_modules_layer_modules_3_parameters_lambda_1_
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_parameters_lambda_2_ = (
            L_self_modules_encoder_modules_layer_modules_3_parameters_lambda_2_
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_parameters_lambda_1_ = (
            L_self_modules_encoder_modules_layer_modules_4_parameters_lambda_1_
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_parameters_lambda_2_ = (
            L_self_modules_encoder_modules_layer_modules_4_parameters_lambda_2_
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_parameters_lambda_1_ = (
            L_self_modules_encoder_modules_layer_modules_5_parameters_lambda_1_
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_parameters_lambda_2_ = (
            L_self_modules_encoder_modules_layer_modules_5_parameters_lambda_2_
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_parameters_lambda_1_ = (
            L_self_modules_encoder_modules_layer_modules_6_parameters_lambda_1_
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_parameters_lambda_2_ = (
            L_self_modules_encoder_modules_layer_modules_6_parameters_lambda_2_
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_parameters_lambda_1_ = (
            L_self_modules_encoder_modules_layer_modules_7_parameters_lambda_1_
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_parameters_lambda_2_ = (
            L_self_modules_encoder_modules_layer_modules_7_parameters_lambda_2_
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_parameters_lambda_1_ = (
            L_self_modules_encoder_modules_layer_modules_8_parameters_lambda_1_
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_parameters_lambda_2_ = (
            L_self_modules_encoder_modules_layer_modules_8_parameters_lambda_2_
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_parameters_lambda_1_ = (
            L_self_modules_encoder_modules_layer_modules_9_parameters_lambda_1_
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_parameters_lambda_2_ = (
            L_self_modules_encoder_modules_layer_modules_9_parameters_lambda_2_
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_parameters_lambda_1_ = (
            L_self_modules_encoder_modules_layer_modules_10_parameters_lambda_1_
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_parameters_lambda_2_ = (
            L_self_modules_encoder_modules_layer_modules_10_parameters_lambda_2_
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_parameters_lambda_1_ = (
            L_self_modules_encoder_modules_layer_modules_11_parameters_lambda_1_
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_parameters_lambda_2_ = (
            L_self_modules_encoder_modules_layer_modules_11_parameters_lambda_2_
        )
        l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_parameters_lambda_1_ = (
            L_self_modules_encoder_modules_layer_modules_12_parameters_lambda_1_
        )
        l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_parameters_lambda_2_ = (
            L_self_modules_encoder_modules_layer_modules_12_parameters_lambda_2_
        )
        l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_parameters_lambda_1_ = (
            L_self_modules_encoder_modules_layer_modules_13_parameters_lambda_1_
        )
        l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_parameters_lambda_2_ = (
            L_self_modules_encoder_modules_layer_modules_13_parameters_lambda_2_
        )
        l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_parameters_lambda_1_ = (
            L_self_modules_encoder_modules_layer_modules_14_parameters_lambda_1_
        )
        l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_parameters_lambda_2_ = (
            L_self_modules_encoder_modules_layer_modules_14_parameters_lambda_2_
        )
        l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_parameters_lambda_1_ = (
            L_self_modules_encoder_modules_layer_modules_15_parameters_lambda_1_
        )
        l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_parameters_lambda_2_ = (
            L_self_modules_encoder_modules_layer_modules_15_parameters_lambda_2_
        )
        l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_parameters_lambda_1_ = (
            L_self_modules_encoder_modules_layer_modules_16_parameters_lambda_1_
        )
        l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_parameters_lambda_2_ = (
            L_self_modules_encoder_modules_layer_modules_16_parameters_lambda_2_
        )
        l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_parameters_lambda_1_ = (
            L_self_modules_encoder_modules_layer_modules_17_parameters_lambda_1_
        )
        l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_parameters_lambda_2_ = (
            L_self_modules_encoder_modules_layer_modules_17_parameters_lambda_2_
        )
        l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_parameters_lambda_1_ = (
            L_self_modules_encoder_modules_layer_modules_18_parameters_lambda_1_
        )
        l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_parameters_lambda_2_ = (
            L_self_modules_encoder_modules_layer_modules_18_parameters_lambda_2_
        )
        l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_parameters_lambda_1_ = (
            L_self_modules_encoder_modules_layer_modules_19_parameters_lambda_1_
        )
        l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_parameters_lambda_2_ = (
            L_self_modules_encoder_modules_layer_modules_19_parameters_lambda_2_
        )
        l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_parameters_lambda_1_ = (
            L_self_modules_encoder_modules_layer_modules_20_parameters_lambda_1_
        )
        l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_parameters_lambda_2_ = (
            L_self_modules_encoder_modules_layer_modules_20_parameters_lambda_2_
        )
        l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_parameters_lambda_1_ = (
            L_self_modules_encoder_modules_layer_modules_21_parameters_lambda_1_
        )
        l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_parameters_lambda_2_ = (
            L_self_modules_encoder_modules_layer_modules_21_parameters_lambda_2_
        )
        l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_parameters_lambda_1_ = (
            L_self_modules_encoder_modules_layer_modules_22_parameters_lambda_1_
        )
        l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_parameters_lambda_2_ = (
            L_self_modules_encoder_modules_layer_modules_22_parameters_lambda_2_
        )
        l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_parameters_lambda_1_ = (
            L_self_modules_encoder_modules_layer_modules_23_parameters_lambda_1_
        )
        l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_parameters_lambda_2_ = (
            L_self_modules_encoder_modules_layer_modules_23_parameters_lambda_2_
        )
        l_self_modules_pooler_modules_layernorm_parameters_weight_ = (
            L_self_modules_pooler_modules_layernorm_parameters_weight_
        )
        l_self_modules_pooler_modules_layernorm_parameters_bias_ = (
            L_self_modules_pooler_modules_layernorm_parameters_bias_
        )
        embeddings = torch.conv2d(
            l_pixel_values_,
            l_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_,
            l_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_,
            (16, 16),
            (0, 0),
            (1, 1),
            1,
        )
        l_pixel_values_ = l_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_ = l_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_ = (None)
        flatten = embeddings.flatten(2)
        embeddings = None
        embeddings_1 = flatten.transpose(1, 2)
        flatten = None
        cls_tokens = l_self_modules_embeddings_parameters_cls_token_.expand(1, -1, -1)
        l_self_modules_embeddings_parameters_cls_token_ = None
        embeddings_2 = torch.cat((cls_tokens, embeddings_1), dim=1)
        cls_tokens = embeddings_1 = None
        embeddings_3 = torch.nn.functional.dropout(embeddings_2, 0.0, False, False)
        embeddings_2 = None
        old_sub_table = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(None, 729, None)
        ]
        reshape = old_sub_table.reshape(1, 27, 27, -1)
        old_sub_table = None
        old_sub_table_1 = reshape.permute(0, 3, 1, 2)
        reshape = None
        new_sub_table = torch.nn.functional.interpolate(
            old_sub_table_1, size=(27, 27), mode="bilinear"
        )
        old_sub_table_1 = None
        permute_1 = new_sub_table.permute(0, 2, 3, 1)
        new_sub_table = None
        new_sub_table_1 = permute_1.reshape(729, -1)
        permute_1 = None
        getitem_1 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(729, None, None)
        ]
        new_relative_position_bias_table = torch.cat([new_sub_table_1, getitem_1])
        new_sub_table_1 = getitem_1 = None
        arange = torch.arange(14)
        arange_1 = torch.arange(14)
        meshgrid = torch.functional.meshgrid(arange, arange_1, indexing="ij")
        arange = arange_1 = None
        getitem_2 = meshgrid[0]
        getitem_3 = meshgrid[1]
        meshgrid = None
        coords = torch.stack((getitem_2, getitem_3))
        getitem_2 = getitem_3 = None
        coords_flatten = torch.flatten(coords, 1)
        coords = None
        getitem_4 = coords_flatten[
            (slice(None, None, None), slice(None, None, None), None)
        ]
        getitem_5 = coords_flatten[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        coords_flatten = None
        relative_coords = getitem_4 - getitem_5
        getitem_4 = getitem_5 = None
        permute_2 = relative_coords.permute(1, 2, 0)
        relative_coords = None
        relative_coords_1 = permute_2.contiguous()
        permute_2 = None
        getitem_6 = relative_coords_1[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_6 += 13
        iadd = getitem_6
        getitem_6 = None
        relative_coords_1[(slice(None, None, None), slice(None, None, None), 0)] = iadd
        setitem = relative_coords_1
        iadd = setitem = None
        getitem_7 = relative_coords_1[
            (slice(None, None, None), slice(None, None, None), 1)
        ]
        getitem_7 += 13
        iadd_1 = getitem_7
        getitem_7 = None
        relative_coords_1[
            (slice(None, None, None), slice(None, None, None), 1)
        ] = iadd_1
        setitem_1 = relative_coords_1
        iadd_1 = setitem_1 = None
        getitem_8 = relative_coords_1[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_8 *= 27
        imul = getitem_8
        getitem_8 = None
        relative_coords_1[(slice(None, None, None), slice(None, None, None), 0)] = imul
        setitem_2 = relative_coords_1
        imul = setitem_2 = None
        relative_position_index = torch.zeros(size=(197, 197), dtype=torch.int64)
        sum_1 = relative_coords_1.sum(-1)
        relative_coords_1 = None
        relative_position_index[(slice(1, None, None), slice(1, None, None))] = sum_1
        setitem_3 = relative_position_index
        sum_1 = setitem_3 = None
        relative_position_index[(0, slice(0, None, None))] = 729
        setitem_4 = relative_position_index
        setitem_4 = None
        relative_position_index[(slice(0, None, None), 0)] = 730
        setitem_5 = relative_position_index
        setitem_5 = None
        relative_position_index[(0, 0)] = 731
        setitem_6 = relative_position_index
        setitem_6 = None
        view = relative_position_index.view(-1)
        relative_position_index = None
        relative_position_bias = new_relative_position_bias_table[view]
        new_relative_position_bias_table = view = None
        relative_position_bias_1 = relative_position_bias.view(197, 197, -1)
        relative_position_bias = None
        permute_3 = relative_position_bias_1.permute(2, 0, 1)
        relative_position_bias_1 = None
        relative_position_bias_2 = permute_3.contiguous()
        permute_3 = None
        relative_position_bias_3 = relative_position_bias_2.unsqueeze(0)
        relative_position_bias_2 = None
        layer_norm = torch.nn.functional.layer_norm(
            embeddings_3,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_bias_ = (None)
        linear = torch._C._nn.linear(
            layer_norm,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_2 = linear.view(1, -1, 16, 64)
        linear = None
        query_layer = view_2.transpose(1, 2)
        view_2 = None
        linear_1 = torch._C._nn.linear(
            layer_norm,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_3 = linear_1.view(1, -1, 16, 64)
        linear_1 = None
        key_layer = view_3.transpose(1, 2)
        view_3 = None
        linear_2 = torch._C._nn.linear(
            layer_norm,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        layer_norm = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_4 = linear_2.view(1, -1, 16, 64)
        linear_2 = None
        value_layer = view_4.transpose(1, 2)
        view_4 = None
        context_layer = torch._C._nn.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=relative_position_bias_3,
            dropout_p=0.0,
            is_causal=False,
            scale=0.125,
        )
        query_layer = key_layer = value_layer = relative_position_bias_3 = None
        permute_4 = context_layer.permute(0, 2, 1, 3)
        context_layer = None
        context_layer_1 = permute_4.contiguous()
        permute_4 = None
        context_layer_2 = context_layer_1.view(1, 197, 1024)
        context_layer_1 = None
        hidden_states = torch._C._nn.linear(
            context_layer_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_2 = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_1 = torch.nn.functional.dropout(hidden_states, 0.0, False, False)
        hidden_states = None
        attention_output = (
            l_self_modules_encoder_modules_layer_modules_0_parameters_lambda_1_
            * hidden_states_1
        )
        l_self_modules_encoder_modules_layer_modules_0_parameters_lambda_1_ = (
            hidden_states_1
        ) = None
        hidden_states_2 = attention_output + embeddings_3
        attention_output = embeddings_3 = None
        layer_output = torch.nn.functional.layer_norm(
            hidden_states_2,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_bias_ = (None)
        hidden_states_3 = torch._C._nn.linear(
            layer_output,
            l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output = l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_4 = torch._C._nn.gelu(hidden_states_3)
        hidden_states_3 = None
        hidden_states_5 = torch._C._nn.linear(
            hidden_states_4,
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_4 = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_6 = torch.nn.functional.dropout(
            hidden_states_5, 0.0, False, False
        )
        hidden_states_5 = None
        layer_output_1 = (
            l_self_modules_encoder_modules_layer_modules_0_parameters_lambda_2_
            * hidden_states_6
        )
        l_self_modules_encoder_modules_layer_modules_0_parameters_lambda_2_ = (
            hidden_states_6
        ) = None
        layer_output_2 = layer_output_1 + hidden_states_2
        layer_output_1 = hidden_states_2 = None
        old_sub_table_2 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(None, 729, None)
        ]
        reshape_2 = old_sub_table_2.reshape(1, 27, 27, -1)
        old_sub_table_2 = None
        old_sub_table_3 = reshape_2.permute(0, 3, 1, 2)
        reshape_2 = None
        new_sub_table_2 = torch.nn.functional.interpolate(
            old_sub_table_3, size=(27, 27), mode="bilinear"
        )
        old_sub_table_3 = None
        permute_6 = new_sub_table_2.permute(0, 2, 3, 1)
        new_sub_table_2 = None
        new_sub_table_3 = permute_6.reshape(729, -1)
        permute_6 = None
        getitem_11 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(729, None, None)
        ]
        new_relative_position_bias_table_1 = torch.cat([new_sub_table_3, getitem_11])
        new_sub_table_3 = getitem_11 = None
        arange_2 = torch.arange(14)
        arange_3 = torch.arange(14)
        meshgrid_1 = torch.functional.meshgrid(arange_2, arange_3, indexing="ij")
        arange_2 = arange_3 = None
        getitem_12 = meshgrid_1[0]
        getitem_13 = meshgrid_1[1]
        meshgrid_1 = None
        coords_1 = torch.stack((getitem_12, getitem_13))
        getitem_12 = getitem_13 = None
        coords_flatten_1 = torch.flatten(coords_1, 1)
        coords_1 = None
        getitem_14 = coords_flatten_1[
            (slice(None, None, None), slice(None, None, None), None)
        ]
        getitem_15 = coords_flatten_1[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        coords_flatten_1 = None
        relative_coords_2 = getitem_14 - getitem_15
        getitem_14 = getitem_15 = None
        permute_7 = relative_coords_2.permute(1, 2, 0)
        relative_coords_2 = None
        relative_coords_3 = permute_7.contiguous()
        permute_7 = None
        getitem_16 = relative_coords_3[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_16 += 13
        iadd_2 = getitem_16
        getitem_16 = None
        relative_coords_3[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = iadd_2
        setitem_7 = relative_coords_3
        iadd_2 = setitem_7 = None
        getitem_17 = relative_coords_3[
            (slice(None, None, None), slice(None, None, None), 1)
        ]
        getitem_17 += 13
        iadd_3 = getitem_17
        getitem_17 = None
        relative_coords_3[
            (slice(None, None, None), slice(None, None, None), 1)
        ] = iadd_3
        setitem_8 = relative_coords_3
        iadd_3 = setitem_8 = None
        getitem_18 = relative_coords_3[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_18 *= 27
        imul_1 = getitem_18
        getitem_18 = None
        relative_coords_3[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = imul_1
        setitem_9 = relative_coords_3
        imul_1 = setitem_9 = None
        relative_position_index_1 = torch.zeros(size=(197, 197), dtype=torch.int64)
        sum_2 = relative_coords_3.sum(-1)
        relative_coords_3 = None
        relative_position_index_1[(slice(1, None, None), slice(1, None, None))] = sum_2
        setitem_10 = relative_position_index_1
        sum_2 = setitem_10 = None
        relative_position_index_1[(0, slice(0, None, None))] = 729
        setitem_11 = relative_position_index_1
        setitem_11 = None
        relative_position_index_1[(slice(0, None, None), 0)] = 730
        setitem_12 = relative_position_index_1
        setitem_12 = None
        relative_position_index_1[(0, 0)] = 731
        setitem_13 = relative_position_index_1
        setitem_13 = None
        view_6 = relative_position_index_1.view(-1)
        relative_position_index_1 = None
        relative_position_bias_4 = new_relative_position_bias_table_1[view_6]
        new_relative_position_bias_table_1 = view_6 = None
        relative_position_bias_5 = relative_position_bias_4.view(197, 197, -1)
        relative_position_bias_4 = None
        permute_8 = relative_position_bias_5.permute(2, 0, 1)
        relative_position_bias_5 = None
        relative_position_bias_6 = permute_8.contiguous()
        permute_8 = None
        relative_position_bias_7 = relative_position_bias_6.unsqueeze(0)
        relative_position_bias_6 = None
        layer_norm_2 = torch.nn.functional.layer_norm(
            layer_output_2,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_bias_ = (None)
        linear_6 = torch._C._nn.linear(
            layer_norm_2,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_8 = linear_6.view(1, -1, 16, 64)
        linear_6 = None
        query_layer_1 = view_8.transpose(1, 2)
        view_8 = None
        linear_7 = torch._C._nn.linear(
            layer_norm_2,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_9 = linear_7.view(1, -1, 16, 64)
        linear_7 = None
        key_layer_1 = view_9.transpose(1, 2)
        view_9 = None
        linear_8 = torch._C._nn.linear(
            layer_norm_2,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        layer_norm_2 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_10 = linear_8.view(1, -1, 16, 64)
        linear_8 = None
        value_layer_1 = view_10.transpose(1, 2)
        view_10 = None
        context_layer_3 = torch._C._nn.scaled_dot_product_attention(
            query_layer_1,
            key_layer_1,
            value_layer_1,
            attn_mask=relative_position_bias_7,
            dropout_p=0.0,
            is_causal=False,
            scale=0.125,
        )
        query_layer_1 = key_layer_1 = value_layer_1 = relative_position_bias_7 = None
        permute_9 = context_layer_3.permute(0, 2, 1, 3)
        context_layer_3 = None
        context_layer_4 = permute_9.contiguous()
        permute_9 = None
        context_layer_5 = context_layer_4.view(1, 197, 1024)
        context_layer_4 = None
        hidden_states_7 = torch._C._nn.linear(
            context_layer_5,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_5 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_8 = torch.nn.functional.dropout(
            hidden_states_7, 0.0, False, False
        )
        hidden_states_7 = None
        attention_output_1 = (
            l_self_modules_encoder_modules_layer_modules_1_parameters_lambda_1_
            * hidden_states_8
        )
        l_self_modules_encoder_modules_layer_modules_1_parameters_lambda_1_ = (
            hidden_states_8
        ) = None
        hidden_states_9 = attention_output_1 + layer_output_2
        attention_output_1 = layer_output_2 = None
        layer_output_3 = torch.nn.functional.layer_norm(
            hidden_states_9,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_bias_ = (None)
        hidden_states_10 = torch._C._nn.linear(
            layer_output_3,
            l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_3 = l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_11 = torch._C._nn.gelu(hidden_states_10)
        hidden_states_10 = None
        hidden_states_12 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_11 = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_13 = torch.nn.functional.dropout(
            hidden_states_12, 0.0, False, False
        )
        hidden_states_12 = None
        layer_output_4 = (
            l_self_modules_encoder_modules_layer_modules_1_parameters_lambda_2_
            * hidden_states_13
        )
        l_self_modules_encoder_modules_layer_modules_1_parameters_lambda_2_ = (
            hidden_states_13
        ) = None
        layer_output_5 = layer_output_4 + hidden_states_9
        layer_output_4 = hidden_states_9 = None
        old_sub_table_4 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(None, 729, None)
        ]
        reshape_4 = old_sub_table_4.reshape(1, 27, 27, -1)
        old_sub_table_4 = None
        old_sub_table_5 = reshape_4.permute(0, 3, 1, 2)
        reshape_4 = None
        new_sub_table_4 = torch.nn.functional.interpolate(
            old_sub_table_5, size=(27, 27), mode="bilinear"
        )
        old_sub_table_5 = None
        permute_11 = new_sub_table_4.permute(0, 2, 3, 1)
        new_sub_table_4 = None
        new_sub_table_5 = permute_11.reshape(729, -1)
        permute_11 = None
        getitem_21 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(729, None, None)
        ]
        new_relative_position_bias_table_2 = torch.cat([new_sub_table_5, getitem_21])
        new_sub_table_5 = getitem_21 = None
        arange_4 = torch.arange(14)
        arange_5 = torch.arange(14)
        meshgrid_2 = torch.functional.meshgrid(arange_4, arange_5, indexing="ij")
        arange_4 = arange_5 = None
        getitem_22 = meshgrid_2[0]
        getitem_23 = meshgrid_2[1]
        meshgrid_2 = None
        coords_2 = torch.stack((getitem_22, getitem_23))
        getitem_22 = getitem_23 = None
        coords_flatten_2 = torch.flatten(coords_2, 1)
        coords_2 = None
        getitem_24 = coords_flatten_2[
            (slice(None, None, None), slice(None, None, None), None)
        ]
        getitem_25 = coords_flatten_2[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        coords_flatten_2 = None
        relative_coords_4 = getitem_24 - getitem_25
        getitem_24 = getitem_25 = None
        permute_12 = relative_coords_4.permute(1, 2, 0)
        relative_coords_4 = None
        relative_coords_5 = permute_12.contiguous()
        permute_12 = None
        getitem_26 = relative_coords_5[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_26 += 13
        iadd_4 = getitem_26
        getitem_26 = None
        relative_coords_5[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = iadd_4
        setitem_14 = relative_coords_5
        iadd_4 = setitem_14 = None
        getitem_27 = relative_coords_5[
            (slice(None, None, None), slice(None, None, None), 1)
        ]
        getitem_27 += 13
        iadd_5 = getitem_27
        getitem_27 = None
        relative_coords_5[
            (slice(None, None, None), slice(None, None, None), 1)
        ] = iadd_5
        setitem_15 = relative_coords_5
        iadd_5 = setitem_15 = None
        getitem_28 = relative_coords_5[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_28 *= 27
        imul_2 = getitem_28
        getitem_28 = None
        relative_coords_5[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = imul_2
        setitem_16 = relative_coords_5
        imul_2 = setitem_16 = None
        relative_position_index_2 = torch.zeros(size=(197, 197), dtype=torch.int64)
        sum_3 = relative_coords_5.sum(-1)
        relative_coords_5 = None
        relative_position_index_2[(slice(1, None, None), slice(1, None, None))] = sum_3
        setitem_17 = relative_position_index_2
        sum_3 = setitem_17 = None
        relative_position_index_2[(0, slice(0, None, None))] = 729
        setitem_18 = relative_position_index_2
        setitem_18 = None
        relative_position_index_2[(slice(0, None, None), 0)] = 730
        setitem_19 = relative_position_index_2
        setitem_19 = None
        relative_position_index_2[(0, 0)] = 731
        setitem_20 = relative_position_index_2
        setitem_20 = None
        view_12 = relative_position_index_2.view(-1)
        relative_position_index_2 = None
        relative_position_bias_8 = new_relative_position_bias_table_2[view_12]
        new_relative_position_bias_table_2 = view_12 = None
        relative_position_bias_9 = relative_position_bias_8.view(197, 197, -1)
        relative_position_bias_8 = None
        permute_13 = relative_position_bias_9.permute(2, 0, 1)
        relative_position_bias_9 = None
        relative_position_bias_10 = permute_13.contiguous()
        permute_13 = None
        relative_position_bias_11 = relative_position_bias_10.unsqueeze(0)
        relative_position_bias_10 = None
        layer_norm_4 = torch.nn.functional.layer_norm(
            layer_output_5,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_bias_ = (None)
        linear_12 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_14 = linear_12.view(1, -1, 16, 64)
        linear_12 = None
        query_layer_2 = view_14.transpose(1, 2)
        view_14 = None
        linear_13 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_15 = linear_13.view(1, -1, 16, 64)
        linear_13 = None
        key_layer_2 = view_15.transpose(1, 2)
        view_15 = None
        linear_14 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        layer_norm_4 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_16 = linear_14.view(1, -1, 16, 64)
        linear_14 = None
        value_layer_2 = view_16.transpose(1, 2)
        view_16 = None
        context_layer_6 = torch._C._nn.scaled_dot_product_attention(
            query_layer_2,
            key_layer_2,
            value_layer_2,
            attn_mask=relative_position_bias_11,
            dropout_p=0.0,
            is_causal=False,
            scale=0.125,
        )
        query_layer_2 = key_layer_2 = value_layer_2 = relative_position_bias_11 = None
        permute_14 = context_layer_6.permute(0, 2, 1, 3)
        context_layer_6 = None
        context_layer_7 = permute_14.contiguous()
        permute_14 = None
        context_layer_8 = context_layer_7.view(1, 197, 1024)
        context_layer_7 = None
        hidden_states_14 = torch._C._nn.linear(
            context_layer_8,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_8 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_15 = torch.nn.functional.dropout(
            hidden_states_14, 0.0, False, False
        )
        hidden_states_14 = None
        attention_output_2 = (
            l_self_modules_encoder_modules_layer_modules_2_parameters_lambda_1_
            * hidden_states_15
        )
        l_self_modules_encoder_modules_layer_modules_2_parameters_lambda_1_ = (
            hidden_states_15
        ) = None
        hidden_states_16 = attention_output_2 + layer_output_5
        attention_output_2 = layer_output_5 = None
        layer_output_6 = torch.nn.functional.layer_norm(
            hidden_states_16,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_bias_ = (None)
        hidden_states_17 = torch._C._nn.linear(
            layer_output_6,
            l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_6 = l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_18 = torch._C._nn.gelu(hidden_states_17)
        hidden_states_17 = None
        hidden_states_19 = torch._C._nn.linear(
            hidden_states_18,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_18 = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_20 = torch.nn.functional.dropout(
            hidden_states_19, 0.0, False, False
        )
        hidden_states_19 = None
        layer_output_7 = (
            l_self_modules_encoder_modules_layer_modules_2_parameters_lambda_2_
            * hidden_states_20
        )
        l_self_modules_encoder_modules_layer_modules_2_parameters_lambda_2_ = (
            hidden_states_20
        ) = None
        layer_output_8 = layer_output_7 + hidden_states_16
        layer_output_7 = hidden_states_16 = None
        old_sub_table_6 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(None, 729, None)
        ]
        reshape_6 = old_sub_table_6.reshape(1, 27, 27, -1)
        old_sub_table_6 = None
        old_sub_table_7 = reshape_6.permute(0, 3, 1, 2)
        reshape_6 = None
        new_sub_table_6 = torch.nn.functional.interpolate(
            old_sub_table_7, size=(27, 27), mode="bilinear"
        )
        old_sub_table_7 = None
        permute_16 = new_sub_table_6.permute(0, 2, 3, 1)
        new_sub_table_6 = None
        new_sub_table_7 = permute_16.reshape(729, -1)
        permute_16 = None
        getitem_31 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(729, None, None)
        ]
        new_relative_position_bias_table_3 = torch.cat([new_sub_table_7, getitem_31])
        new_sub_table_7 = getitem_31 = None
        arange_6 = torch.arange(14)
        arange_7 = torch.arange(14)
        meshgrid_3 = torch.functional.meshgrid(arange_6, arange_7, indexing="ij")
        arange_6 = arange_7 = None
        getitem_32 = meshgrid_3[0]
        getitem_33 = meshgrid_3[1]
        meshgrid_3 = None
        coords_3 = torch.stack((getitem_32, getitem_33))
        getitem_32 = getitem_33 = None
        coords_flatten_3 = torch.flatten(coords_3, 1)
        coords_3 = None
        getitem_34 = coords_flatten_3[
            (slice(None, None, None), slice(None, None, None), None)
        ]
        getitem_35 = coords_flatten_3[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        coords_flatten_3 = None
        relative_coords_6 = getitem_34 - getitem_35
        getitem_34 = getitem_35 = None
        permute_17 = relative_coords_6.permute(1, 2, 0)
        relative_coords_6 = None
        relative_coords_7 = permute_17.contiguous()
        permute_17 = None
        getitem_36 = relative_coords_7[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_36 += 13
        iadd_6 = getitem_36
        getitem_36 = None
        relative_coords_7[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = iadd_6
        setitem_21 = relative_coords_7
        iadd_6 = setitem_21 = None
        getitem_37 = relative_coords_7[
            (slice(None, None, None), slice(None, None, None), 1)
        ]
        getitem_37 += 13
        iadd_7 = getitem_37
        getitem_37 = None
        relative_coords_7[
            (slice(None, None, None), slice(None, None, None), 1)
        ] = iadd_7
        setitem_22 = relative_coords_7
        iadd_7 = setitem_22 = None
        getitem_38 = relative_coords_7[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_38 *= 27
        imul_3 = getitem_38
        getitem_38 = None
        relative_coords_7[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = imul_3
        setitem_23 = relative_coords_7
        imul_3 = setitem_23 = None
        relative_position_index_3 = torch.zeros(size=(197, 197), dtype=torch.int64)
        sum_4 = relative_coords_7.sum(-1)
        relative_coords_7 = None
        relative_position_index_3[(slice(1, None, None), slice(1, None, None))] = sum_4
        setitem_24 = relative_position_index_3
        sum_4 = setitem_24 = None
        relative_position_index_3[(0, slice(0, None, None))] = 729
        setitem_25 = relative_position_index_3
        setitem_25 = None
        relative_position_index_3[(slice(0, None, None), 0)] = 730
        setitem_26 = relative_position_index_3
        setitem_26 = None
        relative_position_index_3[(0, 0)] = 731
        setitem_27 = relative_position_index_3
        setitem_27 = None
        view_18 = relative_position_index_3.view(-1)
        relative_position_index_3 = None
        relative_position_bias_12 = new_relative_position_bias_table_3[view_18]
        new_relative_position_bias_table_3 = view_18 = None
        relative_position_bias_13 = relative_position_bias_12.view(197, 197, -1)
        relative_position_bias_12 = None
        permute_18 = relative_position_bias_13.permute(2, 0, 1)
        relative_position_bias_13 = None
        relative_position_bias_14 = permute_18.contiguous()
        permute_18 = None
        relative_position_bias_15 = relative_position_bias_14.unsqueeze(0)
        relative_position_bias_14 = None
        layer_norm_6 = torch.nn.functional.layer_norm(
            layer_output_8,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_bias_ = (None)
        linear_18 = torch._C._nn.linear(
            layer_norm_6,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_20 = linear_18.view(1, -1, 16, 64)
        linear_18 = None
        query_layer_3 = view_20.transpose(1, 2)
        view_20 = None
        linear_19 = torch._C._nn.linear(
            layer_norm_6,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_21 = linear_19.view(1, -1, 16, 64)
        linear_19 = None
        key_layer_3 = view_21.transpose(1, 2)
        view_21 = None
        linear_20 = torch._C._nn.linear(
            layer_norm_6,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        layer_norm_6 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_22 = linear_20.view(1, -1, 16, 64)
        linear_20 = None
        value_layer_3 = view_22.transpose(1, 2)
        view_22 = None
        context_layer_9 = torch._C._nn.scaled_dot_product_attention(
            query_layer_3,
            key_layer_3,
            value_layer_3,
            attn_mask=relative_position_bias_15,
            dropout_p=0.0,
            is_causal=False,
            scale=0.125,
        )
        query_layer_3 = key_layer_3 = value_layer_3 = relative_position_bias_15 = None
        permute_19 = context_layer_9.permute(0, 2, 1, 3)
        context_layer_9 = None
        context_layer_10 = permute_19.contiguous()
        permute_19 = None
        context_layer_11 = context_layer_10.view(1, 197, 1024)
        context_layer_10 = None
        hidden_states_21 = torch._C._nn.linear(
            context_layer_11,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_11 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_22 = torch.nn.functional.dropout(
            hidden_states_21, 0.0, False, False
        )
        hidden_states_21 = None
        attention_output_3 = (
            l_self_modules_encoder_modules_layer_modules_3_parameters_lambda_1_
            * hidden_states_22
        )
        l_self_modules_encoder_modules_layer_modules_3_parameters_lambda_1_ = (
            hidden_states_22
        ) = None
        hidden_states_23 = attention_output_3 + layer_output_8
        attention_output_3 = layer_output_8 = None
        layer_output_9 = torch.nn.functional.layer_norm(
            hidden_states_23,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_bias_ = (None)
        hidden_states_24 = torch._C._nn.linear(
            layer_output_9,
            l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_9 = l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_25 = torch._C._nn.gelu(hidden_states_24)
        hidden_states_24 = None
        hidden_states_26 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_25 = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_27 = torch.nn.functional.dropout(
            hidden_states_26, 0.0, False, False
        )
        hidden_states_26 = None
        layer_output_10 = (
            l_self_modules_encoder_modules_layer_modules_3_parameters_lambda_2_
            * hidden_states_27
        )
        l_self_modules_encoder_modules_layer_modules_3_parameters_lambda_2_ = (
            hidden_states_27
        ) = None
        layer_output_11 = layer_output_10 + hidden_states_23
        layer_output_10 = hidden_states_23 = None
        old_sub_table_8 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(None, 729, None)
        ]
        reshape_8 = old_sub_table_8.reshape(1, 27, 27, -1)
        old_sub_table_8 = None
        old_sub_table_9 = reshape_8.permute(0, 3, 1, 2)
        reshape_8 = None
        new_sub_table_8 = torch.nn.functional.interpolate(
            old_sub_table_9, size=(27, 27), mode="bilinear"
        )
        old_sub_table_9 = None
        permute_21 = new_sub_table_8.permute(0, 2, 3, 1)
        new_sub_table_8 = None
        new_sub_table_9 = permute_21.reshape(729, -1)
        permute_21 = None
        getitem_41 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(729, None, None)
        ]
        new_relative_position_bias_table_4 = torch.cat([new_sub_table_9, getitem_41])
        new_sub_table_9 = getitem_41 = None
        arange_8 = torch.arange(14)
        arange_9 = torch.arange(14)
        meshgrid_4 = torch.functional.meshgrid(arange_8, arange_9, indexing="ij")
        arange_8 = arange_9 = None
        getitem_42 = meshgrid_4[0]
        getitem_43 = meshgrid_4[1]
        meshgrid_4 = None
        coords_4 = torch.stack((getitem_42, getitem_43))
        getitem_42 = getitem_43 = None
        coords_flatten_4 = torch.flatten(coords_4, 1)
        coords_4 = None
        getitem_44 = coords_flatten_4[
            (slice(None, None, None), slice(None, None, None), None)
        ]
        getitem_45 = coords_flatten_4[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        coords_flatten_4 = None
        relative_coords_8 = getitem_44 - getitem_45
        getitem_44 = getitem_45 = None
        permute_22 = relative_coords_8.permute(1, 2, 0)
        relative_coords_8 = None
        relative_coords_9 = permute_22.contiguous()
        permute_22 = None
        getitem_46 = relative_coords_9[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_46 += 13
        iadd_8 = getitem_46
        getitem_46 = None
        relative_coords_9[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = iadd_8
        setitem_28 = relative_coords_9
        iadd_8 = setitem_28 = None
        getitem_47 = relative_coords_9[
            (slice(None, None, None), slice(None, None, None), 1)
        ]
        getitem_47 += 13
        iadd_9 = getitem_47
        getitem_47 = None
        relative_coords_9[
            (slice(None, None, None), slice(None, None, None), 1)
        ] = iadd_9
        setitem_29 = relative_coords_9
        iadd_9 = setitem_29 = None
        getitem_48 = relative_coords_9[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_48 *= 27
        imul_4 = getitem_48
        getitem_48 = None
        relative_coords_9[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = imul_4
        setitem_30 = relative_coords_9
        imul_4 = setitem_30 = None
        relative_position_index_4 = torch.zeros(size=(197, 197), dtype=torch.int64)
        sum_5 = relative_coords_9.sum(-1)
        relative_coords_9 = None
        relative_position_index_4[(slice(1, None, None), slice(1, None, None))] = sum_5
        setitem_31 = relative_position_index_4
        sum_5 = setitem_31 = None
        relative_position_index_4[(0, slice(0, None, None))] = 729
        setitem_32 = relative_position_index_4
        setitem_32 = None
        relative_position_index_4[(slice(0, None, None), 0)] = 730
        setitem_33 = relative_position_index_4
        setitem_33 = None
        relative_position_index_4[(0, 0)] = 731
        setitem_34 = relative_position_index_4
        setitem_34 = None
        view_24 = relative_position_index_4.view(-1)
        relative_position_index_4 = None
        relative_position_bias_16 = new_relative_position_bias_table_4[view_24]
        new_relative_position_bias_table_4 = view_24 = None
        relative_position_bias_17 = relative_position_bias_16.view(197, 197, -1)
        relative_position_bias_16 = None
        permute_23 = relative_position_bias_17.permute(2, 0, 1)
        relative_position_bias_17 = None
        relative_position_bias_18 = permute_23.contiguous()
        permute_23 = None
        relative_position_bias_19 = relative_position_bias_18.unsqueeze(0)
        relative_position_bias_18 = None
        layer_norm_8 = torch.nn.functional.layer_norm(
            layer_output_11,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_bias_ = (None)
        linear_24 = torch._C._nn.linear(
            layer_norm_8,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_26 = linear_24.view(1, -1, 16, 64)
        linear_24 = None
        query_layer_4 = view_26.transpose(1, 2)
        view_26 = None
        linear_25 = torch._C._nn.linear(
            layer_norm_8,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_27 = linear_25.view(1, -1, 16, 64)
        linear_25 = None
        key_layer_4 = view_27.transpose(1, 2)
        view_27 = None
        linear_26 = torch._C._nn.linear(
            layer_norm_8,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        layer_norm_8 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_28 = linear_26.view(1, -1, 16, 64)
        linear_26 = None
        value_layer_4 = view_28.transpose(1, 2)
        view_28 = None
        context_layer_12 = torch._C._nn.scaled_dot_product_attention(
            query_layer_4,
            key_layer_4,
            value_layer_4,
            attn_mask=relative_position_bias_19,
            dropout_p=0.0,
            is_causal=False,
            scale=0.125,
        )
        query_layer_4 = key_layer_4 = value_layer_4 = relative_position_bias_19 = None
        permute_24 = context_layer_12.permute(0, 2, 1, 3)
        context_layer_12 = None
        context_layer_13 = permute_24.contiguous()
        permute_24 = None
        context_layer_14 = context_layer_13.view(1, 197, 1024)
        context_layer_13 = None
        hidden_states_28 = torch._C._nn.linear(
            context_layer_14,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_14 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_29 = torch.nn.functional.dropout(
            hidden_states_28, 0.0, False, False
        )
        hidden_states_28 = None
        attention_output_4 = (
            l_self_modules_encoder_modules_layer_modules_4_parameters_lambda_1_
            * hidden_states_29
        )
        l_self_modules_encoder_modules_layer_modules_4_parameters_lambda_1_ = (
            hidden_states_29
        ) = None
        hidden_states_30 = attention_output_4 + layer_output_11
        attention_output_4 = layer_output_11 = None
        layer_output_12 = torch.nn.functional.layer_norm(
            hidden_states_30,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_bias_ = (None)
        hidden_states_31 = torch._C._nn.linear(
            layer_output_12,
            l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_12 = l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_32 = torch._C._nn.gelu(hidden_states_31)
        hidden_states_31 = None
        hidden_states_33 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_32 = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_34 = torch.nn.functional.dropout(
            hidden_states_33, 0.0, False, False
        )
        hidden_states_33 = None
        layer_output_13 = (
            l_self_modules_encoder_modules_layer_modules_4_parameters_lambda_2_
            * hidden_states_34
        )
        l_self_modules_encoder_modules_layer_modules_4_parameters_lambda_2_ = (
            hidden_states_34
        ) = None
        layer_output_14 = layer_output_13 + hidden_states_30
        layer_output_13 = hidden_states_30 = None
        old_sub_table_10 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(None, 729, None)
        ]
        reshape_10 = old_sub_table_10.reshape(1, 27, 27, -1)
        old_sub_table_10 = None
        old_sub_table_11 = reshape_10.permute(0, 3, 1, 2)
        reshape_10 = None
        new_sub_table_10 = torch.nn.functional.interpolate(
            old_sub_table_11, size=(27, 27), mode="bilinear"
        )
        old_sub_table_11 = None
        permute_26 = new_sub_table_10.permute(0, 2, 3, 1)
        new_sub_table_10 = None
        new_sub_table_11 = permute_26.reshape(729, -1)
        permute_26 = None
        getitem_51 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(729, None, None)
        ]
        new_relative_position_bias_table_5 = torch.cat([new_sub_table_11, getitem_51])
        new_sub_table_11 = getitem_51 = None
        arange_10 = torch.arange(14)
        arange_11 = torch.arange(14)
        meshgrid_5 = torch.functional.meshgrid(arange_10, arange_11, indexing="ij")
        arange_10 = arange_11 = None
        getitem_52 = meshgrid_5[0]
        getitem_53 = meshgrid_5[1]
        meshgrid_5 = None
        coords_5 = torch.stack((getitem_52, getitem_53))
        getitem_52 = getitem_53 = None
        coords_flatten_5 = torch.flatten(coords_5, 1)
        coords_5 = None
        getitem_54 = coords_flatten_5[
            (slice(None, None, None), slice(None, None, None), None)
        ]
        getitem_55 = coords_flatten_5[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        coords_flatten_5 = None
        relative_coords_10 = getitem_54 - getitem_55
        getitem_54 = getitem_55 = None
        permute_27 = relative_coords_10.permute(1, 2, 0)
        relative_coords_10 = None
        relative_coords_11 = permute_27.contiguous()
        permute_27 = None
        getitem_56 = relative_coords_11[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_56 += 13
        iadd_10 = getitem_56
        getitem_56 = None
        relative_coords_11[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = iadd_10
        setitem_35 = relative_coords_11
        iadd_10 = setitem_35 = None
        getitem_57 = relative_coords_11[
            (slice(None, None, None), slice(None, None, None), 1)
        ]
        getitem_57 += 13
        iadd_11 = getitem_57
        getitem_57 = None
        relative_coords_11[
            (slice(None, None, None), slice(None, None, None), 1)
        ] = iadd_11
        setitem_36 = relative_coords_11
        iadd_11 = setitem_36 = None
        getitem_58 = relative_coords_11[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_58 *= 27
        imul_5 = getitem_58
        getitem_58 = None
        relative_coords_11[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = imul_5
        setitem_37 = relative_coords_11
        imul_5 = setitem_37 = None
        relative_position_index_5 = torch.zeros(size=(197, 197), dtype=torch.int64)
        sum_6 = relative_coords_11.sum(-1)
        relative_coords_11 = None
        relative_position_index_5[(slice(1, None, None), slice(1, None, None))] = sum_6
        setitem_38 = relative_position_index_5
        sum_6 = setitem_38 = None
        relative_position_index_5[(0, slice(0, None, None))] = 729
        setitem_39 = relative_position_index_5
        setitem_39 = None
        relative_position_index_5[(slice(0, None, None), 0)] = 730
        setitem_40 = relative_position_index_5
        setitem_40 = None
        relative_position_index_5[(0, 0)] = 731
        setitem_41 = relative_position_index_5
        setitem_41 = None
        view_30 = relative_position_index_5.view(-1)
        relative_position_index_5 = None
        relative_position_bias_20 = new_relative_position_bias_table_5[view_30]
        new_relative_position_bias_table_5 = view_30 = None
        relative_position_bias_21 = relative_position_bias_20.view(197, 197, -1)
        relative_position_bias_20 = None
        permute_28 = relative_position_bias_21.permute(2, 0, 1)
        relative_position_bias_21 = None
        relative_position_bias_22 = permute_28.contiguous()
        permute_28 = None
        relative_position_bias_23 = relative_position_bias_22.unsqueeze(0)
        relative_position_bias_22 = None
        layer_norm_10 = torch.nn.functional.layer_norm(
            layer_output_14,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_bias_ = (None)
        linear_30 = torch._C._nn.linear(
            layer_norm_10,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_32 = linear_30.view(1, -1, 16, 64)
        linear_30 = None
        query_layer_5 = view_32.transpose(1, 2)
        view_32 = None
        linear_31 = torch._C._nn.linear(
            layer_norm_10,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_33 = linear_31.view(1, -1, 16, 64)
        linear_31 = None
        key_layer_5 = view_33.transpose(1, 2)
        view_33 = None
        linear_32 = torch._C._nn.linear(
            layer_norm_10,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        layer_norm_10 = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_34 = linear_32.view(1, -1, 16, 64)
        linear_32 = None
        value_layer_5 = view_34.transpose(1, 2)
        view_34 = None
        context_layer_15 = torch._C._nn.scaled_dot_product_attention(
            query_layer_5,
            key_layer_5,
            value_layer_5,
            attn_mask=relative_position_bias_23,
            dropout_p=0.0,
            is_causal=False,
            scale=0.125,
        )
        query_layer_5 = key_layer_5 = value_layer_5 = relative_position_bias_23 = None
        permute_29 = context_layer_15.permute(0, 2, 1, 3)
        context_layer_15 = None
        context_layer_16 = permute_29.contiguous()
        permute_29 = None
        context_layer_17 = context_layer_16.view(1, 197, 1024)
        context_layer_16 = None
        hidden_states_35 = torch._C._nn.linear(
            context_layer_17,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_17 = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_36 = torch.nn.functional.dropout(
            hidden_states_35, 0.0, False, False
        )
        hidden_states_35 = None
        attention_output_5 = (
            l_self_modules_encoder_modules_layer_modules_5_parameters_lambda_1_
            * hidden_states_36
        )
        l_self_modules_encoder_modules_layer_modules_5_parameters_lambda_1_ = (
            hidden_states_36
        ) = None
        hidden_states_37 = attention_output_5 + layer_output_14
        attention_output_5 = layer_output_14 = None
        layer_output_15 = torch.nn.functional.layer_norm(
            hidden_states_37,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_bias_ = (None)
        hidden_states_38 = torch._C._nn.linear(
            layer_output_15,
            l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_15 = l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_39 = torch._C._nn.gelu(hidden_states_38)
        hidden_states_38 = None
        hidden_states_40 = torch._C._nn.linear(
            hidden_states_39,
            l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_39 = l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_41 = torch.nn.functional.dropout(
            hidden_states_40, 0.0, False, False
        )
        hidden_states_40 = None
        layer_output_16 = (
            l_self_modules_encoder_modules_layer_modules_5_parameters_lambda_2_
            * hidden_states_41
        )
        l_self_modules_encoder_modules_layer_modules_5_parameters_lambda_2_ = (
            hidden_states_41
        ) = None
        layer_output_17 = layer_output_16 + hidden_states_37
        layer_output_16 = hidden_states_37 = None
        old_sub_table_12 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(None, 729, None)
        ]
        reshape_12 = old_sub_table_12.reshape(1, 27, 27, -1)
        old_sub_table_12 = None
        old_sub_table_13 = reshape_12.permute(0, 3, 1, 2)
        reshape_12 = None
        new_sub_table_12 = torch.nn.functional.interpolate(
            old_sub_table_13, size=(27, 27), mode="bilinear"
        )
        old_sub_table_13 = None
        permute_31 = new_sub_table_12.permute(0, 2, 3, 1)
        new_sub_table_12 = None
        new_sub_table_13 = permute_31.reshape(729, -1)
        permute_31 = None
        getitem_61 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(729, None, None)
        ]
        new_relative_position_bias_table_6 = torch.cat([new_sub_table_13, getitem_61])
        new_sub_table_13 = getitem_61 = None
        arange_12 = torch.arange(14)
        arange_13 = torch.arange(14)
        meshgrid_6 = torch.functional.meshgrid(arange_12, arange_13, indexing="ij")
        arange_12 = arange_13 = None
        getitem_62 = meshgrid_6[0]
        getitem_63 = meshgrid_6[1]
        meshgrid_6 = None
        coords_6 = torch.stack((getitem_62, getitem_63))
        getitem_62 = getitem_63 = None
        coords_flatten_6 = torch.flatten(coords_6, 1)
        coords_6 = None
        getitem_64 = coords_flatten_6[
            (slice(None, None, None), slice(None, None, None), None)
        ]
        getitem_65 = coords_flatten_6[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        coords_flatten_6 = None
        relative_coords_12 = getitem_64 - getitem_65
        getitem_64 = getitem_65 = None
        permute_32 = relative_coords_12.permute(1, 2, 0)
        relative_coords_12 = None
        relative_coords_13 = permute_32.contiguous()
        permute_32 = None
        getitem_66 = relative_coords_13[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_66 += 13
        iadd_12 = getitem_66
        getitem_66 = None
        relative_coords_13[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = iadd_12
        setitem_42 = relative_coords_13
        iadd_12 = setitem_42 = None
        getitem_67 = relative_coords_13[
            (slice(None, None, None), slice(None, None, None), 1)
        ]
        getitem_67 += 13
        iadd_13 = getitem_67
        getitem_67 = None
        relative_coords_13[
            (slice(None, None, None), slice(None, None, None), 1)
        ] = iadd_13
        setitem_43 = relative_coords_13
        iadd_13 = setitem_43 = None
        getitem_68 = relative_coords_13[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_68 *= 27
        imul_6 = getitem_68
        getitem_68 = None
        relative_coords_13[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = imul_6
        setitem_44 = relative_coords_13
        imul_6 = setitem_44 = None
        relative_position_index_6 = torch.zeros(size=(197, 197), dtype=torch.int64)
        sum_7 = relative_coords_13.sum(-1)
        relative_coords_13 = None
        relative_position_index_6[(slice(1, None, None), slice(1, None, None))] = sum_7
        setitem_45 = relative_position_index_6
        sum_7 = setitem_45 = None
        relative_position_index_6[(0, slice(0, None, None))] = 729
        setitem_46 = relative_position_index_6
        setitem_46 = None
        relative_position_index_6[(slice(0, None, None), 0)] = 730
        setitem_47 = relative_position_index_6
        setitem_47 = None
        relative_position_index_6[(0, 0)] = 731
        setitem_48 = relative_position_index_6
        setitem_48 = None
        view_36 = relative_position_index_6.view(-1)
        relative_position_index_6 = None
        relative_position_bias_24 = new_relative_position_bias_table_6[view_36]
        new_relative_position_bias_table_6 = view_36 = None
        relative_position_bias_25 = relative_position_bias_24.view(197, 197, -1)
        relative_position_bias_24 = None
        permute_33 = relative_position_bias_25.permute(2, 0, 1)
        relative_position_bias_25 = None
        relative_position_bias_26 = permute_33.contiguous()
        permute_33 = None
        relative_position_bias_27 = relative_position_bias_26.unsqueeze(0)
        relative_position_bias_26 = None
        layer_norm_12 = torch.nn.functional.layer_norm(
            layer_output_17,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_bias_ = (None)
        linear_36 = torch._C._nn.linear(
            layer_norm_12,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_38 = linear_36.view(1, -1, 16, 64)
        linear_36 = None
        query_layer_6 = view_38.transpose(1, 2)
        view_38 = None
        linear_37 = torch._C._nn.linear(
            layer_norm_12,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_39 = linear_37.view(1, -1, 16, 64)
        linear_37 = None
        key_layer_6 = view_39.transpose(1, 2)
        view_39 = None
        linear_38 = torch._C._nn.linear(
            layer_norm_12,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        layer_norm_12 = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_40 = linear_38.view(1, -1, 16, 64)
        linear_38 = None
        value_layer_6 = view_40.transpose(1, 2)
        view_40 = None
        context_layer_18 = torch._C._nn.scaled_dot_product_attention(
            query_layer_6,
            key_layer_6,
            value_layer_6,
            attn_mask=relative_position_bias_27,
            dropout_p=0.0,
            is_causal=False,
            scale=0.125,
        )
        query_layer_6 = key_layer_6 = value_layer_6 = relative_position_bias_27 = None
        permute_34 = context_layer_18.permute(0, 2, 1, 3)
        context_layer_18 = None
        context_layer_19 = permute_34.contiguous()
        permute_34 = None
        context_layer_20 = context_layer_19.view(1, 197, 1024)
        context_layer_19 = None
        hidden_states_42 = torch._C._nn.linear(
            context_layer_20,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_20 = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_43 = torch.nn.functional.dropout(
            hidden_states_42, 0.0, False, False
        )
        hidden_states_42 = None
        attention_output_6 = (
            l_self_modules_encoder_modules_layer_modules_6_parameters_lambda_1_
            * hidden_states_43
        )
        l_self_modules_encoder_modules_layer_modules_6_parameters_lambda_1_ = (
            hidden_states_43
        ) = None
        hidden_states_44 = attention_output_6 + layer_output_17
        attention_output_6 = layer_output_17 = None
        layer_output_18 = torch.nn.functional.layer_norm(
            hidden_states_44,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_bias_ = (None)
        hidden_states_45 = torch._C._nn.linear(
            layer_output_18,
            l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_18 = l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_46 = torch._C._nn.gelu(hidden_states_45)
        hidden_states_45 = None
        hidden_states_47 = torch._C._nn.linear(
            hidden_states_46,
            l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_46 = l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_48 = torch.nn.functional.dropout(
            hidden_states_47, 0.0, False, False
        )
        hidden_states_47 = None
        layer_output_19 = (
            l_self_modules_encoder_modules_layer_modules_6_parameters_lambda_2_
            * hidden_states_48
        )
        l_self_modules_encoder_modules_layer_modules_6_parameters_lambda_2_ = (
            hidden_states_48
        ) = None
        layer_output_20 = layer_output_19 + hidden_states_44
        layer_output_19 = hidden_states_44 = None
        old_sub_table_14 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(None, 729, None)
        ]
        reshape_14 = old_sub_table_14.reshape(1, 27, 27, -1)
        old_sub_table_14 = None
        old_sub_table_15 = reshape_14.permute(0, 3, 1, 2)
        reshape_14 = None
        new_sub_table_14 = torch.nn.functional.interpolate(
            old_sub_table_15, size=(27, 27), mode="bilinear"
        )
        old_sub_table_15 = None
        permute_36 = new_sub_table_14.permute(0, 2, 3, 1)
        new_sub_table_14 = None
        new_sub_table_15 = permute_36.reshape(729, -1)
        permute_36 = None
        getitem_71 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(729, None, None)
        ]
        new_relative_position_bias_table_7 = torch.cat([new_sub_table_15, getitem_71])
        new_sub_table_15 = getitem_71 = None
        arange_14 = torch.arange(14)
        arange_15 = torch.arange(14)
        meshgrid_7 = torch.functional.meshgrid(arange_14, arange_15, indexing="ij")
        arange_14 = arange_15 = None
        getitem_72 = meshgrid_7[0]
        getitem_73 = meshgrid_7[1]
        meshgrid_7 = None
        coords_7 = torch.stack((getitem_72, getitem_73))
        getitem_72 = getitem_73 = None
        coords_flatten_7 = torch.flatten(coords_7, 1)
        coords_7 = None
        getitem_74 = coords_flatten_7[
            (slice(None, None, None), slice(None, None, None), None)
        ]
        getitem_75 = coords_flatten_7[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        coords_flatten_7 = None
        relative_coords_14 = getitem_74 - getitem_75
        getitem_74 = getitem_75 = None
        permute_37 = relative_coords_14.permute(1, 2, 0)
        relative_coords_14 = None
        relative_coords_15 = permute_37.contiguous()
        permute_37 = None
        getitem_76 = relative_coords_15[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_76 += 13
        iadd_14 = getitem_76
        getitem_76 = None
        relative_coords_15[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = iadd_14
        setitem_49 = relative_coords_15
        iadd_14 = setitem_49 = None
        getitem_77 = relative_coords_15[
            (slice(None, None, None), slice(None, None, None), 1)
        ]
        getitem_77 += 13
        iadd_15 = getitem_77
        getitem_77 = None
        relative_coords_15[
            (slice(None, None, None), slice(None, None, None), 1)
        ] = iadd_15
        setitem_50 = relative_coords_15
        iadd_15 = setitem_50 = None
        getitem_78 = relative_coords_15[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_78 *= 27
        imul_7 = getitem_78
        getitem_78 = None
        relative_coords_15[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = imul_7
        setitem_51 = relative_coords_15
        imul_7 = setitem_51 = None
        relative_position_index_7 = torch.zeros(size=(197, 197), dtype=torch.int64)
        sum_8 = relative_coords_15.sum(-1)
        relative_coords_15 = None
        relative_position_index_7[(slice(1, None, None), slice(1, None, None))] = sum_8
        setitem_52 = relative_position_index_7
        sum_8 = setitem_52 = None
        relative_position_index_7[(0, slice(0, None, None))] = 729
        setitem_53 = relative_position_index_7
        setitem_53 = None
        relative_position_index_7[(slice(0, None, None), 0)] = 730
        setitem_54 = relative_position_index_7
        setitem_54 = None
        relative_position_index_7[(0, 0)] = 731
        setitem_55 = relative_position_index_7
        setitem_55 = None
        view_42 = relative_position_index_7.view(-1)
        relative_position_index_7 = None
        relative_position_bias_28 = new_relative_position_bias_table_7[view_42]
        new_relative_position_bias_table_7 = view_42 = None
        relative_position_bias_29 = relative_position_bias_28.view(197, 197, -1)
        relative_position_bias_28 = None
        permute_38 = relative_position_bias_29.permute(2, 0, 1)
        relative_position_bias_29 = None
        relative_position_bias_30 = permute_38.contiguous()
        permute_38 = None
        relative_position_bias_31 = relative_position_bias_30.unsqueeze(0)
        relative_position_bias_30 = None
        layer_norm_14 = torch.nn.functional.layer_norm(
            layer_output_20,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_bias_ = (None)
        linear_42 = torch._C._nn.linear(
            layer_norm_14,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_44 = linear_42.view(1, -1, 16, 64)
        linear_42 = None
        query_layer_7 = view_44.transpose(1, 2)
        view_44 = None
        linear_43 = torch._C._nn.linear(
            layer_norm_14,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_45 = linear_43.view(1, -1, 16, 64)
        linear_43 = None
        key_layer_7 = view_45.transpose(1, 2)
        view_45 = None
        linear_44 = torch._C._nn.linear(
            layer_norm_14,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        layer_norm_14 = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_46 = linear_44.view(1, -1, 16, 64)
        linear_44 = None
        value_layer_7 = view_46.transpose(1, 2)
        view_46 = None
        context_layer_21 = torch._C._nn.scaled_dot_product_attention(
            query_layer_7,
            key_layer_7,
            value_layer_7,
            attn_mask=relative_position_bias_31,
            dropout_p=0.0,
            is_causal=False,
            scale=0.125,
        )
        query_layer_7 = key_layer_7 = value_layer_7 = relative_position_bias_31 = None
        permute_39 = context_layer_21.permute(0, 2, 1, 3)
        context_layer_21 = None
        context_layer_22 = permute_39.contiguous()
        permute_39 = None
        context_layer_23 = context_layer_22.view(1, 197, 1024)
        context_layer_22 = None
        hidden_states_49 = torch._C._nn.linear(
            context_layer_23,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_23 = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_50 = torch.nn.functional.dropout(
            hidden_states_49, 0.0, False, False
        )
        hidden_states_49 = None
        attention_output_7 = (
            l_self_modules_encoder_modules_layer_modules_7_parameters_lambda_1_
            * hidden_states_50
        )
        l_self_modules_encoder_modules_layer_modules_7_parameters_lambda_1_ = (
            hidden_states_50
        ) = None
        hidden_states_51 = attention_output_7 + layer_output_20
        attention_output_7 = layer_output_20 = None
        layer_output_21 = torch.nn.functional.layer_norm(
            hidden_states_51,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_bias_ = (None)
        hidden_states_52 = torch._C._nn.linear(
            layer_output_21,
            l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_21 = l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_53 = torch._C._nn.gelu(hidden_states_52)
        hidden_states_52 = None
        hidden_states_54 = torch._C._nn.linear(
            hidden_states_53,
            l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_53 = l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_55 = torch.nn.functional.dropout(
            hidden_states_54, 0.0, False, False
        )
        hidden_states_54 = None
        layer_output_22 = (
            l_self_modules_encoder_modules_layer_modules_7_parameters_lambda_2_
            * hidden_states_55
        )
        l_self_modules_encoder_modules_layer_modules_7_parameters_lambda_2_ = (
            hidden_states_55
        ) = None
        layer_output_23 = layer_output_22 + hidden_states_51
        layer_output_22 = hidden_states_51 = None
        old_sub_table_16 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(None, 729, None)
        ]
        reshape_16 = old_sub_table_16.reshape(1, 27, 27, -1)
        old_sub_table_16 = None
        old_sub_table_17 = reshape_16.permute(0, 3, 1, 2)
        reshape_16 = None
        new_sub_table_16 = torch.nn.functional.interpolate(
            old_sub_table_17, size=(27, 27), mode="bilinear"
        )
        old_sub_table_17 = None
        permute_41 = new_sub_table_16.permute(0, 2, 3, 1)
        new_sub_table_16 = None
        new_sub_table_17 = permute_41.reshape(729, -1)
        permute_41 = None
        getitem_81 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(729, None, None)
        ]
        new_relative_position_bias_table_8 = torch.cat([new_sub_table_17, getitem_81])
        new_sub_table_17 = getitem_81 = None
        arange_16 = torch.arange(14)
        arange_17 = torch.arange(14)
        meshgrid_8 = torch.functional.meshgrid(arange_16, arange_17, indexing="ij")
        arange_16 = arange_17 = None
        getitem_82 = meshgrid_8[0]
        getitem_83 = meshgrid_8[1]
        meshgrid_8 = None
        coords_8 = torch.stack((getitem_82, getitem_83))
        getitem_82 = getitem_83 = None
        coords_flatten_8 = torch.flatten(coords_8, 1)
        coords_8 = None
        getitem_84 = coords_flatten_8[
            (slice(None, None, None), slice(None, None, None), None)
        ]
        getitem_85 = coords_flatten_8[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        coords_flatten_8 = None
        relative_coords_16 = getitem_84 - getitem_85
        getitem_84 = getitem_85 = None
        permute_42 = relative_coords_16.permute(1, 2, 0)
        relative_coords_16 = None
        relative_coords_17 = permute_42.contiguous()
        permute_42 = None
        getitem_86 = relative_coords_17[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_86 += 13
        iadd_16 = getitem_86
        getitem_86 = None
        relative_coords_17[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = iadd_16
        setitem_56 = relative_coords_17
        iadd_16 = setitem_56 = None
        getitem_87 = relative_coords_17[
            (slice(None, None, None), slice(None, None, None), 1)
        ]
        getitem_87 += 13
        iadd_17 = getitem_87
        getitem_87 = None
        relative_coords_17[
            (slice(None, None, None), slice(None, None, None), 1)
        ] = iadd_17
        setitem_57 = relative_coords_17
        iadd_17 = setitem_57 = None
        getitem_88 = relative_coords_17[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_88 *= 27
        imul_8 = getitem_88
        getitem_88 = None
        relative_coords_17[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = imul_8
        setitem_58 = relative_coords_17
        imul_8 = setitem_58 = None
        relative_position_index_8 = torch.zeros(size=(197, 197), dtype=torch.int64)
        sum_9 = relative_coords_17.sum(-1)
        relative_coords_17 = None
        relative_position_index_8[(slice(1, None, None), slice(1, None, None))] = sum_9
        setitem_59 = relative_position_index_8
        sum_9 = setitem_59 = None
        relative_position_index_8[(0, slice(0, None, None))] = 729
        setitem_60 = relative_position_index_8
        setitem_60 = None
        relative_position_index_8[(slice(0, None, None), 0)] = 730
        setitem_61 = relative_position_index_8
        setitem_61 = None
        relative_position_index_8[(0, 0)] = 731
        setitem_62 = relative_position_index_8
        setitem_62 = None
        view_48 = relative_position_index_8.view(-1)
        relative_position_index_8 = None
        relative_position_bias_32 = new_relative_position_bias_table_8[view_48]
        new_relative_position_bias_table_8 = view_48 = None
        relative_position_bias_33 = relative_position_bias_32.view(197, 197, -1)
        relative_position_bias_32 = None
        permute_43 = relative_position_bias_33.permute(2, 0, 1)
        relative_position_bias_33 = None
        relative_position_bias_34 = permute_43.contiguous()
        permute_43 = None
        relative_position_bias_35 = relative_position_bias_34.unsqueeze(0)
        relative_position_bias_34 = None
        layer_norm_16 = torch.nn.functional.layer_norm(
            layer_output_23,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_bias_ = (None)
        linear_48 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_50 = linear_48.view(1, -1, 16, 64)
        linear_48 = None
        query_layer_8 = view_50.transpose(1, 2)
        view_50 = None
        linear_49 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_51 = linear_49.view(1, -1, 16, 64)
        linear_49 = None
        key_layer_8 = view_51.transpose(1, 2)
        view_51 = None
        linear_50 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        layer_norm_16 = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_52 = linear_50.view(1, -1, 16, 64)
        linear_50 = None
        value_layer_8 = view_52.transpose(1, 2)
        view_52 = None
        context_layer_24 = torch._C._nn.scaled_dot_product_attention(
            query_layer_8,
            key_layer_8,
            value_layer_8,
            attn_mask=relative_position_bias_35,
            dropout_p=0.0,
            is_causal=False,
            scale=0.125,
        )
        query_layer_8 = key_layer_8 = value_layer_8 = relative_position_bias_35 = None
        permute_44 = context_layer_24.permute(0, 2, 1, 3)
        context_layer_24 = None
        context_layer_25 = permute_44.contiguous()
        permute_44 = None
        context_layer_26 = context_layer_25.view(1, 197, 1024)
        context_layer_25 = None
        hidden_states_56 = torch._C._nn.linear(
            context_layer_26,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_26 = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_57 = torch.nn.functional.dropout(
            hidden_states_56, 0.0, False, False
        )
        hidden_states_56 = None
        attention_output_8 = (
            l_self_modules_encoder_modules_layer_modules_8_parameters_lambda_1_
            * hidden_states_57
        )
        l_self_modules_encoder_modules_layer_modules_8_parameters_lambda_1_ = (
            hidden_states_57
        ) = None
        hidden_states_58 = attention_output_8 + layer_output_23
        attention_output_8 = layer_output_23 = None
        layer_output_24 = torch.nn.functional.layer_norm(
            hidden_states_58,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_bias_ = (None)
        hidden_states_59 = torch._C._nn.linear(
            layer_output_24,
            l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_24 = l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_60 = torch._C._nn.gelu(hidden_states_59)
        hidden_states_59 = None
        hidden_states_61 = torch._C._nn.linear(
            hidden_states_60,
            l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_60 = l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_62 = torch.nn.functional.dropout(
            hidden_states_61, 0.0, False, False
        )
        hidden_states_61 = None
        layer_output_25 = (
            l_self_modules_encoder_modules_layer_modules_8_parameters_lambda_2_
            * hidden_states_62
        )
        l_self_modules_encoder_modules_layer_modules_8_parameters_lambda_2_ = (
            hidden_states_62
        ) = None
        layer_output_26 = layer_output_25 + hidden_states_58
        layer_output_25 = hidden_states_58 = None
        old_sub_table_18 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(None, 729, None)
        ]
        reshape_18 = old_sub_table_18.reshape(1, 27, 27, -1)
        old_sub_table_18 = None
        old_sub_table_19 = reshape_18.permute(0, 3, 1, 2)
        reshape_18 = None
        new_sub_table_18 = torch.nn.functional.interpolate(
            old_sub_table_19, size=(27, 27), mode="bilinear"
        )
        old_sub_table_19 = None
        permute_46 = new_sub_table_18.permute(0, 2, 3, 1)
        new_sub_table_18 = None
        new_sub_table_19 = permute_46.reshape(729, -1)
        permute_46 = None
        getitem_91 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(729, None, None)
        ]
        new_relative_position_bias_table_9 = torch.cat([new_sub_table_19, getitem_91])
        new_sub_table_19 = getitem_91 = None
        arange_18 = torch.arange(14)
        arange_19 = torch.arange(14)
        meshgrid_9 = torch.functional.meshgrid(arange_18, arange_19, indexing="ij")
        arange_18 = arange_19 = None
        getitem_92 = meshgrid_9[0]
        getitem_93 = meshgrid_9[1]
        meshgrid_9 = None
        coords_9 = torch.stack((getitem_92, getitem_93))
        getitem_92 = getitem_93 = None
        coords_flatten_9 = torch.flatten(coords_9, 1)
        coords_9 = None
        getitem_94 = coords_flatten_9[
            (slice(None, None, None), slice(None, None, None), None)
        ]
        getitem_95 = coords_flatten_9[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        coords_flatten_9 = None
        relative_coords_18 = getitem_94 - getitem_95
        getitem_94 = getitem_95 = None
        permute_47 = relative_coords_18.permute(1, 2, 0)
        relative_coords_18 = None
        relative_coords_19 = permute_47.contiguous()
        permute_47 = None
        getitem_96 = relative_coords_19[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_96 += 13
        iadd_18 = getitem_96
        getitem_96 = None
        relative_coords_19[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = iadd_18
        setitem_63 = relative_coords_19
        iadd_18 = setitem_63 = None
        getitem_97 = relative_coords_19[
            (slice(None, None, None), slice(None, None, None), 1)
        ]
        getitem_97 += 13
        iadd_19 = getitem_97
        getitem_97 = None
        relative_coords_19[
            (slice(None, None, None), slice(None, None, None), 1)
        ] = iadd_19
        setitem_64 = relative_coords_19
        iadd_19 = setitem_64 = None
        getitem_98 = relative_coords_19[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_98 *= 27
        imul_9 = getitem_98
        getitem_98 = None
        relative_coords_19[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = imul_9
        setitem_65 = relative_coords_19
        imul_9 = setitem_65 = None
        relative_position_index_9 = torch.zeros(size=(197, 197), dtype=torch.int64)
        sum_10 = relative_coords_19.sum(-1)
        relative_coords_19 = None
        relative_position_index_9[(slice(1, None, None), slice(1, None, None))] = sum_10
        setitem_66 = relative_position_index_9
        sum_10 = setitem_66 = None
        relative_position_index_9[(0, slice(0, None, None))] = 729
        setitem_67 = relative_position_index_9
        setitem_67 = None
        relative_position_index_9[(slice(0, None, None), 0)] = 730
        setitem_68 = relative_position_index_9
        setitem_68 = None
        relative_position_index_9[(0, 0)] = 731
        setitem_69 = relative_position_index_9
        setitem_69 = None
        view_54 = relative_position_index_9.view(-1)
        relative_position_index_9 = None
        relative_position_bias_36 = new_relative_position_bias_table_9[view_54]
        new_relative_position_bias_table_9 = view_54 = None
        relative_position_bias_37 = relative_position_bias_36.view(197, 197, -1)
        relative_position_bias_36 = None
        permute_48 = relative_position_bias_37.permute(2, 0, 1)
        relative_position_bias_37 = None
        relative_position_bias_38 = permute_48.contiguous()
        permute_48 = None
        relative_position_bias_39 = relative_position_bias_38.unsqueeze(0)
        relative_position_bias_38 = None
        layer_norm_18 = torch.nn.functional.layer_norm(
            layer_output_26,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_bias_ = (None)
        linear_54 = torch._C._nn.linear(
            layer_norm_18,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_56 = linear_54.view(1, -1, 16, 64)
        linear_54 = None
        query_layer_9 = view_56.transpose(1, 2)
        view_56 = None
        linear_55 = torch._C._nn.linear(
            layer_norm_18,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_57 = linear_55.view(1, -1, 16, 64)
        linear_55 = None
        key_layer_9 = view_57.transpose(1, 2)
        view_57 = None
        linear_56 = torch._C._nn.linear(
            layer_norm_18,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        layer_norm_18 = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_58 = linear_56.view(1, -1, 16, 64)
        linear_56 = None
        value_layer_9 = view_58.transpose(1, 2)
        view_58 = None
        context_layer_27 = torch._C._nn.scaled_dot_product_attention(
            query_layer_9,
            key_layer_9,
            value_layer_9,
            attn_mask=relative_position_bias_39,
            dropout_p=0.0,
            is_causal=False,
            scale=0.125,
        )
        query_layer_9 = key_layer_9 = value_layer_9 = relative_position_bias_39 = None
        permute_49 = context_layer_27.permute(0, 2, 1, 3)
        context_layer_27 = None
        context_layer_28 = permute_49.contiguous()
        permute_49 = None
        context_layer_29 = context_layer_28.view(1, 197, 1024)
        context_layer_28 = None
        hidden_states_63 = torch._C._nn.linear(
            context_layer_29,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_29 = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_64 = torch.nn.functional.dropout(
            hidden_states_63, 0.0, False, False
        )
        hidden_states_63 = None
        attention_output_9 = (
            l_self_modules_encoder_modules_layer_modules_9_parameters_lambda_1_
            * hidden_states_64
        )
        l_self_modules_encoder_modules_layer_modules_9_parameters_lambda_1_ = (
            hidden_states_64
        ) = None
        hidden_states_65 = attention_output_9 + layer_output_26
        attention_output_9 = layer_output_26 = None
        layer_output_27 = torch.nn.functional.layer_norm(
            hidden_states_65,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_bias_ = (None)
        hidden_states_66 = torch._C._nn.linear(
            layer_output_27,
            l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_27 = l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_67 = torch._C._nn.gelu(hidden_states_66)
        hidden_states_66 = None
        hidden_states_68 = torch._C._nn.linear(
            hidden_states_67,
            l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_67 = l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_69 = torch.nn.functional.dropout(
            hidden_states_68, 0.0, False, False
        )
        hidden_states_68 = None
        layer_output_28 = (
            l_self_modules_encoder_modules_layer_modules_9_parameters_lambda_2_
            * hidden_states_69
        )
        l_self_modules_encoder_modules_layer_modules_9_parameters_lambda_2_ = (
            hidden_states_69
        ) = None
        layer_output_29 = layer_output_28 + hidden_states_65
        layer_output_28 = hidden_states_65 = None
        old_sub_table_20 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(None, 729, None)
        ]
        reshape_20 = old_sub_table_20.reshape(1, 27, 27, -1)
        old_sub_table_20 = None
        old_sub_table_21 = reshape_20.permute(0, 3, 1, 2)
        reshape_20 = None
        new_sub_table_20 = torch.nn.functional.interpolate(
            old_sub_table_21, size=(27, 27), mode="bilinear"
        )
        old_sub_table_21 = None
        permute_51 = new_sub_table_20.permute(0, 2, 3, 1)
        new_sub_table_20 = None
        new_sub_table_21 = permute_51.reshape(729, -1)
        permute_51 = None
        getitem_101 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(729, None, None)
        ]
        new_relative_position_bias_table_10 = torch.cat([new_sub_table_21, getitem_101])
        new_sub_table_21 = getitem_101 = None
        arange_20 = torch.arange(14)
        arange_21 = torch.arange(14)
        meshgrid_10 = torch.functional.meshgrid(arange_20, arange_21, indexing="ij")
        arange_20 = arange_21 = None
        getitem_102 = meshgrid_10[0]
        getitem_103 = meshgrid_10[1]
        meshgrid_10 = None
        coords_10 = torch.stack((getitem_102, getitem_103))
        getitem_102 = getitem_103 = None
        coords_flatten_10 = torch.flatten(coords_10, 1)
        coords_10 = None
        getitem_104 = coords_flatten_10[
            (slice(None, None, None), slice(None, None, None), None)
        ]
        getitem_105 = coords_flatten_10[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        coords_flatten_10 = None
        relative_coords_20 = getitem_104 - getitem_105
        getitem_104 = getitem_105 = None
        permute_52 = relative_coords_20.permute(1, 2, 0)
        relative_coords_20 = None
        relative_coords_21 = permute_52.contiguous()
        permute_52 = None
        getitem_106 = relative_coords_21[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_106 += 13
        iadd_20 = getitem_106
        getitem_106 = None
        relative_coords_21[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = iadd_20
        setitem_70 = relative_coords_21
        iadd_20 = setitem_70 = None
        getitem_107 = relative_coords_21[
            (slice(None, None, None), slice(None, None, None), 1)
        ]
        getitem_107 += 13
        iadd_21 = getitem_107
        getitem_107 = None
        relative_coords_21[
            (slice(None, None, None), slice(None, None, None), 1)
        ] = iadd_21
        setitem_71 = relative_coords_21
        iadd_21 = setitem_71 = None
        getitem_108 = relative_coords_21[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_108 *= 27
        imul_10 = getitem_108
        getitem_108 = None
        relative_coords_21[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = imul_10
        setitem_72 = relative_coords_21
        imul_10 = setitem_72 = None
        relative_position_index_10 = torch.zeros(size=(197, 197), dtype=torch.int64)
        sum_11 = relative_coords_21.sum(-1)
        relative_coords_21 = None
        relative_position_index_10[
            (slice(1, None, None), slice(1, None, None))
        ] = sum_11
        setitem_73 = relative_position_index_10
        sum_11 = setitem_73 = None
        relative_position_index_10[(0, slice(0, None, None))] = 729
        setitem_74 = relative_position_index_10
        setitem_74 = None
        relative_position_index_10[(slice(0, None, None), 0)] = 730
        setitem_75 = relative_position_index_10
        setitem_75 = None
        relative_position_index_10[(0, 0)] = 731
        setitem_76 = relative_position_index_10
        setitem_76 = None
        view_60 = relative_position_index_10.view(-1)
        relative_position_index_10 = None
        relative_position_bias_40 = new_relative_position_bias_table_10[view_60]
        new_relative_position_bias_table_10 = view_60 = None
        relative_position_bias_41 = relative_position_bias_40.view(197, 197, -1)
        relative_position_bias_40 = None
        permute_53 = relative_position_bias_41.permute(2, 0, 1)
        relative_position_bias_41 = None
        relative_position_bias_42 = permute_53.contiguous()
        permute_53 = None
        relative_position_bias_43 = relative_position_bias_42.unsqueeze(0)
        relative_position_bias_42 = None
        layer_norm_20 = torch.nn.functional.layer_norm(
            layer_output_29,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_bias_ = (None)
        linear_60 = torch._C._nn.linear(
            layer_norm_20,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_62 = linear_60.view(1, -1, 16, 64)
        linear_60 = None
        query_layer_10 = view_62.transpose(1, 2)
        view_62 = None
        linear_61 = torch._C._nn.linear(
            layer_norm_20,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_63 = linear_61.view(1, -1, 16, 64)
        linear_61 = None
        key_layer_10 = view_63.transpose(1, 2)
        view_63 = None
        linear_62 = torch._C._nn.linear(
            layer_norm_20,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        layer_norm_20 = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_64 = linear_62.view(1, -1, 16, 64)
        linear_62 = None
        value_layer_10 = view_64.transpose(1, 2)
        view_64 = None
        context_layer_30 = torch._C._nn.scaled_dot_product_attention(
            query_layer_10,
            key_layer_10,
            value_layer_10,
            attn_mask=relative_position_bias_43,
            dropout_p=0.0,
            is_causal=False,
            scale=0.125,
        )
        query_layer_10 = (
            key_layer_10
        ) = value_layer_10 = relative_position_bias_43 = None
        permute_54 = context_layer_30.permute(0, 2, 1, 3)
        context_layer_30 = None
        context_layer_31 = permute_54.contiguous()
        permute_54 = None
        context_layer_32 = context_layer_31.view(1, 197, 1024)
        context_layer_31 = None
        hidden_states_70 = torch._C._nn.linear(
            context_layer_32,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_32 = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_71 = torch.nn.functional.dropout(
            hidden_states_70, 0.0, False, False
        )
        hidden_states_70 = None
        attention_output_10 = (
            l_self_modules_encoder_modules_layer_modules_10_parameters_lambda_1_
            * hidden_states_71
        )
        l_self_modules_encoder_modules_layer_modules_10_parameters_lambda_1_ = (
            hidden_states_71
        ) = None
        hidden_states_72 = attention_output_10 + layer_output_29
        attention_output_10 = layer_output_29 = None
        layer_output_30 = torch.nn.functional.layer_norm(
            hidden_states_72,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_bias_ = (None)
        hidden_states_73 = torch._C._nn.linear(
            layer_output_30,
            l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_30 = l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_74 = torch._C._nn.gelu(hidden_states_73)
        hidden_states_73 = None
        hidden_states_75 = torch._C._nn.linear(
            hidden_states_74,
            l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_74 = l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_76 = torch.nn.functional.dropout(
            hidden_states_75, 0.0, False, False
        )
        hidden_states_75 = None
        layer_output_31 = (
            l_self_modules_encoder_modules_layer_modules_10_parameters_lambda_2_
            * hidden_states_76
        )
        l_self_modules_encoder_modules_layer_modules_10_parameters_lambda_2_ = (
            hidden_states_76
        ) = None
        layer_output_32 = layer_output_31 + hidden_states_72
        layer_output_31 = hidden_states_72 = None
        old_sub_table_22 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(None, 729, None)
        ]
        reshape_22 = old_sub_table_22.reshape(1, 27, 27, -1)
        old_sub_table_22 = None
        old_sub_table_23 = reshape_22.permute(0, 3, 1, 2)
        reshape_22 = None
        new_sub_table_22 = torch.nn.functional.interpolate(
            old_sub_table_23, size=(27, 27), mode="bilinear"
        )
        old_sub_table_23 = None
        permute_56 = new_sub_table_22.permute(0, 2, 3, 1)
        new_sub_table_22 = None
        new_sub_table_23 = permute_56.reshape(729, -1)
        permute_56 = None
        getitem_111 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(729, None, None)
        ]
        new_relative_position_bias_table_11 = torch.cat([new_sub_table_23, getitem_111])
        new_sub_table_23 = getitem_111 = None
        arange_22 = torch.arange(14)
        arange_23 = torch.arange(14)
        meshgrid_11 = torch.functional.meshgrid(arange_22, arange_23, indexing="ij")
        arange_22 = arange_23 = None
        getitem_112 = meshgrid_11[0]
        getitem_113 = meshgrid_11[1]
        meshgrid_11 = None
        coords_11 = torch.stack((getitem_112, getitem_113))
        getitem_112 = getitem_113 = None
        coords_flatten_11 = torch.flatten(coords_11, 1)
        coords_11 = None
        getitem_114 = coords_flatten_11[
            (slice(None, None, None), slice(None, None, None), None)
        ]
        getitem_115 = coords_flatten_11[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        coords_flatten_11 = None
        relative_coords_22 = getitem_114 - getitem_115
        getitem_114 = getitem_115 = None
        permute_57 = relative_coords_22.permute(1, 2, 0)
        relative_coords_22 = None
        relative_coords_23 = permute_57.contiguous()
        permute_57 = None
        getitem_116 = relative_coords_23[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_116 += 13
        iadd_22 = getitem_116
        getitem_116 = None
        relative_coords_23[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = iadd_22
        setitem_77 = relative_coords_23
        iadd_22 = setitem_77 = None
        getitem_117 = relative_coords_23[
            (slice(None, None, None), slice(None, None, None), 1)
        ]
        getitem_117 += 13
        iadd_23 = getitem_117
        getitem_117 = None
        relative_coords_23[
            (slice(None, None, None), slice(None, None, None), 1)
        ] = iadd_23
        setitem_78 = relative_coords_23
        iadd_23 = setitem_78 = None
        getitem_118 = relative_coords_23[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_118 *= 27
        imul_11 = getitem_118
        getitem_118 = None
        relative_coords_23[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = imul_11
        setitem_79 = relative_coords_23
        imul_11 = setitem_79 = None
        relative_position_index_11 = torch.zeros(size=(197, 197), dtype=torch.int64)
        sum_12 = relative_coords_23.sum(-1)
        relative_coords_23 = None
        relative_position_index_11[
            (slice(1, None, None), slice(1, None, None))
        ] = sum_12
        setitem_80 = relative_position_index_11
        sum_12 = setitem_80 = None
        relative_position_index_11[(0, slice(0, None, None))] = 729
        setitem_81 = relative_position_index_11
        setitem_81 = None
        relative_position_index_11[(slice(0, None, None), 0)] = 730
        setitem_82 = relative_position_index_11
        setitem_82 = None
        relative_position_index_11[(0, 0)] = 731
        setitem_83 = relative_position_index_11
        setitem_83 = None
        view_66 = relative_position_index_11.view(-1)
        relative_position_index_11 = None
        relative_position_bias_44 = new_relative_position_bias_table_11[view_66]
        new_relative_position_bias_table_11 = view_66 = None
        relative_position_bias_45 = relative_position_bias_44.view(197, 197, -1)
        relative_position_bias_44 = None
        permute_58 = relative_position_bias_45.permute(2, 0, 1)
        relative_position_bias_45 = None
        relative_position_bias_46 = permute_58.contiguous()
        permute_58 = None
        relative_position_bias_47 = relative_position_bias_46.unsqueeze(0)
        relative_position_bias_46 = None
        layer_norm_22 = torch.nn.functional.layer_norm(
            layer_output_32,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_bias_ = (None)
        linear_66 = torch._C._nn.linear(
            layer_norm_22,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_68 = linear_66.view(1, -1, 16, 64)
        linear_66 = None
        query_layer_11 = view_68.transpose(1, 2)
        view_68 = None
        linear_67 = torch._C._nn.linear(
            layer_norm_22,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_69 = linear_67.view(1, -1, 16, 64)
        linear_67 = None
        key_layer_11 = view_69.transpose(1, 2)
        view_69 = None
        linear_68 = torch._C._nn.linear(
            layer_norm_22,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        layer_norm_22 = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_70 = linear_68.view(1, -1, 16, 64)
        linear_68 = None
        value_layer_11 = view_70.transpose(1, 2)
        view_70 = None
        context_layer_33 = torch._C._nn.scaled_dot_product_attention(
            query_layer_11,
            key_layer_11,
            value_layer_11,
            attn_mask=relative_position_bias_47,
            dropout_p=0.0,
            is_causal=False,
            scale=0.125,
        )
        query_layer_11 = (
            key_layer_11
        ) = value_layer_11 = relative_position_bias_47 = None
        permute_59 = context_layer_33.permute(0, 2, 1, 3)
        context_layer_33 = None
        context_layer_34 = permute_59.contiguous()
        permute_59 = None
        context_layer_35 = context_layer_34.view(1, 197, 1024)
        context_layer_34 = None
        hidden_states_77 = torch._C._nn.linear(
            context_layer_35,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_35 = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_78 = torch.nn.functional.dropout(
            hidden_states_77, 0.0, False, False
        )
        hidden_states_77 = None
        attention_output_11 = (
            l_self_modules_encoder_modules_layer_modules_11_parameters_lambda_1_
            * hidden_states_78
        )
        l_self_modules_encoder_modules_layer_modules_11_parameters_lambda_1_ = (
            hidden_states_78
        ) = None
        hidden_states_79 = attention_output_11 + layer_output_32
        attention_output_11 = layer_output_32 = None
        layer_output_33 = torch.nn.functional.layer_norm(
            hidden_states_79,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_bias_ = (None)
        hidden_states_80 = torch._C._nn.linear(
            layer_output_33,
            l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_33 = l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_81 = torch._C._nn.gelu(hidden_states_80)
        hidden_states_80 = None
        hidden_states_82 = torch._C._nn.linear(
            hidden_states_81,
            l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_81 = l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_83 = torch.nn.functional.dropout(
            hidden_states_82, 0.0, False, False
        )
        hidden_states_82 = None
        layer_output_34 = (
            l_self_modules_encoder_modules_layer_modules_11_parameters_lambda_2_
            * hidden_states_83
        )
        l_self_modules_encoder_modules_layer_modules_11_parameters_lambda_2_ = (
            hidden_states_83
        ) = None
        layer_output_35 = layer_output_34 + hidden_states_79
        layer_output_34 = hidden_states_79 = None
        old_sub_table_24 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(None, 729, None)
        ]
        reshape_24 = old_sub_table_24.reshape(1, 27, 27, -1)
        old_sub_table_24 = None
        old_sub_table_25 = reshape_24.permute(0, 3, 1, 2)
        reshape_24 = None
        new_sub_table_24 = torch.nn.functional.interpolate(
            old_sub_table_25, size=(27, 27), mode="bilinear"
        )
        old_sub_table_25 = None
        permute_61 = new_sub_table_24.permute(0, 2, 3, 1)
        new_sub_table_24 = None
        new_sub_table_25 = permute_61.reshape(729, -1)
        permute_61 = None
        getitem_121 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(729, None, None)
        ]
        new_relative_position_bias_table_12 = torch.cat([new_sub_table_25, getitem_121])
        new_sub_table_25 = getitem_121 = None
        arange_24 = torch.arange(14)
        arange_25 = torch.arange(14)
        meshgrid_12 = torch.functional.meshgrid(arange_24, arange_25, indexing="ij")
        arange_24 = arange_25 = None
        getitem_122 = meshgrid_12[0]
        getitem_123 = meshgrid_12[1]
        meshgrid_12 = None
        coords_12 = torch.stack((getitem_122, getitem_123))
        getitem_122 = getitem_123 = None
        coords_flatten_12 = torch.flatten(coords_12, 1)
        coords_12 = None
        getitem_124 = coords_flatten_12[
            (slice(None, None, None), slice(None, None, None), None)
        ]
        getitem_125 = coords_flatten_12[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        coords_flatten_12 = None
        relative_coords_24 = getitem_124 - getitem_125
        getitem_124 = getitem_125 = None
        permute_62 = relative_coords_24.permute(1, 2, 0)
        relative_coords_24 = None
        relative_coords_25 = permute_62.contiguous()
        permute_62 = None
        getitem_126 = relative_coords_25[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_126 += 13
        iadd_24 = getitem_126
        getitem_126 = None
        relative_coords_25[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = iadd_24
        setitem_84 = relative_coords_25
        iadd_24 = setitem_84 = None
        getitem_127 = relative_coords_25[
            (slice(None, None, None), slice(None, None, None), 1)
        ]
        getitem_127 += 13
        iadd_25 = getitem_127
        getitem_127 = None
        relative_coords_25[
            (slice(None, None, None), slice(None, None, None), 1)
        ] = iadd_25
        setitem_85 = relative_coords_25
        iadd_25 = setitem_85 = None
        getitem_128 = relative_coords_25[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_128 *= 27
        imul_12 = getitem_128
        getitem_128 = None
        relative_coords_25[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = imul_12
        setitem_86 = relative_coords_25
        imul_12 = setitem_86 = None
        relative_position_index_12 = torch.zeros(size=(197, 197), dtype=torch.int64)
        sum_13 = relative_coords_25.sum(-1)
        relative_coords_25 = None
        relative_position_index_12[
            (slice(1, None, None), slice(1, None, None))
        ] = sum_13
        setitem_87 = relative_position_index_12
        sum_13 = setitem_87 = None
        relative_position_index_12[(0, slice(0, None, None))] = 729
        setitem_88 = relative_position_index_12
        setitem_88 = None
        relative_position_index_12[(slice(0, None, None), 0)] = 730
        setitem_89 = relative_position_index_12
        setitem_89 = None
        relative_position_index_12[(0, 0)] = 731
        setitem_90 = relative_position_index_12
        setitem_90 = None
        view_72 = relative_position_index_12.view(-1)
        relative_position_index_12 = None
        relative_position_bias_48 = new_relative_position_bias_table_12[view_72]
        new_relative_position_bias_table_12 = view_72 = None
        relative_position_bias_49 = relative_position_bias_48.view(197, 197, -1)
        relative_position_bias_48 = None
        permute_63 = relative_position_bias_49.permute(2, 0, 1)
        relative_position_bias_49 = None
        relative_position_bias_50 = permute_63.contiguous()
        permute_63 = None
        relative_position_bias_51 = relative_position_bias_50.unsqueeze(0)
        relative_position_bias_50 = None
        layer_norm_24 = torch.nn.functional.layer_norm(
            layer_output_35,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_before_parameters_bias_ = (None)
        linear_72 = torch._C._nn.linear(
            layer_norm_24,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_74 = linear_72.view(1, -1, 16, 64)
        linear_72 = None
        query_layer_12 = view_74.transpose(1, 2)
        view_74 = None
        linear_73 = torch._C._nn.linear(
            layer_norm_24,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_75 = linear_73.view(1, -1, 16, 64)
        linear_73 = None
        key_layer_12 = view_75.transpose(1, 2)
        view_75 = None
        linear_74 = torch._C._nn.linear(
            layer_norm_24,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        layer_norm_24 = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_76 = linear_74.view(1, -1, 16, 64)
        linear_74 = None
        value_layer_12 = view_76.transpose(1, 2)
        view_76 = None
        context_layer_36 = torch._C._nn.scaled_dot_product_attention(
            query_layer_12,
            key_layer_12,
            value_layer_12,
            attn_mask=relative_position_bias_51,
            dropout_p=0.0,
            is_causal=False,
            scale=0.125,
        )
        query_layer_12 = (
            key_layer_12
        ) = value_layer_12 = relative_position_bias_51 = None
        permute_64 = context_layer_36.permute(0, 2, 1, 3)
        context_layer_36 = None
        context_layer_37 = permute_64.contiguous()
        permute_64 = None
        context_layer_38 = context_layer_37.view(1, 197, 1024)
        context_layer_37 = None
        hidden_states_84 = torch._C._nn.linear(
            context_layer_38,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_38 = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_85 = torch.nn.functional.dropout(
            hidden_states_84, 0.0, False, False
        )
        hidden_states_84 = None
        attention_output_12 = (
            l_self_modules_encoder_modules_layer_modules_12_parameters_lambda_1_
            * hidden_states_85
        )
        l_self_modules_encoder_modules_layer_modules_12_parameters_lambda_1_ = (
            hidden_states_85
        ) = None
        hidden_states_86 = attention_output_12 + layer_output_35
        attention_output_12 = layer_output_35 = None
        layer_output_36 = torch.nn.functional.layer_norm(
            hidden_states_86,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_after_parameters_bias_ = (None)
        hidden_states_87 = torch._C._nn.linear(
            layer_output_36,
            l_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_36 = l_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_88 = torch._C._nn.gelu(hidden_states_87)
        hidden_states_87 = None
        hidden_states_89 = torch._C._nn.linear(
            hidden_states_88,
            l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_88 = l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_90 = torch.nn.functional.dropout(
            hidden_states_89, 0.0, False, False
        )
        hidden_states_89 = None
        layer_output_37 = (
            l_self_modules_encoder_modules_layer_modules_12_parameters_lambda_2_
            * hidden_states_90
        )
        l_self_modules_encoder_modules_layer_modules_12_parameters_lambda_2_ = (
            hidden_states_90
        ) = None
        layer_output_38 = layer_output_37 + hidden_states_86
        layer_output_37 = hidden_states_86 = None
        old_sub_table_26 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(None, 729, None)
        ]
        reshape_26 = old_sub_table_26.reshape(1, 27, 27, -1)
        old_sub_table_26 = None
        old_sub_table_27 = reshape_26.permute(0, 3, 1, 2)
        reshape_26 = None
        new_sub_table_26 = torch.nn.functional.interpolate(
            old_sub_table_27, size=(27, 27), mode="bilinear"
        )
        old_sub_table_27 = None
        permute_66 = new_sub_table_26.permute(0, 2, 3, 1)
        new_sub_table_26 = None
        new_sub_table_27 = permute_66.reshape(729, -1)
        permute_66 = None
        getitem_131 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(729, None, None)
        ]
        new_relative_position_bias_table_13 = torch.cat([new_sub_table_27, getitem_131])
        new_sub_table_27 = getitem_131 = None
        arange_26 = torch.arange(14)
        arange_27 = torch.arange(14)
        meshgrid_13 = torch.functional.meshgrid(arange_26, arange_27, indexing="ij")
        arange_26 = arange_27 = None
        getitem_132 = meshgrid_13[0]
        getitem_133 = meshgrid_13[1]
        meshgrid_13 = None
        coords_13 = torch.stack((getitem_132, getitem_133))
        getitem_132 = getitem_133 = None
        coords_flatten_13 = torch.flatten(coords_13, 1)
        coords_13 = None
        getitem_134 = coords_flatten_13[
            (slice(None, None, None), slice(None, None, None), None)
        ]
        getitem_135 = coords_flatten_13[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        coords_flatten_13 = None
        relative_coords_26 = getitem_134 - getitem_135
        getitem_134 = getitem_135 = None
        permute_67 = relative_coords_26.permute(1, 2, 0)
        relative_coords_26 = None
        relative_coords_27 = permute_67.contiguous()
        permute_67 = None
        getitem_136 = relative_coords_27[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_136 += 13
        iadd_26 = getitem_136
        getitem_136 = None
        relative_coords_27[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = iadd_26
        setitem_91 = relative_coords_27
        iadd_26 = setitem_91 = None
        getitem_137 = relative_coords_27[
            (slice(None, None, None), slice(None, None, None), 1)
        ]
        getitem_137 += 13
        iadd_27 = getitem_137
        getitem_137 = None
        relative_coords_27[
            (slice(None, None, None), slice(None, None, None), 1)
        ] = iadd_27
        setitem_92 = relative_coords_27
        iadd_27 = setitem_92 = None
        getitem_138 = relative_coords_27[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_138 *= 27
        imul_13 = getitem_138
        getitem_138 = None
        relative_coords_27[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = imul_13
        setitem_93 = relative_coords_27
        imul_13 = setitem_93 = None
        relative_position_index_13 = torch.zeros(size=(197, 197), dtype=torch.int64)
        sum_14 = relative_coords_27.sum(-1)
        relative_coords_27 = None
        relative_position_index_13[
            (slice(1, None, None), slice(1, None, None))
        ] = sum_14
        setitem_94 = relative_position_index_13
        sum_14 = setitem_94 = None
        relative_position_index_13[(0, slice(0, None, None))] = 729
        setitem_95 = relative_position_index_13
        setitem_95 = None
        relative_position_index_13[(slice(0, None, None), 0)] = 730
        setitem_96 = relative_position_index_13
        setitem_96 = None
        relative_position_index_13[(0, 0)] = 731
        setitem_97 = relative_position_index_13
        setitem_97 = None
        view_78 = relative_position_index_13.view(-1)
        relative_position_index_13 = None
        relative_position_bias_52 = new_relative_position_bias_table_13[view_78]
        new_relative_position_bias_table_13 = view_78 = None
        relative_position_bias_53 = relative_position_bias_52.view(197, 197, -1)
        relative_position_bias_52 = None
        permute_68 = relative_position_bias_53.permute(2, 0, 1)
        relative_position_bias_53 = None
        relative_position_bias_54 = permute_68.contiguous()
        permute_68 = None
        relative_position_bias_55 = relative_position_bias_54.unsqueeze(0)
        relative_position_bias_54 = None
        layer_norm_26 = torch.nn.functional.layer_norm(
            layer_output_38,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_before_parameters_bias_ = (None)
        linear_78 = torch._C._nn.linear(
            layer_norm_26,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_80 = linear_78.view(1, -1, 16, 64)
        linear_78 = None
        query_layer_13 = view_80.transpose(1, 2)
        view_80 = None
        linear_79 = torch._C._nn.linear(
            layer_norm_26,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_81 = linear_79.view(1, -1, 16, 64)
        linear_79 = None
        key_layer_13 = view_81.transpose(1, 2)
        view_81 = None
        linear_80 = torch._C._nn.linear(
            layer_norm_26,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        layer_norm_26 = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_82 = linear_80.view(1, -1, 16, 64)
        linear_80 = None
        value_layer_13 = view_82.transpose(1, 2)
        view_82 = None
        context_layer_39 = torch._C._nn.scaled_dot_product_attention(
            query_layer_13,
            key_layer_13,
            value_layer_13,
            attn_mask=relative_position_bias_55,
            dropout_p=0.0,
            is_causal=False,
            scale=0.125,
        )
        query_layer_13 = (
            key_layer_13
        ) = value_layer_13 = relative_position_bias_55 = None
        permute_69 = context_layer_39.permute(0, 2, 1, 3)
        context_layer_39 = None
        context_layer_40 = permute_69.contiguous()
        permute_69 = None
        context_layer_41 = context_layer_40.view(1, 197, 1024)
        context_layer_40 = None
        hidden_states_91 = torch._C._nn.linear(
            context_layer_41,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_41 = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_92 = torch.nn.functional.dropout(
            hidden_states_91, 0.0, False, False
        )
        hidden_states_91 = None
        attention_output_13 = (
            l_self_modules_encoder_modules_layer_modules_13_parameters_lambda_1_
            * hidden_states_92
        )
        l_self_modules_encoder_modules_layer_modules_13_parameters_lambda_1_ = (
            hidden_states_92
        ) = None
        hidden_states_93 = attention_output_13 + layer_output_38
        attention_output_13 = layer_output_38 = None
        layer_output_39 = torch.nn.functional.layer_norm(
            hidden_states_93,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_after_parameters_bias_ = (None)
        hidden_states_94 = torch._C._nn.linear(
            layer_output_39,
            l_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_39 = l_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_95 = torch._C._nn.gelu(hidden_states_94)
        hidden_states_94 = None
        hidden_states_96 = torch._C._nn.linear(
            hidden_states_95,
            l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_95 = l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_97 = torch.nn.functional.dropout(
            hidden_states_96, 0.0, False, False
        )
        hidden_states_96 = None
        layer_output_40 = (
            l_self_modules_encoder_modules_layer_modules_13_parameters_lambda_2_
            * hidden_states_97
        )
        l_self_modules_encoder_modules_layer_modules_13_parameters_lambda_2_ = (
            hidden_states_97
        ) = None
        layer_output_41 = layer_output_40 + hidden_states_93
        layer_output_40 = hidden_states_93 = None
        old_sub_table_28 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(None, 729, None)
        ]
        reshape_28 = old_sub_table_28.reshape(1, 27, 27, -1)
        old_sub_table_28 = None
        old_sub_table_29 = reshape_28.permute(0, 3, 1, 2)
        reshape_28 = None
        new_sub_table_28 = torch.nn.functional.interpolate(
            old_sub_table_29, size=(27, 27), mode="bilinear"
        )
        old_sub_table_29 = None
        permute_71 = new_sub_table_28.permute(0, 2, 3, 1)
        new_sub_table_28 = None
        new_sub_table_29 = permute_71.reshape(729, -1)
        permute_71 = None
        getitem_141 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(729, None, None)
        ]
        new_relative_position_bias_table_14 = torch.cat([new_sub_table_29, getitem_141])
        new_sub_table_29 = getitem_141 = None
        arange_28 = torch.arange(14)
        arange_29 = torch.arange(14)
        meshgrid_14 = torch.functional.meshgrid(arange_28, arange_29, indexing="ij")
        arange_28 = arange_29 = None
        getitem_142 = meshgrid_14[0]
        getitem_143 = meshgrid_14[1]
        meshgrid_14 = None
        coords_14 = torch.stack((getitem_142, getitem_143))
        getitem_142 = getitem_143 = None
        coords_flatten_14 = torch.flatten(coords_14, 1)
        coords_14 = None
        getitem_144 = coords_flatten_14[
            (slice(None, None, None), slice(None, None, None), None)
        ]
        getitem_145 = coords_flatten_14[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        coords_flatten_14 = None
        relative_coords_28 = getitem_144 - getitem_145
        getitem_144 = getitem_145 = None
        permute_72 = relative_coords_28.permute(1, 2, 0)
        relative_coords_28 = None
        relative_coords_29 = permute_72.contiguous()
        permute_72 = None
        getitem_146 = relative_coords_29[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_146 += 13
        iadd_28 = getitem_146
        getitem_146 = None
        relative_coords_29[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = iadd_28
        setitem_98 = relative_coords_29
        iadd_28 = setitem_98 = None
        getitem_147 = relative_coords_29[
            (slice(None, None, None), slice(None, None, None), 1)
        ]
        getitem_147 += 13
        iadd_29 = getitem_147
        getitem_147 = None
        relative_coords_29[
            (slice(None, None, None), slice(None, None, None), 1)
        ] = iadd_29
        setitem_99 = relative_coords_29
        iadd_29 = setitem_99 = None
        getitem_148 = relative_coords_29[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_148 *= 27
        imul_14 = getitem_148
        getitem_148 = None
        relative_coords_29[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = imul_14
        setitem_100 = relative_coords_29
        imul_14 = setitem_100 = None
        relative_position_index_14 = torch.zeros(size=(197, 197), dtype=torch.int64)
        sum_15 = relative_coords_29.sum(-1)
        relative_coords_29 = None
        relative_position_index_14[
            (slice(1, None, None), slice(1, None, None))
        ] = sum_15
        setitem_101 = relative_position_index_14
        sum_15 = setitem_101 = None
        relative_position_index_14[(0, slice(0, None, None))] = 729
        setitem_102 = relative_position_index_14
        setitem_102 = None
        relative_position_index_14[(slice(0, None, None), 0)] = 730
        setitem_103 = relative_position_index_14
        setitem_103 = None
        relative_position_index_14[(0, 0)] = 731
        setitem_104 = relative_position_index_14
        setitem_104 = None
        view_84 = relative_position_index_14.view(-1)
        relative_position_index_14 = None
        relative_position_bias_56 = new_relative_position_bias_table_14[view_84]
        new_relative_position_bias_table_14 = view_84 = None
        relative_position_bias_57 = relative_position_bias_56.view(197, 197, -1)
        relative_position_bias_56 = None
        permute_73 = relative_position_bias_57.permute(2, 0, 1)
        relative_position_bias_57 = None
        relative_position_bias_58 = permute_73.contiguous()
        permute_73 = None
        relative_position_bias_59 = relative_position_bias_58.unsqueeze(0)
        relative_position_bias_58 = None
        layer_norm_28 = torch.nn.functional.layer_norm(
            layer_output_41,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_before_parameters_bias_ = (None)
        linear_84 = torch._C._nn.linear(
            layer_norm_28,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_86 = linear_84.view(1, -1, 16, 64)
        linear_84 = None
        query_layer_14 = view_86.transpose(1, 2)
        view_86 = None
        linear_85 = torch._C._nn.linear(
            layer_norm_28,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_87 = linear_85.view(1, -1, 16, 64)
        linear_85 = None
        key_layer_14 = view_87.transpose(1, 2)
        view_87 = None
        linear_86 = torch._C._nn.linear(
            layer_norm_28,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        layer_norm_28 = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_88 = linear_86.view(1, -1, 16, 64)
        linear_86 = None
        value_layer_14 = view_88.transpose(1, 2)
        view_88 = None
        context_layer_42 = torch._C._nn.scaled_dot_product_attention(
            query_layer_14,
            key_layer_14,
            value_layer_14,
            attn_mask=relative_position_bias_59,
            dropout_p=0.0,
            is_causal=False,
            scale=0.125,
        )
        query_layer_14 = (
            key_layer_14
        ) = value_layer_14 = relative_position_bias_59 = None
        permute_74 = context_layer_42.permute(0, 2, 1, 3)
        context_layer_42 = None
        context_layer_43 = permute_74.contiguous()
        permute_74 = None
        context_layer_44 = context_layer_43.view(1, 197, 1024)
        context_layer_43 = None
        hidden_states_98 = torch._C._nn.linear(
            context_layer_44,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_44 = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_99 = torch.nn.functional.dropout(
            hidden_states_98, 0.0, False, False
        )
        hidden_states_98 = None
        attention_output_14 = (
            l_self_modules_encoder_modules_layer_modules_14_parameters_lambda_1_
            * hidden_states_99
        )
        l_self_modules_encoder_modules_layer_modules_14_parameters_lambda_1_ = (
            hidden_states_99
        ) = None
        hidden_states_100 = attention_output_14 + layer_output_41
        attention_output_14 = layer_output_41 = None
        layer_output_42 = torch.nn.functional.layer_norm(
            hidden_states_100,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_after_parameters_bias_ = (None)
        hidden_states_101 = torch._C._nn.linear(
            layer_output_42,
            l_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_42 = l_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_102 = torch._C._nn.gelu(hidden_states_101)
        hidden_states_101 = None
        hidden_states_103 = torch._C._nn.linear(
            hidden_states_102,
            l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_102 = l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_104 = torch.nn.functional.dropout(
            hidden_states_103, 0.0, False, False
        )
        hidden_states_103 = None
        layer_output_43 = (
            l_self_modules_encoder_modules_layer_modules_14_parameters_lambda_2_
            * hidden_states_104
        )
        l_self_modules_encoder_modules_layer_modules_14_parameters_lambda_2_ = (
            hidden_states_104
        ) = None
        layer_output_44 = layer_output_43 + hidden_states_100
        layer_output_43 = hidden_states_100 = None
        old_sub_table_30 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(None, 729, None)
        ]
        reshape_30 = old_sub_table_30.reshape(1, 27, 27, -1)
        old_sub_table_30 = None
        old_sub_table_31 = reshape_30.permute(0, 3, 1, 2)
        reshape_30 = None
        new_sub_table_30 = torch.nn.functional.interpolate(
            old_sub_table_31, size=(27, 27), mode="bilinear"
        )
        old_sub_table_31 = None
        permute_76 = new_sub_table_30.permute(0, 2, 3, 1)
        new_sub_table_30 = None
        new_sub_table_31 = permute_76.reshape(729, -1)
        permute_76 = None
        getitem_151 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(729, None, None)
        ]
        new_relative_position_bias_table_15 = torch.cat([new_sub_table_31, getitem_151])
        new_sub_table_31 = getitem_151 = None
        arange_30 = torch.arange(14)
        arange_31 = torch.arange(14)
        meshgrid_15 = torch.functional.meshgrid(arange_30, arange_31, indexing="ij")
        arange_30 = arange_31 = None
        getitem_152 = meshgrid_15[0]
        getitem_153 = meshgrid_15[1]
        meshgrid_15 = None
        coords_15 = torch.stack((getitem_152, getitem_153))
        getitem_152 = getitem_153 = None
        coords_flatten_15 = torch.flatten(coords_15, 1)
        coords_15 = None
        getitem_154 = coords_flatten_15[
            (slice(None, None, None), slice(None, None, None), None)
        ]
        getitem_155 = coords_flatten_15[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        coords_flatten_15 = None
        relative_coords_30 = getitem_154 - getitem_155
        getitem_154 = getitem_155 = None
        permute_77 = relative_coords_30.permute(1, 2, 0)
        relative_coords_30 = None
        relative_coords_31 = permute_77.contiguous()
        permute_77 = None
        getitem_156 = relative_coords_31[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_156 += 13
        iadd_30 = getitem_156
        getitem_156 = None
        relative_coords_31[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = iadd_30
        setitem_105 = relative_coords_31
        iadd_30 = setitem_105 = None
        getitem_157 = relative_coords_31[
            (slice(None, None, None), slice(None, None, None), 1)
        ]
        getitem_157 += 13
        iadd_31 = getitem_157
        getitem_157 = None
        relative_coords_31[
            (slice(None, None, None), slice(None, None, None), 1)
        ] = iadd_31
        setitem_106 = relative_coords_31
        iadd_31 = setitem_106 = None
        getitem_158 = relative_coords_31[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_158 *= 27
        imul_15 = getitem_158
        getitem_158 = None
        relative_coords_31[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = imul_15
        setitem_107 = relative_coords_31
        imul_15 = setitem_107 = None
        relative_position_index_15 = torch.zeros(size=(197, 197), dtype=torch.int64)
        sum_16 = relative_coords_31.sum(-1)
        relative_coords_31 = None
        relative_position_index_15[
            (slice(1, None, None), slice(1, None, None))
        ] = sum_16
        setitem_108 = relative_position_index_15
        sum_16 = setitem_108 = None
        relative_position_index_15[(0, slice(0, None, None))] = 729
        setitem_109 = relative_position_index_15
        setitem_109 = None
        relative_position_index_15[(slice(0, None, None), 0)] = 730
        setitem_110 = relative_position_index_15
        setitem_110 = None
        relative_position_index_15[(0, 0)] = 731
        setitem_111 = relative_position_index_15
        setitem_111 = None
        view_90 = relative_position_index_15.view(-1)
        relative_position_index_15 = None
        relative_position_bias_60 = new_relative_position_bias_table_15[view_90]
        new_relative_position_bias_table_15 = view_90 = None
        relative_position_bias_61 = relative_position_bias_60.view(197, 197, -1)
        relative_position_bias_60 = None
        permute_78 = relative_position_bias_61.permute(2, 0, 1)
        relative_position_bias_61 = None
        relative_position_bias_62 = permute_78.contiguous()
        permute_78 = None
        relative_position_bias_63 = relative_position_bias_62.unsqueeze(0)
        relative_position_bias_62 = None
        layer_norm_30 = torch.nn.functional.layer_norm(
            layer_output_44,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_before_parameters_bias_ = (None)
        linear_90 = torch._C._nn.linear(
            layer_norm_30,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_92 = linear_90.view(1, -1, 16, 64)
        linear_90 = None
        query_layer_15 = view_92.transpose(1, 2)
        view_92 = None
        linear_91 = torch._C._nn.linear(
            layer_norm_30,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_93 = linear_91.view(1, -1, 16, 64)
        linear_91 = None
        key_layer_15 = view_93.transpose(1, 2)
        view_93 = None
        linear_92 = torch._C._nn.linear(
            layer_norm_30,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        layer_norm_30 = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_94 = linear_92.view(1, -1, 16, 64)
        linear_92 = None
        value_layer_15 = view_94.transpose(1, 2)
        view_94 = None
        context_layer_45 = torch._C._nn.scaled_dot_product_attention(
            query_layer_15,
            key_layer_15,
            value_layer_15,
            attn_mask=relative_position_bias_63,
            dropout_p=0.0,
            is_causal=False,
            scale=0.125,
        )
        query_layer_15 = (
            key_layer_15
        ) = value_layer_15 = relative_position_bias_63 = None
        permute_79 = context_layer_45.permute(0, 2, 1, 3)
        context_layer_45 = None
        context_layer_46 = permute_79.contiguous()
        permute_79 = None
        context_layer_47 = context_layer_46.view(1, 197, 1024)
        context_layer_46 = None
        hidden_states_105 = torch._C._nn.linear(
            context_layer_47,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_47 = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_106 = torch.nn.functional.dropout(
            hidden_states_105, 0.0, False, False
        )
        hidden_states_105 = None
        attention_output_15 = (
            l_self_modules_encoder_modules_layer_modules_15_parameters_lambda_1_
            * hidden_states_106
        )
        l_self_modules_encoder_modules_layer_modules_15_parameters_lambda_1_ = (
            hidden_states_106
        ) = None
        hidden_states_107 = attention_output_15 + layer_output_44
        attention_output_15 = layer_output_44 = None
        layer_output_45 = torch.nn.functional.layer_norm(
            hidden_states_107,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_after_parameters_bias_ = (None)
        hidden_states_108 = torch._C._nn.linear(
            layer_output_45,
            l_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_45 = l_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_109 = torch._C._nn.gelu(hidden_states_108)
        hidden_states_108 = None
        hidden_states_110 = torch._C._nn.linear(
            hidden_states_109,
            l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_109 = l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_111 = torch.nn.functional.dropout(
            hidden_states_110, 0.0, False, False
        )
        hidden_states_110 = None
        layer_output_46 = (
            l_self_modules_encoder_modules_layer_modules_15_parameters_lambda_2_
            * hidden_states_111
        )
        l_self_modules_encoder_modules_layer_modules_15_parameters_lambda_2_ = (
            hidden_states_111
        ) = None
        layer_output_47 = layer_output_46 + hidden_states_107
        layer_output_46 = hidden_states_107 = None
        old_sub_table_32 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(None, 729, None)
        ]
        reshape_32 = old_sub_table_32.reshape(1, 27, 27, -1)
        old_sub_table_32 = None
        old_sub_table_33 = reshape_32.permute(0, 3, 1, 2)
        reshape_32 = None
        new_sub_table_32 = torch.nn.functional.interpolate(
            old_sub_table_33, size=(27, 27), mode="bilinear"
        )
        old_sub_table_33 = None
        permute_81 = new_sub_table_32.permute(0, 2, 3, 1)
        new_sub_table_32 = None
        new_sub_table_33 = permute_81.reshape(729, -1)
        permute_81 = None
        getitem_161 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(729, None, None)
        ]
        new_relative_position_bias_table_16 = torch.cat([new_sub_table_33, getitem_161])
        new_sub_table_33 = getitem_161 = None
        arange_32 = torch.arange(14)
        arange_33 = torch.arange(14)
        meshgrid_16 = torch.functional.meshgrid(arange_32, arange_33, indexing="ij")
        arange_32 = arange_33 = None
        getitem_162 = meshgrid_16[0]
        getitem_163 = meshgrid_16[1]
        meshgrid_16 = None
        coords_16 = torch.stack((getitem_162, getitem_163))
        getitem_162 = getitem_163 = None
        coords_flatten_16 = torch.flatten(coords_16, 1)
        coords_16 = None
        getitem_164 = coords_flatten_16[
            (slice(None, None, None), slice(None, None, None), None)
        ]
        getitem_165 = coords_flatten_16[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        coords_flatten_16 = None
        relative_coords_32 = getitem_164 - getitem_165
        getitem_164 = getitem_165 = None
        permute_82 = relative_coords_32.permute(1, 2, 0)
        relative_coords_32 = None
        relative_coords_33 = permute_82.contiguous()
        permute_82 = None
        getitem_166 = relative_coords_33[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_166 += 13
        iadd_32 = getitem_166
        getitem_166 = None
        relative_coords_33[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = iadd_32
        setitem_112 = relative_coords_33
        iadd_32 = setitem_112 = None
        getitem_167 = relative_coords_33[
            (slice(None, None, None), slice(None, None, None), 1)
        ]
        getitem_167 += 13
        iadd_33 = getitem_167
        getitem_167 = None
        relative_coords_33[
            (slice(None, None, None), slice(None, None, None), 1)
        ] = iadd_33
        setitem_113 = relative_coords_33
        iadd_33 = setitem_113 = None
        getitem_168 = relative_coords_33[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_168 *= 27
        imul_16 = getitem_168
        getitem_168 = None
        relative_coords_33[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = imul_16
        setitem_114 = relative_coords_33
        imul_16 = setitem_114 = None
        relative_position_index_16 = torch.zeros(size=(197, 197), dtype=torch.int64)
        sum_17 = relative_coords_33.sum(-1)
        relative_coords_33 = None
        relative_position_index_16[
            (slice(1, None, None), slice(1, None, None))
        ] = sum_17
        setitem_115 = relative_position_index_16
        sum_17 = setitem_115 = None
        relative_position_index_16[(0, slice(0, None, None))] = 729
        setitem_116 = relative_position_index_16
        setitem_116 = None
        relative_position_index_16[(slice(0, None, None), 0)] = 730
        setitem_117 = relative_position_index_16
        setitem_117 = None
        relative_position_index_16[(0, 0)] = 731
        setitem_118 = relative_position_index_16
        setitem_118 = None
        view_96 = relative_position_index_16.view(-1)
        relative_position_index_16 = None
        relative_position_bias_64 = new_relative_position_bias_table_16[view_96]
        new_relative_position_bias_table_16 = view_96 = None
        relative_position_bias_65 = relative_position_bias_64.view(197, 197, -1)
        relative_position_bias_64 = None
        permute_83 = relative_position_bias_65.permute(2, 0, 1)
        relative_position_bias_65 = None
        relative_position_bias_66 = permute_83.contiguous()
        permute_83 = None
        relative_position_bias_67 = relative_position_bias_66.unsqueeze(0)
        relative_position_bias_66 = None
        layer_norm_32 = torch.nn.functional.layer_norm(
            layer_output_47,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_before_parameters_bias_ = (None)
        linear_96 = torch._C._nn.linear(
            layer_norm_32,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_98 = linear_96.view(1, -1, 16, 64)
        linear_96 = None
        query_layer_16 = view_98.transpose(1, 2)
        view_98 = None
        linear_97 = torch._C._nn.linear(
            layer_norm_32,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_99 = linear_97.view(1, -1, 16, 64)
        linear_97 = None
        key_layer_16 = view_99.transpose(1, 2)
        view_99 = None
        linear_98 = torch._C._nn.linear(
            layer_norm_32,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        layer_norm_32 = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_100 = linear_98.view(1, -1, 16, 64)
        linear_98 = None
        value_layer_16 = view_100.transpose(1, 2)
        view_100 = None
        context_layer_48 = torch._C._nn.scaled_dot_product_attention(
            query_layer_16,
            key_layer_16,
            value_layer_16,
            attn_mask=relative_position_bias_67,
            dropout_p=0.0,
            is_causal=False,
            scale=0.125,
        )
        query_layer_16 = (
            key_layer_16
        ) = value_layer_16 = relative_position_bias_67 = None
        permute_84 = context_layer_48.permute(0, 2, 1, 3)
        context_layer_48 = None
        context_layer_49 = permute_84.contiguous()
        permute_84 = None
        context_layer_50 = context_layer_49.view(1, 197, 1024)
        context_layer_49 = None
        hidden_states_112 = torch._C._nn.linear(
            context_layer_50,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_50 = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_113 = torch.nn.functional.dropout(
            hidden_states_112, 0.0, False, False
        )
        hidden_states_112 = None
        attention_output_16 = (
            l_self_modules_encoder_modules_layer_modules_16_parameters_lambda_1_
            * hidden_states_113
        )
        l_self_modules_encoder_modules_layer_modules_16_parameters_lambda_1_ = (
            hidden_states_113
        ) = None
        hidden_states_114 = attention_output_16 + layer_output_47
        attention_output_16 = layer_output_47 = None
        layer_output_48 = torch.nn.functional.layer_norm(
            hidden_states_114,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_after_parameters_bias_ = (None)
        hidden_states_115 = torch._C._nn.linear(
            layer_output_48,
            l_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_48 = l_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_116 = torch._C._nn.gelu(hidden_states_115)
        hidden_states_115 = None
        hidden_states_117 = torch._C._nn.linear(
            hidden_states_116,
            l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_116 = l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_118 = torch.nn.functional.dropout(
            hidden_states_117, 0.0, False, False
        )
        hidden_states_117 = None
        layer_output_49 = (
            l_self_modules_encoder_modules_layer_modules_16_parameters_lambda_2_
            * hidden_states_118
        )
        l_self_modules_encoder_modules_layer_modules_16_parameters_lambda_2_ = (
            hidden_states_118
        ) = None
        layer_output_50 = layer_output_49 + hidden_states_114
        layer_output_49 = hidden_states_114 = None
        old_sub_table_34 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(None, 729, None)
        ]
        reshape_34 = old_sub_table_34.reshape(1, 27, 27, -1)
        old_sub_table_34 = None
        old_sub_table_35 = reshape_34.permute(0, 3, 1, 2)
        reshape_34 = None
        new_sub_table_34 = torch.nn.functional.interpolate(
            old_sub_table_35, size=(27, 27), mode="bilinear"
        )
        old_sub_table_35 = None
        permute_86 = new_sub_table_34.permute(0, 2, 3, 1)
        new_sub_table_34 = None
        new_sub_table_35 = permute_86.reshape(729, -1)
        permute_86 = None
        getitem_171 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(729, None, None)
        ]
        new_relative_position_bias_table_17 = torch.cat([new_sub_table_35, getitem_171])
        new_sub_table_35 = getitem_171 = None
        arange_34 = torch.arange(14)
        arange_35 = torch.arange(14)
        meshgrid_17 = torch.functional.meshgrid(arange_34, arange_35, indexing="ij")
        arange_34 = arange_35 = None
        getitem_172 = meshgrid_17[0]
        getitem_173 = meshgrid_17[1]
        meshgrid_17 = None
        coords_17 = torch.stack((getitem_172, getitem_173))
        getitem_172 = getitem_173 = None
        coords_flatten_17 = torch.flatten(coords_17, 1)
        coords_17 = None
        getitem_174 = coords_flatten_17[
            (slice(None, None, None), slice(None, None, None), None)
        ]
        getitem_175 = coords_flatten_17[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        coords_flatten_17 = None
        relative_coords_34 = getitem_174 - getitem_175
        getitem_174 = getitem_175 = None
        permute_87 = relative_coords_34.permute(1, 2, 0)
        relative_coords_34 = None
        relative_coords_35 = permute_87.contiguous()
        permute_87 = None
        getitem_176 = relative_coords_35[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_176 += 13
        iadd_34 = getitem_176
        getitem_176 = None
        relative_coords_35[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = iadd_34
        setitem_119 = relative_coords_35
        iadd_34 = setitem_119 = None
        getitem_177 = relative_coords_35[
            (slice(None, None, None), slice(None, None, None), 1)
        ]
        getitem_177 += 13
        iadd_35 = getitem_177
        getitem_177 = None
        relative_coords_35[
            (slice(None, None, None), slice(None, None, None), 1)
        ] = iadd_35
        setitem_120 = relative_coords_35
        iadd_35 = setitem_120 = None
        getitem_178 = relative_coords_35[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_178 *= 27
        imul_17 = getitem_178
        getitem_178 = None
        relative_coords_35[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = imul_17
        setitem_121 = relative_coords_35
        imul_17 = setitem_121 = None
        relative_position_index_17 = torch.zeros(size=(197, 197), dtype=torch.int64)
        sum_18 = relative_coords_35.sum(-1)
        relative_coords_35 = None
        relative_position_index_17[
            (slice(1, None, None), slice(1, None, None))
        ] = sum_18
        setitem_122 = relative_position_index_17
        sum_18 = setitem_122 = None
        relative_position_index_17[(0, slice(0, None, None))] = 729
        setitem_123 = relative_position_index_17
        setitem_123 = None
        relative_position_index_17[(slice(0, None, None), 0)] = 730
        setitem_124 = relative_position_index_17
        setitem_124 = None
        relative_position_index_17[(0, 0)] = 731
        setitem_125 = relative_position_index_17
        setitem_125 = None
        view_102 = relative_position_index_17.view(-1)
        relative_position_index_17 = None
        relative_position_bias_68 = new_relative_position_bias_table_17[view_102]
        new_relative_position_bias_table_17 = view_102 = None
        relative_position_bias_69 = relative_position_bias_68.view(197, 197, -1)
        relative_position_bias_68 = None
        permute_88 = relative_position_bias_69.permute(2, 0, 1)
        relative_position_bias_69 = None
        relative_position_bias_70 = permute_88.contiguous()
        permute_88 = None
        relative_position_bias_71 = relative_position_bias_70.unsqueeze(0)
        relative_position_bias_70 = None
        layer_norm_34 = torch.nn.functional.layer_norm(
            layer_output_50,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_before_parameters_bias_ = (None)
        linear_102 = torch._C._nn.linear(
            layer_norm_34,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_104 = linear_102.view(1, -1, 16, 64)
        linear_102 = None
        query_layer_17 = view_104.transpose(1, 2)
        view_104 = None
        linear_103 = torch._C._nn.linear(
            layer_norm_34,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_105 = linear_103.view(1, -1, 16, 64)
        linear_103 = None
        key_layer_17 = view_105.transpose(1, 2)
        view_105 = None
        linear_104 = torch._C._nn.linear(
            layer_norm_34,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        layer_norm_34 = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_106 = linear_104.view(1, -1, 16, 64)
        linear_104 = None
        value_layer_17 = view_106.transpose(1, 2)
        view_106 = None
        context_layer_51 = torch._C._nn.scaled_dot_product_attention(
            query_layer_17,
            key_layer_17,
            value_layer_17,
            attn_mask=relative_position_bias_71,
            dropout_p=0.0,
            is_causal=False,
            scale=0.125,
        )
        query_layer_17 = (
            key_layer_17
        ) = value_layer_17 = relative_position_bias_71 = None
        permute_89 = context_layer_51.permute(0, 2, 1, 3)
        context_layer_51 = None
        context_layer_52 = permute_89.contiguous()
        permute_89 = None
        context_layer_53 = context_layer_52.view(1, 197, 1024)
        context_layer_52 = None
        hidden_states_119 = torch._C._nn.linear(
            context_layer_53,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_53 = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_120 = torch.nn.functional.dropout(
            hidden_states_119, 0.0, False, False
        )
        hidden_states_119 = None
        attention_output_17 = (
            l_self_modules_encoder_modules_layer_modules_17_parameters_lambda_1_
            * hidden_states_120
        )
        l_self_modules_encoder_modules_layer_modules_17_parameters_lambda_1_ = (
            hidden_states_120
        ) = None
        hidden_states_121 = attention_output_17 + layer_output_50
        attention_output_17 = layer_output_50 = None
        layer_output_51 = torch.nn.functional.layer_norm(
            hidden_states_121,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_after_parameters_bias_ = (None)
        hidden_states_122 = torch._C._nn.linear(
            layer_output_51,
            l_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_51 = l_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_123 = torch._C._nn.gelu(hidden_states_122)
        hidden_states_122 = None
        hidden_states_124 = torch._C._nn.linear(
            hidden_states_123,
            l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_123 = l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_125 = torch.nn.functional.dropout(
            hidden_states_124, 0.0, False, False
        )
        hidden_states_124 = None
        layer_output_52 = (
            l_self_modules_encoder_modules_layer_modules_17_parameters_lambda_2_
            * hidden_states_125
        )
        l_self_modules_encoder_modules_layer_modules_17_parameters_lambda_2_ = (
            hidden_states_125
        ) = None
        layer_output_53 = layer_output_52 + hidden_states_121
        layer_output_52 = hidden_states_121 = None
        old_sub_table_36 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(None, 729, None)
        ]
        reshape_36 = old_sub_table_36.reshape(1, 27, 27, -1)
        old_sub_table_36 = None
        old_sub_table_37 = reshape_36.permute(0, 3, 1, 2)
        reshape_36 = None
        new_sub_table_36 = torch.nn.functional.interpolate(
            old_sub_table_37, size=(27, 27), mode="bilinear"
        )
        old_sub_table_37 = None
        permute_91 = new_sub_table_36.permute(0, 2, 3, 1)
        new_sub_table_36 = None
        new_sub_table_37 = permute_91.reshape(729, -1)
        permute_91 = None
        getitem_181 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(729, None, None)
        ]
        new_relative_position_bias_table_18 = torch.cat([new_sub_table_37, getitem_181])
        new_sub_table_37 = getitem_181 = None
        arange_36 = torch.arange(14)
        arange_37 = torch.arange(14)
        meshgrid_18 = torch.functional.meshgrid(arange_36, arange_37, indexing="ij")
        arange_36 = arange_37 = None
        getitem_182 = meshgrid_18[0]
        getitem_183 = meshgrid_18[1]
        meshgrid_18 = None
        coords_18 = torch.stack((getitem_182, getitem_183))
        getitem_182 = getitem_183 = None
        coords_flatten_18 = torch.flatten(coords_18, 1)
        coords_18 = None
        getitem_184 = coords_flatten_18[
            (slice(None, None, None), slice(None, None, None), None)
        ]
        getitem_185 = coords_flatten_18[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        coords_flatten_18 = None
        relative_coords_36 = getitem_184 - getitem_185
        getitem_184 = getitem_185 = None
        permute_92 = relative_coords_36.permute(1, 2, 0)
        relative_coords_36 = None
        relative_coords_37 = permute_92.contiguous()
        permute_92 = None
        getitem_186 = relative_coords_37[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_186 += 13
        iadd_36 = getitem_186
        getitem_186 = None
        relative_coords_37[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = iadd_36
        setitem_126 = relative_coords_37
        iadd_36 = setitem_126 = None
        getitem_187 = relative_coords_37[
            (slice(None, None, None), slice(None, None, None), 1)
        ]
        getitem_187 += 13
        iadd_37 = getitem_187
        getitem_187 = None
        relative_coords_37[
            (slice(None, None, None), slice(None, None, None), 1)
        ] = iadd_37
        setitem_127 = relative_coords_37
        iadd_37 = setitem_127 = None
        getitem_188 = relative_coords_37[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_188 *= 27
        imul_18 = getitem_188
        getitem_188 = None
        relative_coords_37[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = imul_18
        setitem_128 = relative_coords_37
        imul_18 = setitem_128 = None
        relative_position_index_18 = torch.zeros(size=(197, 197), dtype=torch.int64)
        sum_19 = relative_coords_37.sum(-1)
        relative_coords_37 = None
        relative_position_index_18[
            (slice(1, None, None), slice(1, None, None))
        ] = sum_19
        setitem_129 = relative_position_index_18
        sum_19 = setitem_129 = None
        relative_position_index_18[(0, slice(0, None, None))] = 729
        setitem_130 = relative_position_index_18
        setitem_130 = None
        relative_position_index_18[(slice(0, None, None), 0)] = 730
        setitem_131 = relative_position_index_18
        setitem_131 = None
        relative_position_index_18[(0, 0)] = 731
        setitem_132 = relative_position_index_18
        setitem_132 = None
        view_108 = relative_position_index_18.view(-1)
        relative_position_index_18 = None
        relative_position_bias_72 = new_relative_position_bias_table_18[view_108]
        new_relative_position_bias_table_18 = view_108 = None
        relative_position_bias_73 = relative_position_bias_72.view(197, 197, -1)
        relative_position_bias_72 = None
        permute_93 = relative_position_bias_73.permute(2, 0, 1)
        relative_position_bias_73 = None
        relative_position_bias_74 = permute_93.contiguous()
        permute_93 = None
        relative_position_bias_75 = relative_position_bias_74.unsqueeze(0)
        relative_position_bias_74 = None
        layer_norm_36 = torch.nn.functional.layer_norm(
            layer_output_53,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_before_parameters_bias_ = (None)
        linear_108 = torch._C._nn.linear(
            layer_norm_36,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_110 = linear_108.view(1, -1, 16, 64)
        linear_108 = None
        query_layer_18 = view_110.transpose(1, 2)
        view_110 = None
        linear_109 = torch._C._nn.linear(
            layer_norm_36,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_111 = linear_109.view(1, -1, 16, 64)
        linear_109 = None
        key_layer_18 = view_111.transpose(1, 2)
        view_111 = None
        linear_110 = torch._C._nn.linear(
            layer_norm_36,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        layer_norm_36 = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_112 = linear_110.view(1, -1, 16, 64)
        linear_110 = None
        value_layer_18 = view_112.transpose(1, 2)
        view_112 = None
        context_layer_54 = torch._C._nn.scaled_dot_product_attention(
            query_layer_18,
            key_layer_18,
            value_layer_18,
            attn_mask=relative_position_bias_75,
            dropout_p=0.0,
            is_causal=False,
            scale=0.125,
        )
        query_layer_18 = (
            key_layer_18
        ) = value_layer_18 = relative_position_bias_75 = None
        permute_94 = context_layer_54.permute(0, 2, 1, 3)
        context_layer_54 = None
        context_layer_55 = permute_94.contiguous()
        permute_94 = None
        context_layer_56 = context_layer_55.view(1, 197, 1024)
        context_layer_55 = None
        hidden_states_126 = torch._C._nn.linear(
            context_layer_56,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_56 = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_127 = torch.nn.functional.dropout(
            hidden_states_126, 0.0, False, False
        )
        hidden_states_126 = None
        attention_output_18 = (
            l_self_modules_encoder_modules_layer_modules_18_parameters_lambda_1_
            * hidden_states_127
        )
        l_self_modules_encoder_modules_layer_modules_18_parameters_lambda_1_ = (
            hidden_states_127
        ) = None
        hidden_states_128 = attention_output_18 + layer_output_53
        attention_output_18 = layer_output_53 = None
        layer_output_54 = torch.nn.functional.layer_norm(
            hidden_states_128,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_after_parameters_bias_ = (None)
        hidden_states_129 = torch._C._nn.linear(
            layer_output_54,
            l_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_54 = l_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_130 = torch._C._nn.gelu(hidden_states_129)
        hidden_states_129 = None
        hidden_states_131 = torch._C._nn.linear(
            hidden_states_130,
            l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_130 = l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_132 = torch.nn.functional.dropout(
            hidden_states_131, 0.0, False, False
        )
        hidden_states_131 = None
        layer_output_55 = (
            l_self_modules_encoder_modules_layer_modules_18_parameters_lambda_2_
            * hidden_states_132
        )
        l_self_modules_encoder_modules_layer_modules_18_parameters_lambda_2_ = (
            hidden_states_132
        ) = None
        layer_output_56 = layer_output_55 + hidden_states_128
        layer_output_55 = hidden_states_128 = None
        old_sub_table_38 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(None, 729, None)
        ]
        reshape_38 = old_sub_table_38.reshape(1, 27, 27, -1)
        old_sub_table_38 = None
        old_sub_table_39 = reshape_38.permute(0, 3, 1, 2)
        reshape_38 = None
        new_sub_table_38 = torch.nn.functional.interpolate(
            old_sub_table_39, size=(27, 27), mode="bilinear"
        )
        old_sub_table_39 = None
        permute_96 = new_sub_table_38.permute(0, 2, 3, 1)
        new_sub_table_38 = None
        new_sub_table_39 = permute_96.reshape(729, -1)
        permute_96 = None
        getitem_191 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(729, None, None)
        ]
        new_relative_position_bias_table_19 = torch.cat([new_sub_table_39, getitem_191])
        new_sub_table_39 = getitem_191 = None
        arange_38 = torch.arange(14)
        arange_39 = torch.arange(14)
        meshgrid_19 = torch.functional.meshgrid(arange_38, arange_39, indexing="ij")
        arange_38 = arange_39 = None
        getitem_192 = meshgrid_19[0]
        getitem_193 = meshgrid_19[1]
        meshgrid_19 = None
        coords_19 = torch.stack((getitem_192, getitem_193))
        getitem_192 = getitem_193 = None
        coords_flatten_19 = torch.flatten(coords_19, 1)
        coords_19 = None
        getitem_194 = coords_flatten_19[
            (slice(None, None, None), slice(None, None, None), None)
        ]
        getitem_195 = coords_flatten_19[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        coords_flatten_19 = None
        relative_coords_38 = getitem_194 - getitem_195
        getitem_194 = getitem_195 = None
        permute_97 = relative_coords_38.permute(1, 2, 0)
        relative_coords_38 = None
        relative_coords_39 = permute_97.contiguous()
        permute_97 = None
        getitem_196 = relative_coords_39[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_196 += 13
        iadd_38 = getitem_196
        getitem_196 = None
        relative_coords_39[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = iadd_38
        setitem_133 = relative_coords_39
        iadd_38 = setitem_133 = None
        getitem_197 = relative_coords_39[
            (slice(None, None, None), slice(None, None, None), 1)
        ]
        getitem_197 += 13
        iadd_39 = getitem_197
        getitem_197 = None
        relative_coords_39[
            (slice(None, None, None), slice(None, None, None), 1)
        ] = iadd_39
        setitem_134 = relative_coords_39
        iadd_39 = setitem_134 = None
        getitem_198 = relative_coords_39[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_198 *= 27
        imul_19 = getitem_198
        getitem_198 = None
        relative_coords_39[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = imul_19
        setitem_135 = relative_coords_39
        imul_19 = setitem_135 = None
        relative_position_index_19 = torch.zeros(size=(197, 197), dtype=torch.int64)
        sum_20 = relative_coords_39.sum(-1)
        relative_coords_39 = None
        relative_position_index_19[
            (slice(1, None, None), slice(1, None, None))
        ] = sum_20
        setitem_136 = relative_position_index_19
        sum_20 = setitem_136 = None
        relative_position_index_19[(0, slice(0, None, None))] = 729
        setitem_137 = relative_position_index_19
        setitem_137 = None
        relative_position_index_19[(slice(0, None, None), 0)] = 730
        setitem_138 = relative_position_index_19
        setitem_138 = None
        relative_position_index_19[(0, 0)] = 731
        setitem_139 = relative_position_index_19
        setitem_139 = None
        view_114 = relative_position_index_19.view(-1)
        relative_position_index_19 = None
        relative_position_bias_76 = new_relative_position_bias_table_19[view_114]
        new_relative_position_bias_table_19 = view_114 = None
        relative_position_bias_77 = relative_position_bias_76.view(197, 197, -1)
        relative_position_bias_76 = None
        permute_98 = relative_position_bias_77.permute(2, 0, 1)
        relative_position_bias_77 = None
        relative_position_bias_78 = permute_98.contiguous()
        permute_98 = None
        relative_position_bias_79 = relative_position_bias_78.unsqueeze(0)
        relative_position_bias_78 = None
        layer_norm_38 = torch.nn.functional.layer_norm(
            layer_output_56,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_before_parameters_bias_ = (None)
        linear_114 = torch._C._nn.linear(
            layer_norm_38,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_116 = linear_114.view(1, -1, 16, 64)
        linear_114 = None
        query_layer_19 = view_116.transpose(1, 2)
        view_116 = None
        linear_115 = torch._C._nn.linear(
            layer_norm_38,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_117 = linear_115.view(1, -1, 16, 64)
        linear_115 = None
        key_layer_19 = view_117.transpose(1, 2)
        view_117 = None
        linear_116 = torch._C._nn.linear(
            layer_norm_38,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        layer_norm_38 = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_118 = linear_116.view(1, -1, 16, 64)
        linear_116 = None
        value_layer_19 = view_118.transpose(1, 2)
        view_118 = None
        context_layer_57 = torch._C._nn.scaled_dot_product_attention(
            query_layer_19,
            key_layer_19,
            value_layer_19,
            attn_mask=relative_position_bias_79,
            dropout_p=0.0,
            is_causal=False,
            scale=0.125,
        )
        query_layer_19 = (
            key_layer_19
        ) = value_layer_19 = relative_position_bias_79 = None
        permute_99 = context_layer_57.permute(0, 2, 1, 3)
        context_layer_57 = None
        context_layer_58 = permute_99.contiguous()
        permute_99 = None
        context_layer_59 = context_layer_58.view(1, 197, 1024)
        context_layer_58 = None
        hidden_states_133 = torch._C._nn.linear(
            context_layer_59,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_59 = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_134 = torch.nn.functional.dropout(
            hidden_states_133, 0.0, False, False
        )
        hidden_states_133 = None
        attention_output_19 = (
            l_self_modules_encoder_modules_layer_modules_19_parameters_lambda_1_
            * hidden_states_134
        )
        l_self_modules_encoder_modules_layer_modules_19_parameters_lambda_1_ = (
            hidden_states_134
        ) = None
        hidden_states_135 = attention_output_19 + layer_output_56
        attention_output_19 = layer_output_56 = None
        layer_output_57 = torch.nn.functional.layer_norm(
            hidden_states_135,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_after_parameters_bias_ = (None)
        hidden_states_136 = torch._C._nn.linear(
            layer_output_57,
            l_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_57 = l_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_137 = torch._C._nn.gelu(hidden_states_136)
        hidden_states_136 = None
        hidden_states_138 = torch._C._nn.linear(
            hidden_states_137,
            l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_137 = l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_139 = torch.nn.functional.dropout(
            hidden_states_138, 0.0, False, False
        )
        hidden_states_138 = None
        layer_output_58 = (
            l_self_modules_encoder_modules_layer_modules_19_parameters_lambda_2_
            * hidden_states_139
        )
        l_self_modules_encoder_modules_layer_modules_19_parameters_lambda_2_ = (
            hidden_states_139
        ) = None
        layer_output_59 = layer_output_58 + hidden_states_135
        layer_output_58 = hidden_states_135 = None
        old_sub_table_40 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(None, 729, None)
        ]
        reshape_40 = old_sub_table_40.reshape(1, 27, 27, -1)
        old_sub_table_40 = None
        old_sub_table_41 = reshape_40.permute(0, 3, 1, 2)
        reshape_40 = None
        new_sub_table_40 = torch.nn.functional.interpolate(
            old_sub_table_41, size=(27, 27), mode="bilinear"
        )
        old_sub_table_41 = None
        permute_101 = new_sub_table_40.permute(0, 2, 3, 1)
        new_sub_table_40 = None
        new_sub_table_41 = permute_101.reshape(729, -1)
        permute_101 = None
        getitem_201 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(729, None, None)
        ]
        new_relative_position_bias_table_20 = torch.cat([new_sub_table_41, getitem_201])
        new_sub_table_41 = getitem_201 = None
        arange_40 = torch.arange(14)
        arange_41 = torch.arange(14)
        meshgrid_20 = torch.functional.meshgrid(arange_40, arange_41, indexing="ij")
        arange_40 = arange_41 = None
        getitem_202 = meshgrid_20[0]
        getitem_203 = meshgrid_20[1]
        meshgrid_20 = None
        coords_20 = torch.stack((getitem_202, getitem_203))
        getitem_202 = getitem_203 = None
        coords_flatten_20 = torch.flatten(coords_20, 1)
        coords_20 = None
        getitem_204 = coords_flatten_20[
            (slice(None, None, None), slice(None, None, None), None)
        ]
        getitem_205 = coords_flatten_20[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        coords_flatten_20 = None
        relative_coords_40 = getitem_204 - getitem_205
        getitem_204 = getitem_205 = None
        permute_102 = relative_coords_40.permute(1, 2, 0)
        relative_coords_40 = None
        relative_coords_41 = permute_102.contiguous()
        permute_102 = None
        getitem_206 = relative_coords_41[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_206 += 13
        iadd_40 = getitem_206
        getitem_206 = None
        relative_coords_41[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = iadd_40
        setitem_140 = relative_coords_41
        iadd_40 = setitem_140 = None
        getitem_207 = relative_coords_41[
            (slice(None, None, None), slice(None, None, None), 1)
        ]
        getitem_207 += 13
        iadd_41 = getitem_207
        getitem_207 = None
        relative_coords_41[
            (slice(None, None, None), slice(None, None, None), 1)
        ] = iadd_41
        setitem_141 = relative_coords_41
        iadd_41 = setitem_141 = None
        getitem_208 = relative_coords_41[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_208 *= 27
        imul_20 = getitem_208
        getitem_208 = None
        relative_coords_41[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = imul_20
        setitem_142 = relative_coords_41
        imul_20 = setitem_142 = None
        relative_position_index_20 = torch.zeros(size=(197, 197), dtype=torch.int64)
        sum_21 = relative_coords_41.sum(-1)
        relative_coords_41 = None
        relative_position_index_20[
            (slice(1, None, None), slice(1, None, None))
        ] = sum_21
        setitem_143 = relative_position_index_20
        sum_21 = setitem_143 = None
        relative_position_index_20[(0, slice(0, None, None))] = 729
        setitem_144 = relative_position_index_20
        setitem_144 = None
        relative_position_index_20[(slice(0, None, None), 0)] = 730
        setitem_145 = relative_position_index_20
        setitem_145 = None
        relative_position_index_20[(0, 0)] = 731
        setitem_146 = relative_position_index_20
        setitem_146 = None
        view_120 = relative_position_index_20.view(-1)
        relative_position_index_20 = None
        relative_position_bias_80 = new_relative_position_bias_table_20[view_120]
        new_relative_position_bias_table_20 = view_120 = None
        relative_position_bias_81 = relative_position_bias_80.view(197, 197, -1)
        relative_position_bias_80 = None
        permute_103 = relative_position_bias_81.permute(2, 0, 1)
        relative_position_bias_81 = None
        relative_position_bias_82 = permute_103.contiguous()
        permute_103 = None
        relative_position_bias_83 = relative_position_bias_82.unsqueeze(0)
        relative_position_bias_82 = None
        layer_norm_40 = torch.nn.functional.layer_norm(
            layer_output_59,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_before_parameters_bias_ = (None)
        linear_120 = torch._C._nn.linear(
            layer_norm_40,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_122 = linear_120.view(1, -1, 16, 64)
        linear_120 = None
        query_layer_20 = view_122.transpose(1, 2)
        view_122 = None
        linear_121 = torch._C._nn.linear(
            layer_norm_40,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_123 = linear_121.view(1, -1, 16, 64)
        linear_121 = None
        key_layer_20 = view_123.transpose(1, 2)
        view_123 = None
        linear_122 = torch._C._nn.linear(
            layer_norm_40,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        layer_norm_40 = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_124 = linear_122.view(1, -1, 16, 64)
        linear_122 = None
        value_layer_20 = view_124.transpose(1, 2)
        view_124 = None
        context_layer_60 = torch._C._nn.scaled_dot_product_attention(
            query_layer_20,
            key_layer_20,
            value_layer_20,
            attn_mask=relative_position_bias_83,
            dropout_p=0.0,
            is_causal=False,
            scale=0.125,
        )
        query_layer_20 = (
            key_layer_20
        ) = value_layer_20 = relative_position_bias_83 = None
        permute_104 = context_layer_60.permute(0, 2, 1, 3)
        context_layer_60 = None
        context_layer_61 = permute_104.contiguous()
        permute_104 = None
        context_layer_62 = context_layer_61.view(1, 197, 1024)
        context_layer_61 = None
        hidden_states_140 = torch._C._nn.linear(
            context_layer_62,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_62 = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_141 = torch.nn.functional.dropout(
            hidden_states_140, 0.0, False, False
        )
        hidden_states_140 = None
        attention_output_20 = (
            l_self_modules_encoder_modules_layer_modules_20_parameters_lambda_1_
            * hidden_states_141
        )
        l_self_modules_encoder_modules_layer_modules_20_parameters_lambda_1_ = (
            hidden_states_141
        ) = None
        hidden_states_142 = attention_output_20 + layer_output_59
        attention_output_20 = layer_output_59 = None
        layer_output_60 = torch.nn.functional.layer_norm(
            hidden_states_142,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_after_parameters_bias_ = (None)
        hidden_states_143 = torch._C._nn.linear(
            layer_output_60,
            l_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_60 = l_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_144 = torch._C._nn.gelu(hidden_states_143)
        hidden_states_143 = None
        hidden_states_145 = torch._C._nn.linear(
            hidden_states_144,
            l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_144 = l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_146 = torch.nn.functional.dropout(
            hidden_states_145, 0.0, False, False
        )
        hidden_states_145 = None
        layer_output_61 = (
            l_self_modules_encoder_modules_layer_modules_20_parameters_lambda_2_
            * hidden_states_146
        )
        l_self_modules_encoder_modules_layer_modules_20_parameters_lambda_2_ = (
            hidden_states_146
        ) = None
        layer_output_62 = layer_output_61 + hidden_states_142
        layer_output_61 = hidden_states_142 = None
        old_sub_table_42 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(None, 729, None)
        ]
        reshape_42 = old_sub_table_42.reshape(1, 27, 27, -1)
        old_sub_table_42 = None
        old_sub_table_43 = reshape_42.permute(0, 3, 1, 2)
        reshape_42 = None
        new_sub_table_42 = torch.nn.functional.interpolate(
            old_sub_table_43, size=(27, 27), mode="bilinear"
        )
        old_sub_table_43 = None
        permute_106 = new_sub_table_42.permute(0, 2, 3, 1)
        new_sub_table_42 = None
        new_sub_table_43 = permute_106.reshape(729, -1)
        permute_106 = None
        getitem_211 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(729, None, None)
        ]
        new_relative_position_bias_table_21 = torch.cat([new_sub_table_43, getitem_211])
        new_sub_table_43 = getitem_211 = None
        arange_42 = torch.arange(14)
        arange_43 = torch.arange(14)
        meshgrid_21 = torch.functional.meshgrid(arange_42, arange_43, indexing="ij")
        arange_42 = arange_43 = None
        getitem_212 = meshgrid_21[0]
        getitem_213 = meshgrid_21[1]
        meshgrid_21 = None
        coords_21 = torch.stack((getitem_212, getitem_213))
        getitem_212 = getitem_213 = None
        coords_flatten_21 = torch.flatten(coords_21, 1)
        coords_21 = None
        getitem_214 = coords_flatten_21[
            (slice(None, None, None), slice(None, None, None), None)
        ]
        getitem_215 = coords_flatten_21[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        coords_flatten_21 = None
        relative_coords_42 = getitem_214 - getitem_215
        getitem_214 = getitem_215 = None
        permute_107 = relative_coords_42.permute(1, 2, 0)
        relative_coords_42 = None
        relative_coords_43 = permute_107.contiguous()
        permute_107 = None
        getitem_216 = relative_coords_43[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_216 += 13
        iadd_42 = getitem_216
        getitem_216 = None
        relative_coords_43[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = iadd_42
        setitem_147 = relative_coords_43
        iadd_42 = setitem_147 = None
        getitem_217 = relative_coords_43[
            (slice(None, None, None), slice(None, None, None), 1)
        ]
        getitem_217 += 13
        iadd_43 = getitem_217
        getitem_217 = None
        relative_coords_43[
            (slice(None, None, None), slice(None, None, None), 1)
        ] = iadd_43
        setitem_148 = relative_coords_43
        iadd_43 = setitem_148 = None
        getitem_218 = relative_coords_43[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_218 *= 27
        imul_21 = getitem_218
        getitem_218 = None
        relative_coords_43[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = imul_21
        setitem_149 = relative_coords_43
        imul_21 = setitem_149 = None
        relative_position_index_21 = torch.zeros(size=(197, 197), dtype=torch.int64)
        sum_22 = relative_coords_43.sum(-1)
        relative_coords_43 = None
        relative_position_index_21[
            (slice(1, None, None), slice(1, None, None))
        ] = sum_22
        setitem_150 = relative_position_index_21
        sum_22 = setitem_150 = None
        relative_position_index_21[(0, slice(0, None, None))] = 729
        setitem_151 = relative_position_index_21
        setitem_151 = None
        relative_position_index_21[(slice(0, None, None), 0)] = 730
        setitem_152 = relative_position_index_21
        setitem_152 = None
        relative_position_index_21[(0, 0)] = 731
        setitem_153 = relative_position_index_21
        setitem_153 = None
        view_126 = relative_position_index_21.view(-1)
        relative_position_index_21 = None
        relative_position_bias_84 = new_relative_position_bias_table_21[view_126]
        new_relative_position_bias_table_21 = view_126 = None
        relative_position_bias_85 = relative_position_bias_84.view(197, 197, -1)
        relative_position_bias_84 = None
        permute_108 = relative_position_bias_85.permute(2, 0, 1)
        relative_position_bias_85 = None
        relative_position_bias_86 = permute_108.contiguous()
        permute_108 = None
        relative_position_bias_87 = relative_position_bias_86.unsqueeze(0)
        relative_position_bias_86 = None
        layer_norm_42 = torch.nn.functional.layer_norm(
            layer_output_62,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_before_parameters_bias_ = (None)
        linear_126 = torch._C._nn.linear(
            layer_norm_42,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_128 = linear_126.view(1, -1, 16, 64)
        linear_126 = None
        query_layer_21 = view_128.transpose(1, 2)
        view_128 = None
        linear_127 = torch._C._nn.linear(
            layer_norm_42,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_129 = linear_127.view(1, -1, 16, 64)
        linear_127 = None
        key_layer_21 = view_129.transpose(1, 2)
        view_129 = None
        linear_128 = torch._C._nn.linear(
            layer_norm_42,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        layer_norm_42 = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_130 = linear_128.view(1, -1, 16, 64)
        linear_128 = None
        value_layer_21 = view_130.transpose(1, 2)
        view_130 = None
        context_layer_63 = torch._C._nn.scaled_dot_product_attention(
            query_layer_21,
            key_layer_21,
            value_layer_21,
            attn_mask=relative_position_bias_87,
            dropout_p=0.0,
            is_causal=False,
            scale=0.125,
        )
        query_layer_21 = (
            key_layer_21
        ) = value_layer_21 = relative_position_bias_87 = None
        permute_109 = context_layer_63.permute(0, 2, 1, 3)
        context_layer_63 = None
        context_layer_64 = permute_109.contiguous()
        permute_109 = None
        context_layer_65 = context_layer_64.view(1, 197, 1024)
        context_layer_64 = None
        hidden_states_147 = torch._C._nn.linear(
            context_layer_65,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_65 = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_148 = torch.nn.functional.dropout(
            hidden_states_147, 0.0, False, False
        )
        hidden_states_147 = None
        attention_output_21 = (
            l_self_modules_encoder_modules_layer_modules_21_parameters_lambda_1_
            * hidden_states_148
        )
        l_self_modules_encoder_modules_layer_modules_21_parameters_lambda_1_ = (
            hidden_states_148
        ) = None
        hidden_states_149 = attention_output_21 + layer_output_62
        attention_output_21 = layer_output_62 = None
        layer_output_63 = torch.nn.functional.layer_norm(
            hidden_states_149,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_after_parameters_bias_ = (None)
        hidden_states_150 = torch._C._nn.linear(
            layer_output_63,
            l_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_63 = l_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_151 = torch._C._nn.gelu(hidden_states_150)
        hidden_states_150 = None
        hidden_states_152 = torch._C._nn.linear(
            hidden_states_151,
            l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_151 = l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_153 = torch.nn.functional.dropout(
            hidden_states_152, 0.0, False, False
        )
        hidden_states_152 = None
        layer_output_64 = (
            l_self_modules_encoder_modules_layer_modules_21_parameters_lambda_2_
            * hidden_states_153
        )
        l_self_modules_encoder_modules_layer_modules_21_parameters_lambda_2_ = (
            hidden_states_153
        ) = None
        layer_output_65 = layer_output_64 + hidden_states_149
        layer_output_64 = hidden_states_149 = None
        old_sub_table_44 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(None, 729, None)
        ]
        reshape_44 = old_sub_table_44.reshape(1, 27, 27, -1)
        old_sub_table_44 = None
        old_sub_table_45 = reshape_44.permute(0, 3, 1, 2)
        reshape_44 = None
        new_sub_table_44 = torch.nn.functional.interpolate(
            old_sub_table_45, size=(27, 27), mode="bilinear"
        )
        old_sub_table_45 = None
        permute_111 = new_sub_table_44.permute(0, 2, 3, 1)
        new_sub_table_44 = None
        new_sub_table_45 = permute_111.reshape(729, -1)
        permute_111 = None
        getitem_221 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(729, None, None)
        ]
        new_relative_position_bias_table_22 = torch.cat([new_sub_table_45, getitem_221])
        new_sub_table_45 = getitem_221 = None
        arange_44 = torch.arange(14)
        arange_45 = torch.arange(14)
        meshgrid_22 = torch.functional.meshgrid(arange_44, arange_45, indexing="ij")
        arange_44 = arange_45 = None
        getitem_222 = meshgrid_22[0]
        getitem_223 = meshgrid_22[1]
        meshgrid_22 = None
        coords_22 = torch.stack((getitem_222, getitem_223))
        getitem_222 = getitem_223 = None
        coords_flatten_22 = torch.flatten(coords_22, 1)
        coords_22 = None
        getitem_224 = coords_flatten_22[
            (slice(None, None, None), slice(None, None, None), None)
        ]
        getitem_225 = coords_flatten_22[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        coords_flatten_22 = None
        relative_coords_44 = getitem_224 - getitem_225
        getitem_224 = getitem_225 = None
        permute_112 = relative_coords_44.permute(1, 2, 0)
        relative_coords_44 = None
        relative_coords_45 = permute_112.contiguous()
        permute_112 = None
        getitem_226 = relative_coords_45[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_226 += 13
        iadd_44 = getitem_226
        getitem_226 = None
        relative_coords_45[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = iadd_44
        setitem_154 = relative_coords_45
        iadd_44 = setitem_154 = None
        getitem_227 = relative_coords_45[
            (slice(None, None, None), slice(None, None, None), 1)
        ]
        getitem_227 += 13
        iadd_45 = getitem_227
        getitem_227 = None
        relative_coords_45[
            (slice(None, None, None), slice(None, None, None), 1)
        ] = iadd_45
        setitem_155 = relative_coords_45
        iadd_45 = setitem_155 = None
        getitem_228 = relative_coords_45[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_228 *= 27
        imul_22 = getitem_228
        getitem_228 = None
        relative_coords_45[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = imul_22
        setitem_156 = relative_coords_45
        imul_22 = setitem_156 = None
        relative_position_index_22 = torch.zeros(size=(197, 197), dtype=torch.int64)
        sum_23 = relative_coords_45.sum(-1)
        relative_coords_45 = None
        relative_position_index_22[
            (slice(1, None, None), slice(1, None, None))
        ] = sum_23
        setitem_157 = relative_position_index_22
        sum_23 = setitem_157 = None
        relative_position_index_22[(0, slice(0, None, None))] = 729
        setitem_158 = relative_position_index_22
        setitem_158 = None
        relative_position_index_22[(slice(0, None, None), 0)] = 730
        setitem_159 = relative_position_index_22
        setitem_159 = None
        relative_position_index_22[(0, 0)] = 731
        setitem_160 = relative_position_index_22
        setitem_160 = None
        view_132 = relative_position_index_22.view(-1)
        relative_position_index_22 = None
        relative_position_bias_88 = new_relative_position_bias_table_22[view_132]
        new_relative_position_bias_table_22 = view_132 = None
        relative_position_bias_89 = relative_position_bias_88.view(197, 197, -1)
        relative_position_bias_88 = None
        permute_113 = relative_position_bias_89.permute(2, 0, 1)
        relative_position_bias_89 = None
        relative_position_bias_90 = permute_113.contiguous()
        permute_113 = None
        relative_position_bias_91 = relative_position_bias_90.unsqueeze(0)
        relative_position_bias_90 = None
        layer_norm_44 = torch.nn.functional.layer_norm(
            layer_output_65,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_before_parameters_bias_ = (None)
        linear_132 = torch._C._nn.linear(
            layer_norm_44,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_134 = linear_132.view(1, -1, 16, 64)
        linear_132 = None
        query_layer_22 = view_134.transpose(1, 2)
        view_134 = None
        linear_133 = torch._C._nn.linear(
            layer_norm_44,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_135 = linear_133.view(1, -1, 16, 64)
        linear_133 = None
        key_layer_22 = view_135.transpose(1, 2)
        view_135 = None
        linear_134 = torch._C._nn.linear(
            layer_norm_44,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        layer_norm_44 = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_136 = linear_134.view(1, -1, 16, 64)
        linear_134 = None
        value_layer_22 = view_136.transpose(1, 2)
        view_136 = None
        context_layer_66 = torch._C._nn.scaled_dot_product_attention(
            query_layer_22,
            key_layer_22,
            value_layer_22,
            attn_mask=relative_position_bias_91,
            dropout_p=0.0,
            is_causal=False,
            scale=0.125,
        )
        query_layer_22 = (
            key_layer_22
        ) = value_layer_22 = relative_position_bias_91 = None
        permute_114 = context_layer_66.permute(0, 2, 1, 3)
        context_layer_66 = None
        context_layer_67 = permute_114.contiguous()
        permute_114 = None
        context_layer_68 = context_layer_67.view(1, 197, 1024)
        context_layer_67 = None
        hidden_states_154 = torch._C._nn.linear(
            context_layer_68,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_68 = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_155 = torch.nn.functional.dropout(
            hidden_states_154, 0.0, False, False
        )
        hidden_states_154 = None
        attention_output_22 = (
            l_self_modules_encoder_modules_layer_modules_22_parameters_lambda_1_
            * hidden_states_155
        )
        l_self_modules_encoder_modules_layer_modules_22_parameters_lambda_1_ = (
            hidden_states_155
        ) = None
        hidden_states_156 = attention_output_22 + layer_output_65
        attention_output_22 = layer_output_65 = None
        layer_output_66 = torch.nn.functional.layer_norm(
            hidden_states_156,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_after_parameters_bias_ = (None)
        hidden_states_157 = torch._C._nn.linear(
            layer_output_66,
            l_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_66 = l_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_158 = torch._C._nn.gelu(hidden_states_157)
        hidden_states_157 = None
        hidden_states_159 = torch._C._nn.linear(
            hidden_states_158,
            l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_158 = l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_160 = torch.nn.functional.dropout(
            hidden_states_159, 0.0, False, False
        )
        hidden_states_159 = None
        layer_output_67 = (
            l_self_modules_encoder_modules_layer_modules_22_parameters_lambda_2_
            * hidden_states_160
        )
        l_self_modules_encoder_modules_layer_modules_22_parameters_lambda_2_ = (
            hidden_states_160
        ) = None
        layer_output_68 = layer_output_67 + hidden_states_156
        layer_output_67 = hidden_states_156 = None
        old_sub_table_46 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(None, 729, None)
        ]
        reshape_46 = old_sub_table_46.reshape(1, 27, 27, -1)
        old_sub_table_46 = None
        old_sub_table_47 = reshape_46.permute(0, 3, 1, 2)
        reshape_46 = None
        new_sub_table_46 = torch.nn.functional.interpolate(
            old_sub_table_47, size=(27, 27), mode="bilinear"
        )
        old_sub_table_47 = None
        permute_116 = new_sub_table_46.permute(0, 2, 3, 1)
        new_sub_table_46 = None
        new_sub_table_47 = permute_116.reshape(729, -1)
        permute_116 = None
        getitem_231 = l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_[
            slice(729, None, None)
        ]
        l_self_modules_encoder_modules_relative_position_bias_parameters_relative_position_bias_table_ = (
            None
        )
        new_relative_position_bias_table_23 = torch.cat([new_sub_table_47, getitem_231])
        new_sub_table_47 = getitem_231 = None
        arange_46 = torch.arange(14)
        arange_47 = torch.arange(14)
        meshgrid_23 = torch.functional.meshgrid(arange_46, arange_47, indexing="ij")
        arange_46 = arange_47 = None
        getitem_232 = meshgrid_23[0]
        getitem_233 = meshgrid_23[1]
        meshgrid_23 = None
        coords_23 = torch.stack((getitem_232, getitem_233))
        getitem_232 = getitem_233 = None
        coords_flatten_23 = torch.flatten(coords_23, 1)
        coords_23 = None
        getitem_234 = coords_flatten_23[
            (slice(None, None, None), slice(None, None, None), None)
        ]
        getitem_235 = coords_flatten_23[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        coords_flatten_23 = None
        relative_coords_46 = getitem_234 - getitem_235
        getitem_234 = getitem_235 = None
        permute_117 = relative_coords_46.permute(1, 2, 0)
        relative_coords_46 = None
        relative_coords_47 = permute_117.contiguous()
        permute_117 = None
        getitem_236 = relative_coords_47[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_236 += 13
        iadd_46 = getitem_236
        getitem_236 = None
        relative_coords_47[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = iadd_46
        setitem_161 = relative_coords_47
        iadd_46 = setitem_161 = None
        getitem_237 = relative_coords_47[
            (slice(None, None, None), slice(None, None, None), 1)
        ]
        getitem_237 += 13
        iadd_47 = getitem_237
        getitem_237 = None
        relative_coords_47[
            (slice(None, None, None), slice(None, None, None), 1)
        ] = iadd_47
        setitem_162 = relative_coords_47
        iadd_47 = setitem_162 = None
        getitem_238 = relative_coords_47[
            (slice(None, None, None), slice(None, None, None), 0)
        ]
        getitem_238 *= 27
        imul_23 = getitem_238
        getitem_238 = None
        relative_coords_47[
            (slice(None, None, None), slice(None, None, None), 0)
        ] = imul_23
        setitem_163 = relative_coords_47
        imul_23 = setitem_163 = None
        relative_position_index_23 = torch.zeros(size=(197, 197), dtype=torch.int64)
        sum_24 = relative_coords_47.sum(-1)
        relative_coords_47 = None
        relative_position_index_23[
            (slice(1, None, None), slice(1, None, None))
        ] = sum_24
        setitem_164 = relative_position_index_23
        sum_24 = setitem_164 = None
        relative_position_index_23[(0, slice(0, None, None))] = 729
        setitem_165 = relative_position_index_23
        setitem_165 = None
        relative_position_index_23[(slice(0, None, None), 0)] = 730
        setitem_166 = relative_position_index_23
        setitem_166 = None
        relative_position_index_23[(0, 0)] = 731
        setitem_167 = relative_position_index_23
        setitem_167 = None
        view_138 = relative_position_index_23.view(-1)
        relative_position_index_23 = None
        relative_position_bias_92 = new_relative_position_bias_table_23[view_138]
        new_relative_position_bias_table_23 = view_138 = None
        relative_position_bias_93 = relative_position_bias_92.view(197, 197, -1)
        relative_position_bias_92 = None
        permute_118 = relative_position_bias_93.permute(2, 0, 1)
        relative_position_bias_93 = None
        relative_position_bias_94 = permute_118.contiguous()
        permute_118 = None
        relative_position_bias_95 = relative_position_bias_94.unsqueeze(0)
        relative_position_bias_94 = None
        layer_norm_46 = torch.nn.functional.layer_norm(
            layer_output_68,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_before_parameters_bias_ = (None)
        linear_138 = torch._C._nn.linear(
            layer_norm_46,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_140 = linear_138.view(1, -1, 16, 64)
        linear_138 = None
        query_layer_23 = view_140.transpose(1, 2)
        view_140 = None
        linear_139 = torch._C._nn.linear(
            layer_norm_46,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_141 = linear_139.view(1, -1, 16, 64)
        linear_139 = None
        key_layer_23 = view_141.transpose(1, 2)
        view_141 = None
        linear_140 = torch._C._nn.linear(
            layer_norm_46,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        layer_norm_46 = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_142 = linear_140.view(1, -1, 16, 64)
        linear_140 = None
        value_layer_23 = view_142.transpose(1, 2)
        view_142 = None
        context_layer_69 = torch._C._nn.scaled_dot_product_attention(
            query_layer_23,
            key_layer_23,
            value_layer_23,
            attn_mask=relative_position_bias_95,
            dropout_p=0.0,
            is_causal=False,
            scale=0.125,
        )
        query_layer_23 = (
            key_layer_23
        ) = value_layer_23 = relative_position_bias_95 = None
        permute_119 = context_layer_69.permute(0, 2, 1, 3)
        context_layer_69 = None
        context_layer_70 = permute_119.contiguous()
        permute_119 = None
        context_layer_71 = context_layer_70.view(1, 197, 1024)
        context_layer_70 = None
        hidden_states_161 = torch._C._nn.linear(
            context_layer_71,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_71 = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_162 = torch.nn.functional.dropout(
            hidden_states_161, 0.0, False, False
        )
        hidden_states_161 = None
        attention_output_23 = (
            l_self_modules_encoder_modules_layer_modules_23_parameters_lambda_1_
            * hidden_states_162
        )
        l_self_modules_encoder_modules_layer_modules_23_parameters_lambda_1_ = (
            hidden_states_162
        ) = None
        hidden_states_163 = attention_output_23 + layer_output_68
        attention_output_23 = layer_output_68 = None
        layer_output_69 = torch.nn.functional.layer_norm(
            hidden_states_163,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_after_parameters_bias_ = (None)
        hidden_states_164 = torch._C._nn.linear(
            layer_output_69,
            l_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_69 = l_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_165 = torch._C._nn.gelu(hidden_states_164)
        hidden_states_164 = None
        hidden_states_166 = torch._C._nn.linear(
            hidden_states_165,
            l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_165 = l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_167 = torch.nn.functional.dropout(
            hidden_states_166, 0.0, False, False
        )
        hidden_states_166 = None
        layer_output_70 = (
            l_self_modules_encoder_modules_layer_modules_23_parameters_lambda_2_
            * hidden_states_167
        )
        l_self_modules_encoder_modules_layer_modules_23_parameters_lambda_2_ = (
            hidden_states_167
        ) = None
        layer_output_71 = layer_output_70 + hidden_states_163
        layer_output_70 = hidden_states_163 = None
        patch_tokens = layer_output_71[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        mean = patch_tokens.mean(1)
        patch_tokens = None
        pooled_output = torch.nn.functional.layer_norm(
            mean,
            (1024,),
            l_self_modules_pooler_modules_layernorm_parameters_weight_,
            l_self_modules_pooler_modules_layernorm_parameters_bias_,
            1e-12,
        )
        mean = (
            l_self_modules_pooler_modules_layernorm_parameters_weight_
        ) = l_self_modules_pooler_modules_layernorm_parameters_bias_ = None
        return (layer_output_71, pooled_output)
