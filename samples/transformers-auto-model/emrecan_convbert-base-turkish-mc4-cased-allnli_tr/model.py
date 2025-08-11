import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_attention_mask_: torch.Tensor,
        L_token_type_ids_: torch.Tensor,
        L_self_modules_embeddings_modules_word_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_buffers_position_ids_: torch.Tensor,
        L_self_modules_embeddings_modules_position_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_input_ids_ = L_input_ids_
        l_attention_mask_ = L_attention_mask_
        l_token_type_ids_ = L_token_type_ids_
        l_self_modules_embeddings_modules_word_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_word_embeddings_parameters_weight_
        )
        l_self_modules_embeddings_buffers_position_ids_ = (
            L_self_modules_embeddings_buffers_position_ids_
        )
        l_self_modules_embeddings_modules_position_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_position_embeddings_parameters_weight_
        )
        l_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_
        )
        l_self_modules_embeddings_modules_layer_norm_parameters_weight_ = (
            L_self_modules_embeddings_modules_LayerNorm_parameters_weight_
        )
        l_self_modules_embeddings_modules_layer_norm_parameters_bias_ = (
            L_self_modules_embeddings_modules_LayerNorm_parameters_bias_
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_bias_
        extended_attention_mask = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        l_attention_mask_ = None
        extended_attention_mask_1 = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = None
        sub = 1.0 - extended_attention_mask_1
        extended_attention_mask_1 = None
        extended_attention_mask_2 = sub * -3.4028234663852886e38
        sub = None
        position_ids = l_self_modules_embeddings_buffers_position_ids_[
            (slice(None, None, None), slice(None, 23, None))
        ]
        l_self_modules_embeddings_buffers_position_ids_ = None
        inputs_embeds = torch.nn.functional.embedding(
            l_input_ids_,
            l_self_modules_embeddings_modules_word_embeddings_parameters_weight_,
            0,
            None,
            2.0,
            False,
            False,
        )
        l_input_ids_ = (
            l_self_modules_embeddings_modules_word_embeddings_parameters_weight_
        ) = None
        position_embeddings = torch.nn.functional.embedding(
            position_ids,
            l_self_modules_embeddings_modules_position_embeddings_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        position_ids = (
            l_self_modules_embeddings_modules_position_embeddings_parameters_weight_
        ) = None
        token_type_embeddings = torch.nn.functional.embedding(
            l_token_type_ids_,
            l_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        l_token_type_ids_ = (
            l_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_
        ) = None
        add = inputs_embeds + position_embeddings
        inputs_embeds = position_embeddings = None
        embeddings = add + token_type_embeddings
        add = token_type_embeddings = None
        embeddings_1 = torch.nn.functional.layer_norm(
            embeddings,
            (768,),
            l_self_modules_embeddings_modules_layer_norm_parameters_weight_,
            l_self_modules_embeddings_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        embeddings = (
            l_self_modules_embeddings_modules_layer_norm_parameters_weight_
        ) = l_self_modules_embeddings_modules_layer_norm_parameters_bias_ = None
        embeddings_2 = torch.nn.functional.dropout(embeddings_1, 0.1, False, False)
        embeddings_1 = None
        mixed_key_layer = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        mixed_value_layer = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        transpose = embeddings_2.transpose(1, 2)
        x = torch.conv1d(
            transpose,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_,
            None,
            (1,),
            (4,),
            (1,),
            768,
        )
        transpose = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_ = (None)
        x_1 = torch.conv1d(
            x,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_,
            None,
            (1,),
            (0,),
            (1,),
            1,
        )
        x = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_ = (None)
        x_1 += l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_
        x_2 = x_1
        x_1 = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_ = (None)
        mixed_key_conv_attn_layer = x_2.transpose(1, 2)
        x_2 = None
        mixed_query_layer = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view = mixed_query_layer.view(1, -1, 6, 64)
        query_layer = view.transpose(1, 2)
        view = None
        view_1 = mixed_key_layer.view(1, -1, 6, 64)
        mixed_key_layer = None
        key_layer = view_1.transpose(1, 2)
        view_1 = None
        view_2 = mixed_value_layer.view(1, -1, 6, 64)
        mixed_value_layer = None
        value_layer = view_2.transpose(1, 2)
        view_2 = None
        conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mixed_key_conv_attn_layer = mixed_query_layer = None
        conv_kernel_layer = torch._C._nn.linear(
            conv_attn_layer,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_,
        )
        conv_attn_layer = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_ = (None)
        conv_kernel_layer_1 = torch.reshape(conv_kernel_layer, [-1, 9, 1])
        conv_kernel_layer = None
        conv_kernel_layer_2 = torch.softmax(conv_kernel_layer_1, dim=1)
        conv_kernel_layer_1 = None
        conv_out_layer = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_ = (None)
        conv_out_layer_1 = torch.reshape(conv_out_layer, [1, -1, 384])
        conv_out_layer = None
        transpose_5 = conv_out_layer_1.transpose(1, 2)
        conv_out_layer_1 = None
        contiguous = transpose_5.contiguous()
        transpose_5 = None
        conv_out_layer_2 = contiguous.unsqueeze(-1)
        contiguous = None
        conv_out_layer_3 = torch.nn.functional.unfold(
            conv_out_layer_2, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1
        )
        conv_out_layer_2 = None
        transpose_6 = conv_out_layer_3.transpose(1, 2)
        conv_out_layer_3 = None
        conv_out_layer_4 = transpose_6.reshape(1, -1, 384, 9)
        transpose_6 = None
        conv_out_layer_5 = torch.reshape(conv_out_layer_4, [-1, 64, 9])
        conv_out_layer_4 = None
        conv_out_layer_6 = torch.matmul(conv_out_layer_5, conv_kernel_layer_2)
        conv_out_layer_5 = conv_kernel_layer_2 = None
        conv_out_layer_7 = torch.reshape(conv_out_layer_6, [-1, 384])
        conv_out_layer_6 = None
        transpose_7 = key_layer.transpose(-1, -2)
        key_layer = None
        attention_scores = torch.matmul(query_layer, transpose_7)
        query_layer = transpose_7 = None
        attention_scores_1 = attention_scores / 8.0
        attention_scores = None
        attention_scores_2 = attention_scores_1 + extended_attention_mask_2
        attention_scores_1 = None
        attention_probs = torch.nn.functional.softmax(attention_scores_2, dim=-1)
        attention_scores_2 = None
        attention_probs_1 = torch.nn.functional.dropout(
            attention_probs, 0.1, False, False
        )
        attention_probs = None
        context_layer = torch.matmul(attention_probs_1, value_layer)
        attention_probs_1 = value_layer = None
        permute = context_layer.permute(0, 2, 1, 3)
        context_layer = None
        context_layer_1 = permute.contiguous()
        permute = None
        conv_out = torch.reshape(conv_out_layer_7, [1, -1, 6, 64])
        conv_out_layer_7 = None
        context_layer_2 = torch.cat([context_layer_1, conv_out], 2)
        context_layer_1 = conv_out = None
        context_layer_3 = context_layer_2.view(1, 23, 768)
        context_layer_2 = None
        hidden_states = torch._C._nn.linear(
            context_layer_3,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_3 = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_1 = torch.nn.functional.dropout(hidden_states, 0.1, False, False)
        hidden_states = None
        add_3 = hidden_states_1 + embeddings_2
        hidden_states_1 = embeddings_2 = None
        hidden_states_2 = torch.nn.functional.layer_norm(
            add_3,
            (768,),
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_3 = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_3 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_4 = torch._C._nn.gelu(hidden_states_3)
        hidden_states_3 = None
        hidden_states_5 = torch._C._nn.linear(
            hidden_states_4,
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_4 = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_6 = torch.nn.functional.dropout(
            hidden_states_5, 0.1, False, False
        )
        hidden_states_5 = None
        add_4 = hidden_states_6 + hidden_states_2
        hidden_states_6 = hidden_states_2 = None
        hidden_states_7 = torch.nn.functional.layer_norm(
            add_4,
            (768,),
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_4 = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_bias_ = (None)
        mixed_key_layer_1 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        mixed_value_layer_1 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        transpose_8 = hidden_states_7.transpose(1, 2)
        x_3 = torch.conv1d(
            transpose_8,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_,
            None,
            (1,),
            (4,),
            (1,),
            768,
        )
        transpose_8 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_ = (None)
        x_4 = torch.conv1d(
            x_3,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_,
            None,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_3 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_ = (None)
        x_4 += l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_
        x_5 = x_4
        x_4 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_ = (None)
        mixed_key_conv_attn_layer_1 = x_5.transpose(1, 2)
        x_5 = None
        mixed_query_layer_1 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_4 = mixed_query_layer_1.view(1, -1, 6, 64)
        query_layer_1 = view_4.transpose(1, 2)
        view_4 = None
        view_5 = mixed_key_layer_1.view(1, -1, 6, 64)
        mixed_key_layer_1 = None
        key_layer_1 = view_5.transpose(1, 2)
        view_5 = None
        view_6 = mixed_value_layer_1.view(1, -1, 6, 64)
        mixed_value_layer_1 = None
        value_layer_1 = view_6.transpose(1, 2)
        view_6 = None
        conv_attn_layer_1 = torch.multiply(
            mixed_key_conv_attn_layer_1, mixed_query_layer_1
        )
        mixed_key_conv_attn_layer_1 = mixed_query_layer_1 = None
        conv_kernel_layer_3 = torch._C._nn.linear(
            conv_attn_layer_1,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_,
        )
        conv_attn_layer_1 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_ = (None)
        conv_kernel_layer_4 = torch.reshape(conv_kernel_layer_3, [-1, 9, 1])
        conv_kernel_layer_3 = None
        conv_kernel_layer_5 = torch.softmax(conv_kernel_layer_4, dim=1)
        conv_kernel_layer_4 = None
        conv_out_layer_8 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_ = (None)
        conv_out_layer_9 = torch.reshape(conv_out_layer_8, [1, -1, 384])
        conv_out_layer_8 = None
        transpose_13 = conv_out_layer_9.transpose(1, 2)
        conv_out_layer_9 = None
        contiguous_2 = transpose_13.contiguous()
        transpose_13 = None
        conv_out_layer_10 = contiguous_2.unsqueeze(-1)
        contiguous_2 = None
        conv_out_layer_11 = torch.nn.functional.unfold(
            conv_out_layer_10, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1
        )
        conv_out_layer_10 = None
        transpose_14 = conv_out_layer_11.transpose(1, 2)
        conv_out_layer_11 = None
        conv_out_layer_12 = transpose_14.reshape(1, -1, 384, 9)
        transpose_14 = None
        conv_out_layer_13 = torch.reshape(conv_out_layer_12, [-1, 64, 9])
        conv_out_layer_12 = None
        conv_out_layer_14 = torch.matmul(conv_out_layer_13, conv_kernel_layer_5)
        conv_out_layer_13 = conv_kernel_layer_5 = None
        conv_out_layer_15 = torch.reshape(conv_out_layer_14, [-1, 384])
        conv_out_layer_14 = None
        transpose_15 = key_layer_1.transpose(-1, -2)
        key_layer_1 = None
        attention_scores_3 = torch.matmul(query_layer_1, transpose_15)
        query_layer_1 = transpose_15 = None
        attention_scores_4 = attention_scores_3 / 8.0
        attention_scores_3 = None
        attention_scores_5 = attention_scores_4 + extended_attention_mask_2
        attention_scores_4 = None
        attention_probs_2 = torch.nn.functional.softmax(attention_scores_5, dim=-1)
        attention_scores_5 = None
        attention_probs_3 = torch.nn.functional.dropout(
            attention_probs_2, 0.1, False, False
        )
        attention_probs_2 = None
        context_layer_4 = torch.matmul(attention_probs_3, value_layer_1)
        attention_probs_3 = value_layer_1 = None
        permute_1 = context_layer_4.permute(0, 2, 1, 3)
        context_layer_4 = None
        context_layer_5 = permute_1.contiguous()
        permute_1 = None
        conv_out_1 = torch.reshape(conv_out_layer_15, [1, -1, 6, 64])
        conv_out_layer_15 = None
        context_layer_6 = torch.cat([context_layer_5, conv_out_1], 2)
        context_layer_5 = conv_out_1 = None
        context_layer_7 = context_layer_6.view(1, 23, 768)
        context_layer_6 = None
        hidden_states_8 = torch._C._nn.linear(
            context_layer_7,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_7 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_9 = torch.nn.functional.dropout(
            hidden_states_8, 0.1, False, False
        )
        hidden_states_8 = None
        add_6 = hidden_states_9 + hidden_states_7
        hidden_states_9 = hidden_states_7 = None
        hidden_states_10 = torch.nn.functional.layer_norm(
            add_6,
            (768,),
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_6 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_11 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_12 = torch._C._nn.gelu(hidden_states_11)
        hidden_states_11 = None
        hidden_states_13 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_12 = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_14 = torch.nn.functional.dropout(
            hidden_states_13, 0.1, False, False
        )
        hidden_states_13 = None
        add_7 = hidden_states_14 + hidden_states_10
        hidden_states_14 = hidden_states_10 = None
        hidden_states_15 = torch.nn.functional.layer_norm(
            add_7,
            (768,),
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_7 = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_bias_ = (None)
        mixed_key_layer_2 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        mixed_value_layer_2 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        transpose_16 = hidden_states_15.transpose(1, 2)
        x_6 = torch.conv1d(
            transpose_16,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_,
            None,
            (1,),
            (4,),
            (1,),
            768,
        )
        transpose_16 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_ = (None)
        x_7 = torch.conv1d(
            x_6,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_,
            None,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_6 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_ = (None)
        x_7 += l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_
        x_8 = x_7
        x_7 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_ = (None)
        mixed_key_conv_attn_layer_2 = x_8.transpose(1, 2)
        x_8 = None
        mixed_query_layer_2 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_8 = mixed_query_layer_2.view(1, -1, 6, 64)
        query_layer_2 = view_8.transpose(1, 2)
        view_8 = None
        view_9 = mixed_key_layer_2.view(1, -1, 6, 64)
        mixed_key_layer_2 = None
        key_layer_2 = view_9.transpose(1, 2)
        view_9 = None
        view_10 = mixed_value_layer_2.view(1, -1, 6, 64)
        mixed_value_layer_2 = None
        value_layer_2 = view_10.transpose(1, 2)
        view_10 = None
        conv_attn_layer_2 = torch.multiply(
            mixed_key_conv_attn_layer_2, mixed_query_layer_2
        )
        mixed_key_conv_attn_layer_2 = mixed_query_layer_2 = None
        conv_kernel_layer_6 = torch._C._nn.linear(
            conv_attn_layer_2,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_,
        )
        conv_attn_layer_2 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_ = (None)
        conv_kernel_layer_7 = torch.reshape(conv_kernel_layer_6, [-1, 9, 1])
        conv_kernel_layer_6 = None
        conv_kernel_layer_8 = torch.softmax(conv_kernel_layer_7, dim=1)
        conv_kernel_layer_7 = None
        conv_out_layer_16 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_ = (None)
        conv_out_layer_17 = torch.reshape(conv_out_layer_16, [1, -1, 384])
        conv_out_layer_16 = None
        transpose_21 = conv_out_layer_17.transpose(1, 2)
        conv_out_layer_17 = None
        contiguous_4 = transpose_21.contiguous()
        transpose_21 = None
        conv_out_layer_18 = contiguous_4.unsqueeze(-1)
        contiguous_4 = None
        conv_out_layer_19 = torch.nn.functional.unfold(
            conv_out_layer_18, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1
        )
        conv_out_layer_18 = None
        transpose_22 = conv_out_layer_19.transpose(1, 2)
        conv_out_layer_19 = None
        conv_out_layer_20 = transpose_22.reshape(1, -1, 384, 9)
        transpose_22 = None
        conv_out_layer_21 = torch.reshape(conv_out_layer_20, [-1, 64, 9])
        conv_out_layer_20 = None
        conv_out_layer_22 = torch.matmul(conv_out_layer_21, conv_kernel_layer_8)
        conv_out_layer_21 = conv_kernel_layer_8 = None
        conv_out_layer_23 = torch.reshape(conv_out_layer_22, [-1, 384])
        conv_out_layer_22 = None
        transpose_23 = key_layer_2.transpose(-1, -2)
        key_layer_2 = None
        attention_scores_6 = torch.matmul(query_layer_2, transpose_23)
        query_layer_2 = transpose_23 = None
        attention_scores_7 = attention_scores_6 / 8.0
        attention_scores_6 = None
        attention_scores_8 = attention_scores_7 + extended_attention_mask_2
        attention_scores_7 = None
        attention_probs_4 = torch.nn.functional.softmax(attention_scores_8, dim=-1)
        attention_scores_8 = None
        attention_probs_5 = torch.nn.functional.dropout(
            attention_probs_4, 0.1, False, False
        )
        attention_probs_4 = None
        context_layer_8 = torch.matmul(attention_probs_5, value_layer_2)
        attention_probs_5 = value_layer_2 = None
        permute_2 = context_layer_8.permute(0, 2, 1, 3)
        context_layer_8 = None
        context_layer_9 = permute_2.contiguous()
        permute_2 = None
        conv_out_2 = torch.reshape(conv_out_layer_23, [1, -1, 6, 64])
        conv_out_layer_23 = None
        context_layer_10 = torch.cat([context_layer_9, conv_out_2], 2)
        context_layer_9 = conv_out_2 = None
        context_layer_11 = context_layer_10.view(1, 23, 768)
        context_layer_10 = None
        hidden_states_16 = torch._C._nn.linear(
            context_layer_11,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_11 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_17 = torch.nn.functional.dropout(
            hidden_states_16, 0.1, False, False
        )
        hidden_states_16 = None
        add_9 = hidden_states_17 + hidden_states_15
        hidden_states_17 = hidden_states_15 = None
        hidden_states_18 = torch.nn.functional.layer_norm(
            add_9,
            (768,),
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_9 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_19 = torch._C._nn.linear(
            hidden_states_18,
            l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_20 = torch._C._nn.gelu(hidden_states_19)
        hidden_states_19 = None
        hidden_states_21 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_20 = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_22 = torch.nn.functional.dropout(
            hidden_states_21, 0.1, False, False
        )
        hidden_states_21 = None
        add_10 = hidden_states_22 + hidden_states_18
        hidden_states_22 = hidden_states_18 = None
        hidden_states_23 = torch.nn.functional.layer_norm(
            add_10,
            (768,),
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_10 = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_bias_ = (None)
        mixed_key_layer_3 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        mixed_value_layer_3 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        transpose_24 = hidden_states_23.transpose(1, 2)
        x_9 = torch.conv1d(
            transpose_24,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_,
            None,
            (1,),
            (4,),
            (1,),
            768,
        )
        transpose_24 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_ = (None)
        x_10 = torch.conv1d(
            x_9,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_,
            None,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_9 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_ = (None)
        x_10 += l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_
        x_11 = x_10
        x_10 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_ = (None)
        mixed_key_conv_attn_layer_3 = x_11.transpose(1, 2)
        x_11 = None
        mixed_query_layer_3 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_12 = mixed_query_layer_3.view(1, -1, 6, 64)
        query_layer_3 = view_12.transpose(1, 2)
        view_12 = None
        view_13 = mixed_key_layer_3.view(1, -1, 6, 64)
        mixed_key_layer_3 = None
        key_layer_3 = view_13.transpose(1, 2)
        view_13 = None
        view_14 = mixed_value_layer_3.view(1, -1, 6, 64)
        mixed_value_layer_3 = None
        value_layer_3 = view_14.transpose(1, 2)
        view_14 = None
        conv_attn_layer_3 = torch.multiply(
            mixed_key_conv_attn_layer_3, mixed_query_layer_3
        )
        mixed_key_conv_attn_layer_3 = mixed_query_layer_3 = None
        conv_kernel_layer_9 = torch._C._nn.linear(
            conv_attn_layer_3,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_,
        )
        conv_attn_layer_3 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_ = (None)
        conv_kernel_layer_10 = torch.reshape(conv_kernel_layer_9, [-1, 9, 1])
        conv_kernel_layer_9 = None
        conv_kernel_layer_11 = torch.softmax(conv_kernel_layer_10, dim=1)
        conv_kernel_layer_10 = None
        conv_out_layer_24 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_ = (None)
        conv_out_layer_25 = torch.reshape(conv_out_layer_24, [1, -1, 384])
        conv_out_layer_24 = None
        transpose_29 = conv_out_layer_25.transpose(1, 2)
        conv_out_layer_25 = None
        contiguous_6 = transpose_29.contiguous()
        transpose_29 = None
        conv_out_layer_26 = contiguous_6.unsqueeze(-1)
        contiguous_6 = None
        conv_out_layer_27 = torch.nn.functional.unfold(
            conv_out_layer_26, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1
        )
        conv_out_layer_26 = None
        transpose_30 = conv_out_layer_27.transpose(1, 2)
        conv_out_layer_27 = None
        conv_out_layer_28 = transpose_30.reshape(1, -1, 384, 9)
        transpose_30 = None
        conv_out_layer_29 = torch.reshape(conv_out_layer_28, [-1, 64, 9])
        conv_out_layer_28 = None
        conv_out_layer_30 = torch.matmul(conv_out_layer_29, conv_kernel_layer_11)
        conv_out_layer_29 = conv_kernel_layer_11 = None
        conv_out_layer_31 = torch.reshape(conv_out_layer_30, [-1, 384])
        conv_out_layer_30 = None
        transpose_31 = key_layer_3.transpose(-1, -2)
        key_layer_3 = None
        attention_scores_9 = torch.matmul(query_layer_3, transpose_31)
        query_layer_3 = transpose_31 = None
        attention_scores_10 = attention_scores_9 / 8.0
        attention_scores_9 = None
        attention_scores_11 = attention_scores_10 + extended_attention_mask_2
        attention_scores_10 = None
        attention_probs_6 = torch.nn.functional.softmax(attention_scores_11, dim=-1)
        attention_scores_11 = None
        attention_probs_7 = torch.nn.functional.dropout(
            attention_probs_6, 0.1, False, False
        )
        attention_probs_6 = None
        context_layer_12 = torch.matmul(attention_probs_7, value_layer_3)
        attention_probs_7 = value_layer_3 = None
        permute_3 = context_layer_12.permute(0, 2, 1, 3)
        context_layer_12 = None
        context_layer_13 = permute_3.contiguous()
        permute_3 = None
        conv_out_3 = torch.reshape(conv_out_layer_31, [1, -1, 6, 64])
        conv_out_layer_31 = None
        context_layer_14 = torch.cat([context_layer_13, conv_out_3], 2)
        context_layer_13 = conv_out_3 = None
        context_layer_15 = context_layer_14.view(1, 23, 768)
        context_layer_14 = None
        hidden_states_24 = torch._C._nn.linear(
            context_layer_15,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_15 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_25 = torch.nn.functional.dropout(
            hidden_states_24, 0.1, False, False
        )
        hidden_states_24 = None
        add_12 = hidden_states_25 + hidden_states_23
        hidden_states_25 = hidden_states_23 = None
        hidden_states_26 = torch.nn.functional.layer_norm(
            add_12,
            (768,),
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_12 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_27 = torch._C._nn.linear(
            hidden_states_26,
            l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_28 = torch._C._nn.gelu(hidden_states_27)
        hidden_states_27 = None
        hidden_states_29 = torch._C._nn.linear(
            hidden_states_28,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_28 = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_30 = torch.nn.functional.dropout(
            hidden_states_29, 0.1, False, False
        )
        hidden_states_29 = None
        add_13 = hidden_states_30 + hidden_states_26
        hidden_states_30 = hidden_states_26 = None
        hidden_states_31 = torch.nn.functional.layer_norm(
            add_13,
            (768,),
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_13 = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_bias_ = (None)
        mixed_key_layer_4 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        mixed_value_layer_4 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        transpose_32 = hidden_states_31.transpose(1, 2)
        x_12 = torch.conv1d(
            transpose_32,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_,
            None,
            (1,),
            (4,),
            (1,),
            768,
        )
        transpose_32 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_ = (None)
        x_13 = torch.conv1d(
            x_12,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_,
            None,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_12 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_ = (None)
        x_13 += l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_
        x_14 = x_13
        x_13 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_ = (None)
        mixed_key_conv_attn_layer_4 = x_14.transpose(1, 2)
        x_14 = None
        mixed_query_layer_4 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_16 = mixed_query_layer_4.view(1, -1, 6, 64)
        query_layer_4 = view_16.transpose(1, 2)
        view_16 = None
        view_17 = mixed_key_layer_4.view(1, -1, 6, 64)
        mixed_key_layer_4 = None
        key_layer_4 = view_17.transpose(1, 2)
        view_17 = None
        view_18 = mixed_value_layer_4.view(1, -1, 6, 64)
        mixed_value_layer_4 = None
        value_layer_4 = view_18.transpose(1, 2)
        view_18 = None
        conv_attn_layer_4 = torch.multiply(
            mixed_key_conv_attn_layer_4, mixed_query_layer_4
        )
        mixed_key_conv_attn_layer_4 = mixed_query_layer_4 = None
        conv_kernel_layer_12 = torch._C._nn.linear(
            conv_attn_layer_4,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_,
        )
        conv_attn_layer_4 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_ = (None)
        conv_kernel_layer_13 = torch.reshape(conv_kernel_layer_12, [-1, 9, 1])
        conv_kernel_layer_12 = None
        conv_kernel_layer_14 = torch.softmax(conv_kernel_layer_13, dim=1)
        conv_kernel_layer_13 = None
        conv_out_layer_32 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_ = (None)
        conv_out_layer_33 = torch.reshape(conv_out_layer_32, [1, -1, 384])
        conv_out_layer_32 = None
        transpose_37 = conv_out_layer_33.transpose(1, 2)
        conv_out_layer_33 = None
        contiguous_8 = transpose_37.contiguous()
        transpose_37 = None
        conv_out_layer_34 = contiguous_8.unsqueeze(-1)
        contiguous_8 = None
        conv_out_layer_35 = torch.nn.functional.unfold(
            conv_out_layer_34, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1
        )
        conv_out_layer_34 = None
        transpose_38 = conv_out_layer_35.transpose(1, 2)
        conv_out_layer_35 = None
        conv_out_layer_36 = transpose_38.reshape(1, -1, 384, 9)
        transpose_38 = None
        conv_out_layer_37 = torch.reshape(conv_out_layer_36, [-1, 64, 9])
        conv_out_layer_36 = None
        conv_out_layer_38 = torch.matmul(conv_out_layer_37, conv_kernel_layer_14)
        conv_out_layer_37 = conv_kernel_layer_14 = None
        conv_out_layer_39 = torch.reshape(conv_out_layer_38, [-1, 384])
        conv_out_layer_38 = None
        transpose_39 = key_layer_4.transpose(-1, -2)
        key_layer_4 = None
        attention_scores_12 = torch.matmul(query_layer_4, transpose_39)
        query_layer_4 = transpose_39 = None
        attention_scores_13 = attention_scores_12 / 8.0
        attention_scores_12 = None
        attention_scores_14 = attention_scores_13 + extended_attention_mask_2
        attention_scores_13 = None
        attention_probs_8 = torch.nn.functional.softmax(attention_scores_14, dim=-1)
        attention_scores_14 = None
        attention_probs_9 = torch.nn.functional.dropout(
            attention_probs_8, 0.1, False, False
        )
        attention_probs_8 = None
        context_layer_16 = torch.matmul(attention_probs_9, value_layer_4)
        attention_probs_9 = value_layer_4 = None
        permute_4 = context_layer_16.permute(0, 2, 1, 3)
        context_layer_16 = None
        context_layer_17 = permute_4.contiguous()
        permute_4 = None
        conv_out_4 = torch.reshape(conv_out_layer_39, [1, -1, 6, 64])
        conv_out_layer_39 = None
        context_layer_18 = torch.cat([context_layer_17, conv_out_4], 2)
        context_layer_17 = conv_out_4 = None
        context_layer_19 = context_layer_18.view(1, 23, 768)
        context_layer_18 = None
        hidden_states_32 = torch._C._nn.linear(
            context_layer_19,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_19 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_33 = torch.nn.functional.dropout(
            hidden_states_32, 0.1, False, False
        )
        hidden_states_32 = None
        add_15 = hidden_states_33 + hidden_states_31
        hidden_states_33 = hidden_states_31 = None
        hidden_states_34 = torch.nn.functional.layer_norm(
            add_15,
            (768,),
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_15 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_35 = torch._C._nn.linear(
            hidden_states_34,
            l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_36 = torch._C._nn.gelu(hidden_states_35)
        hidden_states_35 = None
        hidden_states_37 = torch._C._nn.linear(
            hidden_states_36,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_36 = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_38 = torch.nn.functional.dropout(
            hidden_states_37, 0.1, False, False
        )
        hidden_states_37 = None
        add_16 = hidden_states_38 + hidden_states_34
        hidden_states_38 = hidden_states_34 = None
        hidden_states_39 = torch.nn.functional.layer_norm(
            add_16,
            (768,),
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_16 = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_bias_ = (None)
        mixed_key_layer_5 = torch._C._nn.linear(
            hidden_states_39,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        mixed_value_layer_5 = torch._C._nn.linear(
            hidden_states_39,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        transpose_40 = hidden_states_39.transpose(1, 2)
        x_15 = torch.conv1d(
            transpose_40,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_,
            None,
            (1,),
            (4,),
            (1,),
            768,
        )
        transpose_40 = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_ = (None)
        x_16 = torch.conv1d(
            x_15,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_,
            None,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_15 = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_ = (None)
        x_16 += l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_
        x_17 = x_16
        x_16 = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_ = (None)
        mixed_key_conv_attn_layer_5 = x_17.transpose(1, 2)
        x_17 = None
        mixed_query_layer_5 = torch._C._nn.linear(
            hidden_states_39,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_20 = mixed_query_layer_5.view(1, -1, 6, 64)
        query_layer_5 = view_20.transpose(1, 2)
        view_20 = None
        view_21 = mixed_key_layer_5.view(1, -1, 6, 64)
        mixed_key_layer_5 = None
        key_layer_5 = view_21.transpose(1, 2)
        view_21 = None
        view_22 = mixed_value_layer_5.view(1, -1, 6, 64)
        mixed_value_layer_5 = None
        value_layer_5 = view_22.transpose(1, 2)
        view_22 = None
        conv_attn_layer_5 = torch.multiply(
            mixed_key_conv_attn_layer_5, mixed_query_layer_5
        )
        mixed_key_conv_attn_layer_5 = mixed_query_layer_5 = None
        conv_kernel_layer_15 = torch._C._nn.linear(
            conv_attn_layer_5,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_,
        )
        conv_attn_layer_5 = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_ = (None)
        conv_kernel_layer_16 = torch.reshape(conv_kernel_layer_15, [-1, 9, 1])
        conv_kernel_layer_15 = None
        conv_kernel_layer_17 = torch.softmax(conv_kernel_layer_16, dim=1)
        conv_kernel_layer_16 = None
        conv_out_layer_40 = torch._C._nn.linear(
            hidden_states_39,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_ = (None)
        conv_out_layer_41 = torch.reshape(conv_out_layer_40, [1, -1, 384])
        conv_out_layer_40 = None
        transpose_45 = conv_out_layer_41.transpose(1, 2)
        conv_out_layer_41 = None
        contiguous_10 = transpose_45.contiguous()
        transpose_45 = None
        conv_out_layer_42 = contiguous_10.unsqueeze(-1)
        contiguous_10 = None
        conv_out_layer_43 = torch.nn.functional.unfold(
            conv_out_layer_42, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1
        )
        conv_out_layer_42 = None
        transpose_46 = conv_out_layer_43.transpose(1, 2)
        conv_out_layer_43 = None
        conv_out_layer_44 = transpose_46.reshape(1, -1, 384, 9)
        transpose_46 = None
        conv_out_layer_45 = torch.reshape(conv_out_layer_44, [-1, 64, 9])
        conv_out_layer_44 = None
        conv_out_layer_46 = torch.matmul(conv_out_layer_45, conv_kernel_layer_17)
        conv_out_layer_45 = conv_kernel_layer_17 = None
        conv_out_layer_47 = torch.reshape(conv_out_layer_46, [-1, 384])
        conv_out_layer_46 = None
        transpose_47 = key_layer_5.transpose(-1, -2)
        key_layer_5 = None
        attention_scores_15 = torch.matmul(query_layer_5, transpose_47)
        query_layer_5 = transpose_47 = None
        attention_scores_16 = attention_scores_15 / 8.0
        attention_scores_15 = None
        attention_scores_17 = attention_scores_16 + extended_attention_mask_2
        attention_scores_16 = None
        attention_probs_10 = torch.nn.functional.softmax(attention_scores_17, dim=-1)
        attention_scores_17 = None
        attention_probs_11 = torch.nn.functional.dropout(
            attention_probs_10, 0.1, False, False
        )
        attention_probs_10 = None
        context_layer_20 = torch.matmul(attention_probs_11, value_layer_5)
        attention_probs_11 = value_layer_5 = None
        permute_5 = context_layer_20.permute(0, 2, 1, 3)
        context_layer_20 = None
        context_layer_21 = permute_5.contiguous()
        permute_5 = None
        conv_out_5 = torch.reshape(conv_out_layer_47, [1, -1, 6, 64])
        conv_out_layer_47 = None
        context_layer_22 = torch.cat([context_layer_21, conv_out_5], 2)
        context_layer_21 = conv_out_5 = None
        context_layer_23 = context_layer_22.view(1, 23, 768)
        context_layer_22 = None
        hidden_states_40 = torch._C._nn.linear(
            context_layer_23,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_23 = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_41 = torch.nn.functional.dropout(
            hidden_states_40, 0.1, False, False
        )
        hidden_states_40 = None
        add_18 = hidden_states_41 + hidden_states_39
        hidden_states_41 = hidden_states_39 = None
        hidden_states_42 = torch.nn.functional.layer_norm(
            add_18,
            (768,),
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_18 = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_43 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_44 = torch._C._nn.gelu(hidden_states_43)
        hidden_states_43 = None
        hidden_states_45 = torch._C._nn.linear(
            hidden_states_44,
            l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_44 = l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_46 = torch.nn.functional.dropout(
            hidden_states_45, 0.1, False, False
        )
        hidden_states_45 = None
        add_19 = hidden_states_46 + hidden_states_42
        hidden_states_46 = hidden_states_42 = None
        hidden_states_47 = torch.nn.functional.layer_norm(
            add_19,
            (768,),
            l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_19 = l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_parameters_bias_ = (None)
        mixed_key_layer_6 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        mixed_value_layer_6 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        transpose_48 = hidden_states_47.transpose(1, 2)
        x_18 = torch.conv1d(
            transpose_48,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_,
            None,
            (1,),
            (4,),
            (1,),
            768,
        )
        transpose_48 = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_ = (None)
        x_19 = torch.conv1d(
            x_18,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_,
            None,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_18 = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_ = (None)
        x_19 += l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_
        x_20 = x_19
        x_19 = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_ = (None)
        mixed_key_conv_attn_layer_6 = x_20.transpose(1, 2)
        x_20 = None
        mixed_query_layer_6 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_24 = mixed_query_layer_6.view(1, -1, 6, 64)
        query_layer_6 = view_24.transpose(1, 2)
        view_24 = None
        view_25 = mixed_key_layer_6.view(1, -1, 6, 64)
        mixed_key_layer_6 = None
        key_layer_6 = view_25.transpose(1, 2)
        view_25 = None
        view_26 = mixed_value_layer_6.view(1, -1, 6, 64)
        mixed_value_layer_6 = None
        value_layer_6 = view_26.transpose(1, 2)
        view_26 = None
        conv_attn_layer_6 = torch.multiply(
            mixed_key_conv_attn_layer_6, mixed_query_layer_6
        )
        mixed_key_conv_attn_layer_6 = mixed_query_layer_6 = None
        conv_kernel_layer_18 = torch._C._nn.linear(
            conv_attn_layer_6,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_,
        )
        conv_attn_layer_6 = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_ = (None)
        conv_kernel_layer_19 = torch.reshape(conv_kernel_layer_18, [-1, 9, 1])
        conv_kernel_layer_18 = None
        conv_kernel_layer_20 = torch.softmax(conv_kernel_layer_19, dim=1)
        conv_kernel_layer_19 = None
        conv_out_layer_48 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_ = (None)
        conv_out_layer_49 = torch.reshape(conv_out_layer_48, [1, -1, 384])
        conv_out_layer_48 = None
        transpose_53 = conv_out_layer_49.transpose(1, 2)
        conv_out_layer_49 = None
        contiguous_12 = transpose_53.contiguous()
        transpose_53 = None
        conv_out_layer_50 = contiguous_12.unsqueeze(-1)
        contiguous_12 = None
        conv_out_layer_51 = torch.nn.functional.unfold(
            conv_out_layer_50, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1
        )
        conv_out_layer_50 = None
        transpose_54 = conv_out_layer_51.transpose(1, 2)
        conv_out_layer_51 = None
        conv_out_layer_52 = transpose_54.reshape(1, -1, 384, 9)
        transpose_54 = None
        conv_out_layer_53 = torch.reshape(conv_out_layer_52, [-1, 64, 9])
        conv_out_layer_52 = None
        conv_out_layer_54 = torch.matmul(conv_out_layer_53, conv_kernel_layer_20)
        conv_out_layer_53 = conv_kernel_layer_20 = None
        conv_out_layer_55 = torch.reshape(conv_out_layer_54, [-1, 384])
        conv_out_layer_54 = None
        transpose_55 = key_layer_6.transpose(-1, -2)
        key_layer_6 = None
        attention_scores_18 = torch.matmul(query_layer_6, transpose_55)
        query_layer_6 = transpose_55 = None
        attention_scores_19 = attention_scores_18 / 8.0
        attention_scores_18 = None
        attention_scores_20 = attention_scores_19 + extended_attention_mask_2
        attention_scores_19 = None
        attention_probs_12 = torch.nn.functional.softmax(attention_scores_20, dim=-1)
        attention_scores_20 = None
        attention_probs_13 = torch.nn.functional.dropout(
            attention_probs_12, 0.1, False, False
        )
        attention_probs_12 = None
        context_layer_24 = torch.matmul(attention_probs_13, value_layer_6)
        attention_probs_13 = value_layer_6 = None
        permute_6 = context_layer_24.permute(0, 2, 1, 3)
        context_layer_24 = None
        context_layer_25 = permute_6.contiguous()
        permute_6 = None
        conv_out_6 = torch.reshape(conv_out_layer_55, [1, -1, 6, 64])
        conv_out_layer_55 = None
        context_layer_26 = torch.cat([context_layer_25, conv_out_6], 2)
        context_layer_25 = conv_out_6 = None
        context_layer_27 = context_layer_26.view(1, 23, 768)
        context_layer_26 = None
        hidden_states_48 = torch._C._nn.linear(
            context_layer_27,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_27 = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_49 = torch.nn.functional.dropout(
            hidden_states_48, 0.1, False, False
        )
        hidden_states_48 = None
        add_21 = hidden_states_49 + hidden_states_47
        hidden_states_49 = hidden_states_47 = None
        hidden_states_50 = torch.nn.functional.layer_norm(
            add_21,
            (768,),
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_21 = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_51 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_52 = torch._C._nn.gelu(hidden_states_51)
        hidden_states_51 = None
        hidden_states_53 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_52 = l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_54 = torch.nn.functional.dropout(
            hidden_states_53, 0.1, False, False
        )
        hidden_states_53 = None
        add_22 = hidden_states_54 + hidden_states_50
        hidden_states_54 = hidden_states_50 = None
        hidden_states_55 = torch.nn.functional.layer_norm(
            add_22,
            (768,),
            l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_22 = l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_parameters_bias_ = (None)
        mixed_key_layer_7 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        mixed_value_layer_7 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        transpose_56 = hidden_states_55.transpose(1, 2)
        x_21 = torch.conv1d(
            transpose_56,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_,
            None,
            (1,),
            (4,),
            (1,),
            768,
        )
        transpose_56 = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_ = (None)
        x_22 = torch.conv1d(
            x_21,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_,
            None,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_21 = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_ = (None)
        x_22 += l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_
        x_23 = x_22
        x_22 = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_ = (None)
        mixed_key_conv_attn_layer_7 = x_23.transpose(1, 2)
        x_23 = None
        mixed_query_layer_7 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_28 = mixed_query_layer_7.view(1, -1, 6, 64)
        query_layer_7 = view_28.transpose(1, 2)
        view_28 = None
        view_29 = mixed_key_layer_7.view(1, -1, 6, 64)
        mixed_key_layer_7 = None
        key_layer_7 = view_29.transpose(1, 2)
        view_29 = None
        view_30 = mixed_value_layer_7.view(1, -1, 6, 64)
        mixed_value_layer_7 = None
        value_layer_7 = view_30.transpose(1, 2)
        view_30 = None
        conv_attn_layer_7 = torch.multiply(
            mixed_key_conv_attn_layer_7, mixed_query_layer_7
        )
        mixed_key_conv_attn_layer_7 = mixed_query_layer_7 = None
        conv_kernel_layer_21 = torch._C._nn.linear(
            conv_attn_layer_7,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_,
        )
        conv_attn_layer_7 = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_ = (None)
        conv_kernel_layer_22 = torch.reshape(conv_kernel_layer_21, [-1, 9, 1])
        conv_kernel_layer_21 = None
        conv_kernel_layer_23 = torch.softmax(conv_kernel_layer_22, dim=1)
        conv_kernel_layer_22 = None
        conv_out_layer_56 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_ = (None)
        conv_out_layer_57 = torch.reshape(conv_out_layer_56, [1, -1, 384])
        conv_out_layer_56 = None
        transpose_61 = conv_out_layer_57.transpose(1, 2)
        conv_out_layer_57 = None
        contiguous_14 = transpose_61.contiguous()
        transpose_61 = None
        conv_out_layer_58 = contiguous_14.unsqueeze(-1)
        contiguous_14 = None
        conv_out_layer_59 = torch.nn.functional.unfold(
            conv_out_layer_58, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1
        )
        conv_out_layer_58 = None
        transpose_62 = conv_out_layer_59.transpose(1, 2)
        conv_out_layer_59 = None
        conv_out_layer_60 = transpose_62.reshape(1, -1, 384, 9)
        transpose_62 = None
        conv_out_layer_61 = torch.reshape(conv_out_layer_60, [-1, 64, 9])
        conv_out_layer_60 = None
        conv_out_layer_62 = torch.matmul(conv_out_layer_61, conv_kernel_layer_23)
        conv_out_layer_61 = conv_kernel_layer_23 = None
        conv_out_layer_63 = torch.reshape(conv_out_layer_62, [-1, 384])
        conv_out_layer_62 = None
        transpose_63 = key_layer_7.transpose(-1, -2)
        key_layer_7 = None
        attention_scores_21 = torch.matmul(query_layer_7, transpose_63)
        query_layer_7 = transpose_63 = None
        attention_scores_22 = attention_scores_21 / 8.0
        attention_scores_21 = None
        attention_scores_23 = attention_scores_22 + extended_attention_mask_2
        attention_scores_22 = None
        attention_probs_14 = torch.nn.functional.softmax(attention_scores_23, dim=-1)
        attention_scores_23 = None
        attention_probs_15 = torch.nn.functional.dropout(
            attention_probs_14, 0.1, False, False
        )
        attention_probs_14 = None
        context_layer_28 = torch.matmul(attention_probs_15, value_layer_7)
        attention_probs_15 = value_layer_7 = None
        permute_7 = context_layer_28.permute(0, 2, 1, 3)
        context_layer_28 = None
        context_layer_29 = permute_7.contiguous()
        permute_7 = None
        conv_out_7 = torch.reshape(conv_out_layer_63, [1, -1, 6, 64])
        conv_out_layer_63 = None
        context_layer_30 = torch.cat([context_layer_29, conv_out_7], 2)
        context_layer_29 = conv_out_7 = None
        context_layer_31 = context_layer_30.view(1, 23, 768)
        context_layer_30 = None
        hidden_states_56 = torch._C._nn.linear(
            context_layer_31,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_31 = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_57 = torch.nn.functional.dropout(
            hidden_states_56, 0.1, False, False
        )
        hidden_states_56 = None
        add_24 = hidden_states_57 + hidden_states_55
        hidden_states_57 = hidden_states_55 = None
        hidden_states_58 = torch.nn.functional.layer_norm(
            add_24,
            (768,),
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_24 = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_59 = torch._C._nn.linear(
            hidden_states_58,
            l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_60 = torch._C._nn.gelu(hidden_states_59)
        hidden_states_59 = None
        hidden_states_61 = torch._C._nn.linear(
            hidden_states_60,
            l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_60 = l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_62 = torch.nn.functional.dropout(
            hidden_states_61, 0.1, False, False
        )
        hidden_states_61 = None
        add_25 = hidden_states_62 + hidden_states_58
        hidden_states_62 = hidden_states_58 = None
        hidden_states_63 = torch.nn.functional.layer_norm(
            add_25,
            (768,),
            l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_25 = l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_parameters_bias_ = (None)
        mixed_key_layer_8 = torch._C._nn.linear(
            hidden_states_63,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        mixed_value_layer_8 = torch._C._nn.linear(
            hidden_states_63,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        transpose_64 = hidden_states_63.transpose(1, 2)
        x_24 = torch.conv1d(
            transpose_64,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_,
            None,
            (1,),
            (4,),
            (1,),
            768,
        )
        transpose_64 = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_ = (None)
        x_25 = torch.conv1d(
            x_24,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_,
            None,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_24 = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_ = (None)
        x_25 += l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_
        x_26 = x_25
        x_25 = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_ = (None)
        mixed_key_conv_attn_layer_8 = x_26.transpose(1, 2)
        x_26 = None
        mixed_query_layer_8 = torch._C._nn.linear(
            hidden_states_63,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_32 = mixed_query_layer_8.view(1, -1, 6, 64)
        query_layer_8 = view_32.transpose(1, 2)
        view_32 = None
        view_33 = mixed_key_layer_8.view(1, -1, 6, 64)
        mixed_key_layer_8 = None
        key_layer_8 = view_33.transpose(1, 2)
        view_33 = None
        view_34 = mixed_value_layer_8.view(1, -1, 6, 64)
        mixed_value_layer_8 = None
        value_layer_8 = view_34.transpose(1, 2)
        view_34 = None
        conv_attn_layer_8 = torch.multiply(
            mixed_key_conv_attn_layer_8, mixed_query_layer_8
        )
        mixed_key_conv_attn_layer_8 = mixed_query_layer_8 = None
        conv_kernel_layer_24 = torch._C._nn.linear(
            conv_attn_layer_8,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_,
        )
        conv_attn_layer_8 = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_ = (None)
        conv_kernel_layer_25 = torch.reshape(conv_kernel_layer_24, [-1, 9, 1])
        conv_kernel_layer_24 = None
        conv_kernel_layer_26 = torch.softmax(conv_kernel_layer_25, dim=1)
        conv_kernel_layer_25 = None
        conv_out_layer_64 = torch._C._nn.linear(
            hidden_states_63,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_ = (None)
        conv_out_layer_65 = torch.reshape(conv_out_layer_64, [1, -1, 384])
        conv_out_layer_64 = None
        transpose_69 = conv_out_layer_65.transpose(1, 2)
        conv_out_layer_65 = None
        contiguous_16 = transpose_69.contiguous()
        transpose_69 = None
        conv_out_layer_66 = contiguous_16.unsqueeze(-1)
        contiguous_16 = None
        conv_out_layer_67 = torch.nn.functional.unfold(
            conv_out_layer_66, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1
        )
        conv_out_layer_66 = None
        transpose_70 = conv_out_layer_67.transpose(1, 2)
        conv_out_layer_67 = None
        conv_out_layer_68 = transpose_70.reshape(1, -1, 384, 9)
        transpose_70 = None
        conv_out_layer_69 = torch.reshape(conv_out_layer_68, [-1, 64, 9])
        conv_out_layer_68 = None
        conv_out_layer_70 = torch.matmul(conv_out_layer_69, conv_kernel_layer_26)
        conv_out_layer_69 = conv_kernel_layer_26 = None
        conv_out_layer_71 = torch.reshape(conv_out_layer_70, [-1, 384])
        conv_out_layer_70 = None
        transpose_71 = key_layer_8.transpose(-1, -2)
        key_layer_8 = None
        attention_scores_24 = torch.matmul(query_layer_8, transpose_71)
        query_layer_8 = transpose_71 = None
        attention_scores_25 = attention_scores_24 / 8.0
        attention_scores_24 = None
        attention_scores_26 = attention_scores_25 + extended_attention_mask_2
        attention_scores_25 = None
        attention_probs_16 = torch.nn.functional.softmax(attention_scores_26, dim=-1)
        attention_scores_26 = None
        attention_probs_17 = torch.nn.functional.dropout(
            attention_probs_16, 0.1, False, False
        )
        attention_probs_16 = None
        context_layer_32 = torch.matmul(attention_probs_17, value_layer_8)
        attention_probs_17 = value_layer_8 = None
        permute_8 = context_layer_32.permute(0, 2, 1, 3)
        context_layer_32 = None
        context_layer_33 = permute_8.contiguous()
        permute_8 = None
        conv_out_8 = torch.reshape(conv_out_layer_71, [1, -1, 6, 64])
        conv_out_layer_71 = None
        context_layer_34 = torch.cat([context_layer_33, conv_out_8], 2)
        context_layer_33 = conv_out_8 = None
        context_layer_35 = context_layer_34.view(1, 23, 768)
        context_layer_34 = None
        hidden_states_64 = torch._C._nn.linear(
            context_layer_35,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_35 = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_65 = torch.nn.functional.dropout(
            hidden_states_64, 0.1, False, False
        )
        hidden_states_64 = None
        add_27 = hidden_states_65 + hidden_states_63
        hidden_states_65 = hidden_states_63 = None
        hidden_states_66 = torch.nn.functional.layer_norm(
            add_27,
            (768,),
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_27 = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_67 = torch._C._nn.linear(
            hidden_states_66,
            l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_68 = torch._C._nn.gelu(hidden_states_67)
        hidden_states_67 = None
        hidden_states_69 = torch._C._nn.linear(
            hidden_states_68,
            l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_68 = l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_70 = torch.nn.functional.dropout(
            hidden_states_69, 0.1, False, False
        )
        hidden_states_69 = None
        add_28 = hidden_states_70 + hidden_states_66
        hidden_states_70 = hidden_states_66 = None
        hidden_states_71 = torch.nn.functional.layer_norm(
            add_28,
            (768,),
            l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_28 = l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_parameters_bias_ = (None)
        mixed_key_layer_9 = torch._C._nn.linear(
            hidden_states_71,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        mixed_value_layer_9 = torch._C._nn.linear(
            hidden_states_71,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        transpose_72 = hidden_states_71.transpose(1, 2)
        x_27 = torch.conv1d(
            transpose_72,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_,
            None,
            (1,),
            (4,),
            (1,),
            768,
        )
        transpose_72 = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_ = (None)
        x_28 = torch.conv1d(
            x_27,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_,
            None,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_27 = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_ = (None)
        x_28 += l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_
        x_29 = x_28
        x_28 = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_ = (None)
        mixed_key_conv_attn_layer_9 = x_29.transpose(1, 2)
        x_29 = None
        mixed_query_layer_9 = torch._C._nn.linear(
            hidden_states_71,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_36 = mixed_query_layer_9.view(1, -1, 6, 64)
        query_layer_9 = view_36.transpose(1, 2)
        view_36 = None
        view_37 = mixed_key_layer_9.view(1, -1, 6, 64)
        mixed_key_layer_9 = None
        key_layer_9 = view_37.transpose(1, 2)
        view_37 = None
        view_38 = mixed_value_layer_9.view(1, -1, 6, 64)
        mixed_value_layer_9 = None
        value_layer_9 = view_38.transpose(1, 2)
        view_38 = None
        conv_attn_layer_9 = torch.multiply(
            mixed_key_conv_attn_layer_9, mixed_query_layer_9
        )
        mixed_key_conv_attn_layer_9 = mixed_query_layer_9 = None
        conv_kernel_layer_27 = torch._C._nn.linear(
            conv_attn_layer_9,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_,
        )
        conv_attn_layer_9 = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_ = (None)
        conv_kernel_layer_28 = torch.reshape(conv_kernel_layer_27, [-1, 9, 1])
        conv_kernel_layer_27 = None
        conv_kernel_layer_29 = torch.softmax(conv_kernel_layer_28, dim=1)
        conv_kernel_layer_28 = None
        conv_out_layer_72 = torch._C._nn.linear(
            hidden_states_71,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_ = (None)
        conv_out_layer_73 = torch.reshape(conv_out_layer_72, [1, -1, 384])
        conv_out_layer_72 = None
        transpose_77 = conv_out_layer_73.transpose(1, 2)
        conv_out_layer_73 = None
        contiguous_18 = transpose_77.contiguous()
        transpose_77 = None
        conv_out_layer_74 = contiguous_18.unsqueeze(-1)
        contiguous_18 = None
        conv_out_layer_75 = torch.nn.functional.unfold(
            conv_out_layer_74, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1
        )
        conv_out_layer_74 = None
        transpose_78 = conv_out_layer_75.transpose(1, 2)
        conv_out_layer_75 = None
        conv_out_layer_76 = transpose_78.reshape(1, -1, 384, 9)
        transpose_78 = None
        conv_out_layer_77 = torch.reshape(conv_out_layer_76, [-1, 64, 9])
        conv_out_layer_76 = None
        conv_out_layer_78 = torch.matmul(conv_out_layer_77, conv_kernel_layer_29)
        conv_out_layer_77 = conv_kernel_layer_29 = None
        conv_out_layer_79 = torch.reshape(conv_out_layer_78, [-1, 384])
        conv_out_layer_78 = None
        transpose_79 = key_layer_9.transpose(-1, -2)
        key_layer_9 = None
        attention_scores_27 = torch.matmul(query_layer_9, transpose_79)
        query_layer_9 = transpose_79 = None
        attention_scores_28 = attention_scores_27 / 8.0
        attention_scores_27 = None
        attention_scores_29 = attention_scores_28 + extended_attention_mask_2
        attention_scores_28 = None
        attention_probs_18 = torch.nn.functional.softmax(attention_scores_29, dim=-1)
        attention_scores_29 = None
        attention_probs_19 = torch.nn.functional.dropout(
            attention_probs_18, 0.1, False, False
        )
        attention_probs_18 = None
        context_layer_36 = torch.matmul(attention_probs_19, value_layer_9)
        attention_probs_19 = value_layer_9 = None
        permute_9 = context_layer_36.permute(0, 2, 1, 3)
        context_layer_36 = None
        context_layer_37 = permute_9.contiguous()
        permute_9 = None
        conv_out_9 = torch.reshape(conv_out_layer_79, [1, -1, 6, 64])
        conv_out_layer_79 = None
        context_layer_38 = torch.cat([context_layer_37, conv_out_9], 2)
        context_layer_37 = conv_out_9 = None
        context_layer_39 = context_layer_38.view(1, 23, 768)
        context_layer_38 = None
        hidden_states_72 = torch._C._nn.linear(
            context_layer_39,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_39 = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_73 = torch.nn.functional.dropout(
            hidden_states_72, 0.1, False, False
        )
        hidden_states_72 = None
        add_30 = hidden_states_73 + hidden_states_71
        hidden_states_73 = hidden_states_71 = None
        hidden_states_74 = torch.nn.functional.layer_norm(
            add_30,
            (768,),
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_30 = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_75 = torch._C._nn.linear(
            hidden_states_74,
            l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_76 = torch._C._nn.gelu(hidden_states_75)
        hidden_states_75 = None
        hidden_states_77 = torch._C._nn.linear(
            hidden_states_76,
            l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_76 = l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_78 = torch.nn.functional.dropout(
            hidden_states_77, 0.1, False, False
        )
        hidden_states_77 = None
        add_31 = hidden_states_78 + hidden_states_74
        hidden_states_78 = hidden_states_74 = None
        hidden_states_79 = torch.nn.functional.layer_norm(
            add_31,
            (768,),
            l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_31 = l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_parameters_bias_ = (None)
        mixed_key_layer_10 = torch._C._nn.linear(
            hidden_states_79,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        mixed_value_layer_10 = torch._C._nn.linear(
            hidden_states_79,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        transpose_80 = hidden_states_79.transpose(1, 2)
        x_30 = torch.conv1d(
            transpose_80,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_,
            None,
            (1,),
            (4,),
            (1,),
            768,
        )
        transpose_80 = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_ = (None)
        x_31 = torch.conv1d(
            x_30,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_,
            None,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_30 = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_ = (None)
        x_31 += l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_
        x_32 = x_31
        x_31 = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_ = (None)
        mixed_key_conv_attn_layer_10 = x_32.transpose(1, 2)
        x_32 = None
        mixed_query_layer_10 = torch._C._nn.linear(
            hidden_states_79,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_40 = mixed_query_layer_10.view(1, -1, 6, 64)
        query_layer_10 = view_40.transpose(1, 2)
        view_40 = None
        view_41 = mixed_key_layer_10.view(1, -1, 6, 64)
        mixed_key_layer_10 = None
        key_layer_10 = view_41.transpose(1, 2)
        view_41 = None
        view_42 = mixed_value_layer_10.view(1, -1, 6, 64)
        mixed_value_layer_10 = None
        value_layer_10 = view_42.transpose(1, 2)
        view_42 = None
        conv_attn_layer_10 = torch.multiply(
            mixed_key_conv_attn_layer_10, mixed_query_layer_10
        )
        mixed_key_conv_attn_layer_10 = mixed_query_layer_10 = None
        conv_kernel_layer_30 = torch._C._nn.linear(
            conv_attn_layer_10,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_,
        )
        conv_attn_layer_10 = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_ = (None)
        conv_kernel_layer_31 = torch.reshape(conv_kernel_layer_30, [-1, 9, 1])
        conv_kernel_layer_30 = None
        conv_kernel_layer_32 = torch.softmax(conv_kernel_layer_31, dim=1)
        conv_kernel_layer_31 = None
        conv_out_layer_80 = torch._C._nn.linear(
            hidden_states_79,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_ = (None)
        conv_out_layer_81 = torch.reshape(conv_out_layer_80, [1, -1, 384])
        conv_out_layer_80 = None
        transpose_85 = conv_out_layer_81.transpose(1, 2)
        conv_out_layer_81 = None
        contiguous_20 = transpose_85.contiguous()
        transpose_85 = None
        conv_out_layer_82 = contiguous_20.unsqueeze(-1)
        contiguous_20 = None
        conv_out_layer_83 = torch.nn.functional.unfold(
            conv_out_layer_82, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1
        )
        conv_out_layer_82 = None
        transpose_86 = conv_out_layer_83.transpose(1, 2)
        conv_out_layer_83 = None
        conv_out_layer_84 = transpose_86.reshape(1, -1, 384, 9)
        transpose_86 = None
        conv_out_layer_85 = torch.reshape(conv_out_layer_84, [-1, 64, 9])
        conv_out_layer_84 = None
        conv_out_layer_86 = torch.matmul(conv_out_layer_85, conv_kernel_layer_32)
        conv_out_layer_85 = conv_kernel_layer_32 = None
        conv_out_layer_87 = torch.reshape(conv_out_layer_86, [-1, 384])
        conv_out_layer_86 = None
        transpose_87 = key_layer_10.transpose(-1, -2)
        key_layer_10 = None
        attention_scores_30 = torch.matmul(query_layer_10, transpose_87)
        query_layer_10 = transpose_87 = None
        attention_scores_31 = attention_scores_30 / 8.0
        attention_scores_30 = None
        attention_scores_32 = attention_scores_31 + extended_attention_mask_2
        attention_scores_31 = None
        attention_probs_20 = torch.nn.functional.softmax(attention_scores_32, dim=-1)
        attention_scores_32 = None
        attention_probs_21 = torch.nn.functional.dropout(
            attention_probs_20, 0.1, False, False
        )
        attention_probs_20 = None
        context_layer_40 = torch.matmul(attention_probs_21, value_layer_10)
        attention_probs_21 = value_layer_10 = None
        permute_10 = context_layer_40.permute(0, 2, 1, 3)
        context_layer_40 = None
        context_layer_41 = permute_10.contiguous()
        permute_10 = None
        conv_out_10 = torch.reshape(conv_out_layer_87, [1, -1, 6, 64])
        conv_out_layer_87 = None
        context_layer_42 = torch.cat([context_layer_41, conv_out_10], 2)
        context_layer_41 = conv_out_10 = None
        context_layer_43 = context_layer_42.view(1, 23, 768)
        context_layer_42 = None
        hidden_states_80 = torch._C._nn.linear(
            context_layer_43,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_43 = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_81 = torch.nn.functional.dropout(
            hidden_states_80, 0.1, False, False
        )
        hidden_states_80 = None
        add_33 = hidden_states_81 + hidden_states_79
        hidden_states_81 = hidden_states_79 = None
        hidden_states_82 = torch.nn.functional.layer_norm(
            add_33,
            (768,),
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_33 = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_83 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_84 = torch._C._nn.gelu(hidden_states_83)
        hidden_states_83 = None
        hidden_states_85 = torch._C._nn.linear(
            hidden_states_84,
            l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_84 = l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_86 = torch.nn.functional.dropout(
            hidden_states_85, 0.1, False, False
        )
        hidden_states_85 = None
        add_34 = hidden_states_86 + hidden_states_82
        hidden_states_86 = hidden_states_82 = None
        hidden_states_87 = torch.nn.functional.layer_norm(
            add_34,
            (768,),
            l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_34 = l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_parameters_bias_ = (None)
        mixed_key_layer_11 = torch._C._nn.linear(
            hidden_states_87,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        mixed_value_layer_11 = torch._C._nn.linear(
            hidden_states_87,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        transpose_88 = hidden_states_87.transpose(1, 2)
        x_33 = torch.conv1d(
            transpose_88,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_,
            None,
            (1,),
            (4,),
            (1,),
            768,
        )
        transpose_88 = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_conv_attn_layer_modules_depthwise_parameters_weight_ = (None)
        x_34 = torch.conv1d(
            x_33,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_,
            None,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_33 = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_conv_attn_layer_modules_pointwise_parameters_weight_ = (None)
        x_34 += l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_
        x_35 = x_34
        x_34 = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_conv_attn_layer_parameters_bias_ = (None)
        mixed_key_conv_attn_layer_11 = x_35.transpose(1, 2)
        x_35 = None
        mixed_query_layer_11 = torch._C._nn.linear(
            hidden_states_87,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_44 = mixed_query_layer_11.view(1, -1, 6, 64)
        query_layer_11 = view_44.transpose(1, 2)
        view_44 = None
        view_45 = mixed_key_layer_11.view(1, -1, 6, 64)
        mixed_key_layer_11 = None
        key_layer_11 = view_45.transpose(1, 2)
        view_45 = None
        view_46 = mixed_value_layer_11.view(1, -1, 6, 64)
        mixed_value_layer_11 = None
        value_layer_11 = view_46.transpose(1, 2)
        view_46 = None
        conv_attn_layer_11 = torch.multiply(
            mixed_key_conv_attn_layer_11, mixed_query_layer_11
        )
        mixed_key_conv_attn_layer_11 = mixed_query_layer_11 = None
        conv_kernel_layer_33 = torch._C._nn.linear(
            conv_attn_layer_11,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_,
        )
        conv_attn_layer_11 = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_conv_kernel_layer_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_conv_kernel_layer_parameters_bias_ = (None)
        conv_kernel_layer_34 = torch.reshape(conv_kernel_layer_33, [-1, 9, 1])
        conv_kernel_layer_33 = None
        conv_kernel_layer_35 = torch.softmax(conv_kernel_layer_34, dim=1)
        conv_kernel_layer_34 = None
        conv_out_layer_88 = torch._C._nn.linear(
            hidden_states_87,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_conv_out_layer_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_conv_out_layer_parameters_bias_ = (None)
        conv_out_layer_89 = torch.reshape(conv_out_layer_88, [1, -1, 384])
        conv_out_layer_88 = None
        transpose_93 = conv_out_layer_89.transpose(1, 2)
        conv_out_layer_89 = None
        contiguous_22 = transpose_93.contiguous()
        transpose_93 = None
        conv_out_layer_90 = contiguous_22.unsqueeze(-1)
        contiguous_22 = None
        conv_out_layer_91 = torch.nn.functional.unfold(
            conv_out_layer_90, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1
        )
        conv_out_layer_90 = None
        transpose_94 = conv_out_layer_91.transpose(1, 2)
        conv_out_layer_91 = None
        conv_out_layer_92 = transpose_94.reshape(1, -1, 384, 9)
        transpose_94 = None
        conv_out_layer_93 = torch.reshape(conv_out_layer_92, [-1, 64, 9])
        conv_out_layer_92 = None
        conv_out_layer_94 = torch.matmul(conv_out_layer_93, conv_kernel_layer_35)
        conv_out_layer_93 = conv_kernel_layer_35 = None
        conv_out_layer_95 = torch.reshape(conv_out_layer_94, [-1, 384])
        conv_out_layer_94 = None
        transpose_95 = key_layer_11.transpose(-1, -2)
        key_layer_11 = None
        attention_scores_33 = torch.matmul(query_layer_11, transpose_95)
        query_layer_11 = transpose_95 = None
        attention_scores_34 = attention_scores_33 / 8.0
        attention_scores_33 = None
        attention_scores_35 = attention_scores_34 + extended_attention_mask_2
        attention_scores_34 = extended_attention_mask_2 = None
        attention_probs_22 = torch.nn.functional.softmax(attention_scores_35, dim=-1)
        attention_scores_35 = None
        attention_probs_23 = torch.nn.functional.dropout(
            attention_probs_22, 0.1, False, False
        )
        attention_probs_22 = None
        context_layer_44 = torch.matmul(attention_probs_23, value_layer_11)
        attention_probs_23 = value_layer_11 = None
        permute_11 = context_layer_44.permute(0, 2, 1, 3)
        context_layer_44 = None
        context_layer_45 = permute_11.contiguous()
        permute_11 = None
        conv_out_11 = torch.reshape(conv_out_layer_95, [1, -1, 6, 64])
        conv_out_layer_95 = None
        context_layer_46 = torch.cat([context_layer_45, conv_out_11], 2)
        context_layer_45 = conv_out_11 = None
        context_layer_47 = context_layer_46.view(1, 23, 768)
        context_layer_46 = None
        hidden_states_88 = torch._C._nn.linear(
            context_layer_47,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_47 = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_89 = torch.nn.functional.dropout(
            hidden_states_88, 0.1, False, False
        )
        hidden_states_88 = None
        add_36 = hidden_states_89 + hidden_states_87
        hidden_states_89 = hidden_states_87 = None
        hidden_states_90 = torch.nn.functional.layer_norm(
            add_36,
            (768,),
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_36 = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_91 = torch._C._nn.linear(
            hidden_states_90,
            l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_92 = torch._C._nn.gelu(hidden_states_91)
        hidden_states_91 = None
        hidden_states_93 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_92 = l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_94 = torch.nn.functional.dropout(
            hidden_states_93, 0.1, False, False
        )
        hidden_states_93 = None
        add_37 = hidden_states_94 + hidden_states_90
        hidden_states_94 = hidden_states_90 = None
        hidden_states_95 = torch.nn.functional.layer_norm(
            add_37,
            (768,),
            l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_37 = l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_parameters_bias_ = (None)
        return (hidden_states_95,)
