import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_attention_mask_: torch.Tensor,
        L_token_type_ids_: torch.Tensor,
        L_self_modules_embeddings_modules_word_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_buffers_position_ids_: torch.Tensor,
        L_self_modules_embeddings_modules_embedding_transformation_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_embedding_transformation_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_position_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_input_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_input_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_input_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_input_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_bottleneck_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_bottleneck_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_bottleneck_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_bottleneck_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_input_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_input_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_input_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_input_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_bottleneck_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_bottleneck_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_bottleneck_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_bottleneck_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_input_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_input_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_input_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_input_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_bottleneck_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_bottleneck_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_bottleneck_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_bottleneck_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_input_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_input_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_input_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_input_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_bottleneck_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_bottleneck_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_bottleneck_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_bottleneck_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_input_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_input_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_input_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_input_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_bottleneck_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_bottleneck_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_bottleneck_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_bottleneck_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_pooler_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_pooler_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_embeddings_modules_embedding_transformation_parameters_weight_ = L_self_modules_embeddings_modules_embedding_transformation_parameters_weight_
        l_self_modules_embeddings_modules_embedding_transformation_parameters_bias_ = (
            L_self_modules_embeddings_modules_embedding_transformation_parameters_bias_
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
        l_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_input_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_input_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_input_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_input_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_input_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_input_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_input_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_input_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_attention_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_attention_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_attention_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_attention_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_bottleneck_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_bottleneck_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_bottleneck_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_bottleneck_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_bottleneck_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_bottleneck_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_bottleneck_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_bottleneck_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_input_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_input_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_input_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_input_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_input_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_input_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_input_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_input_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_attention_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_attention_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_attention_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_attention_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_bottleneck_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_bottleneck_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_bottleneck_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_bottleneck_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_bottleneck_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_bottleneck_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_bottleneck_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_bottleneck_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_input_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_input_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_input_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_input_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_input_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_input_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_input_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_input_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_attention_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_attention_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_attention_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_attention_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_bottleneck_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_bottleneck_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_bottleneck_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_bottleneck_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_bottleneck_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_bottleneck_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_bottleneck_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_bottleneck_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_input_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_input_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_input_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_input_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_input_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_input_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_input_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_input_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_attention_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_attention_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_attention_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_attention_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_bottleneck_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_bottleneck_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_bottleneck_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_bottleneck_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_bottleneck_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_bottleneck_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_bottleneck_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_bottleneck_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_input_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_input_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_input_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_input_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_input_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_input_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_input_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_input_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_attention_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_attention_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_attention_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_attention_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_bottleneck_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_bottleneck_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_bottleneck_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_bottleneck_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_bottleneck_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_bottleneck_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_bottleneck_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_bottleneck_modules_LayerNorm_parameters_bias_
        l_self_modules_pooler_modules_dense_parameters_weight_ = (
            L_self_modules_pooler_modules_dense_parameters_weight_
        )
        l_self_modules_pooler_modules_dense_parameters_bias_ = (
            L_self_modules_pooler_modules_dense_parameters_bias_
        )
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
            (slice(None, None, None), slice(None, 45, None))
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
        getitem_2 = inputs_embeds[(slice(None, None, None), slice(1, None, None))]
        pad = torch._C._nn.pad(getitem_2, [0, 0, 0, 1, 0, 0], "constant", 0.0)
        getitem_2 = None
        getitem_3 = inputs_embeds[(slice(None, None, None), slice(None, -1, None))]
        pad_1 = torch._C._nn.pad(getitem_3, [0, 0, 1, 0, 0, 0], "constant", 0.0)
        getitem_3 = None
        inputs_embeds_1 = torch.cat([pad, inputs_embeds, pad_1], dim=2)
        pad = inputs_embeds = pad_1 = None
        inputs_embeds_2 = torch._C._nn.linear(
            inputs_embeds_1,
            l_self_modules_embeddings_modules_embedding_transformation_parameters_weight_,
            l_self_modules_embeddings_modules_embedding_transformation_parameters_bias_,
        )
        inputs_embeds_1 = l_self_modules_embeddings_modules_embedding_transformation_parameters_weight_ = (
            l_self_modules_embeddings_modules_embedding_transformation_parameters_bias_
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
        add = inputs_embeds_2 + position_embeddings
        inputs_embeds_2 = position_embeddings = None
        embeddings = add + token_type_embeddings
        add = token_type_embeddings = None
        mul_1 = (
            embeddings * l_self_modules_embeddings_modules_layer_norm_parameters_weight_
        )
        embeddings = (
            l_self_modules_embeddings_modules_layer_norm_parameters_weight_
        ) = None
        embeddings_1 = (
            mul_1 + l_self_modules_embeddings_modules_layer_norm_parameters_bias_
        )
        mul_1 = l_self_modules_embeddings_modules_layer_norm_parameters_bias_ = None
        embeddings_2 = torch.nn.functional.dropout(embeddings_1, 0.1, False, False)
        embeddings_1 = None
        layer_input = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_input_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_input_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_input_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_input_modules_dense_parameters_bias_ = (None)
        mul_2 = (
            layer_input
            * l_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_input_modules_layer_norm_parameters_weight_
        )
        layer_input = l_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_input_modules_layer_norm_parameters_weight_ = (None)
        layer_input_1 = (
            mul_2
            + l_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_input_modules_layer_norm_parameters_bias_
        )
        mul_2 = l_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_input_modules_layer_norm_parameters_bias_ = (None)
        layer_input_2 = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_attention_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_attention_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_attention_modules_dense_parameters_bias_ = (None)
        mul_3 = (
            layer_input_2
            * l_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_attention_modules_layer_norm_parameters_weight_
        )
        layer_input_2 = l_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_attention_modules_layer_norm_parameters_weight_ = (None)
        layer_input_3 = (
            mul_3
            + l_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_attention_modules_layer_norm_parameters_bias_
        )
        mul_3 = l_self_modules_encoder_modules_layer_modules_0_modules_bottleneck_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_3 = torch._C._nn.linear(
            layer_input_3,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view = linear_3.view(1, -1, 4, 32)
        linear_3 = None
        query_layer = view.transpose(1, 2)
        view = None
        linear_4 = torch._C._nn.linear(
            layer_input_3,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        layer_input_3 = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_1 = linear_4.view(1, -1, 4, 32)
        linear_4 = None
        key_layer = view_1.transpose(1, 2)
        view_1 = None
        linear_5 = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_2 = linear_5.view(1, -1, 4, 32)
        linear_5 = None
        value_layer = view_2.transpose(1, 2)
        view_2 = None
        transpose_3 = key_layer.transpose(-1, -2)
        key_layer = None
        attention_scores = torch.matmul(query_layer, transpose_3)
        query_layer = transpose_3 = None
        attention_scores_1 = attention_scores / 5.656854249492381
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
        context_layer_2 = context_layer_1.view((1, 45, 128))
        context_layer_1 = None
        layer_outputs = torch._C._nn.linear(
            context_layer_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_2 = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        add_6 = layer_outputs + layer_input_1
        layer_outputs = layer_input_1 = None
        mul_4 = (
            add_6
            * l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_weight_
        )
        add_6 = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = (None)
        layer_outputs_1 = (
            mul_4
            + l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_4 = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states = torch._C._nn.linear(
            layer_outputs_1,
            l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_1 = torch._C._nn.gelu(hidden_states)
        hidden_states = None
        layer_outputs_2 = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_1 = l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_output_modules_dense_parameters_bias_ = (None)
        add_8 = layer_outputs_2 + layer_outputs_1
        layer_outputs_2 = layer_outputs_1 = None
        mul_5 = (
            add_8
            * l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_weight_
        )
        add_8 = l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_weight_ = (None)
        layer_outputs_3 = (
            mul_5
            + l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_5 = l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_2 = torch._C._nn.linear(
            layer_outputs_3,
            l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_3 = torch._C._nn.gelu(hidden_states_2)
        hidden_states_2 = None
        layer_outputs_4 = torch._C._nn.linear(
            hidden_states_3,
            l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_3 = l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_output_modules_dense_parameters_bias_ = (None)
        add_10 = layer_outputs_4 + layer_outputs_3
        layer_outputs_4 = layer_outputs_3 = None
        mul_6 = (
            add_10
            * l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_weight_
        )
        add_10 = l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_weight_ = (None)
        layer_outputs_5 = (
            mul_6
            + l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_6 = l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_4 = torch._C._nn.linear(
            layer_outputs_5,
            l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_5 = torch._C._nn.gelu(hidden_states_4)
        hidden_states_4 = None
        layer_outputs_6 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_5 = l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_output_modules_dense_parameters_bias_ = (None)
        add_12 = layer_outputs_6 + layer_outputs_5
        layer_outputs_6 = layer_outputs_5 = None
        mul_7 = (
            add_12
            * l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_weight_
        )
        add_12 = l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_weight_ = (None)
        layer_outputs_7 = (
            mul_7
            + l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_7 = l_self_modules_encoder_modules_layer_modules_0_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_6 = torch._C._nn.linear(
            layer_outputs_7,
            l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_7 = torch._C._nn.gelu(hidden_states_6)
        hidden_states_6 = None
        layer_output = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_7 = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_ = (None)
        add_14 = layer_output + layer_outputs_7
        layer_output = layer_outputs_7 = None
        mul_8 = (
            add_14
            * l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_weight_
        )
        add_14 = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_weight_ = (None)
        layer_output_1 = (
            mul_8
            + l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_8 = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_bias_ = (None)
        layer_outputs_8 = torch._C._nn.linear(
            layer_output_1,
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_bottleneck_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_bottleneck_modules_dense_parameters_bias_,
        )
        layer_output_1 = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_bottleneck_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_bottleneck_modules_dense_parameters_bias_ = (None)
        layer_outputs_9 = torch.nn.functional.dropout(
            layer_outputs_8, 0.1, False, False
        )
        layer_outputs_8 = None
        add_16 = layer_outputs_9 + embeddings_2
        layer_outputs_9 = embeddings_2 = None
        mul_9 = (
            add_16
            * l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_bottleneck_modules_layer_norm_parameters_weight_
        )
        add_16 = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_bottleneck_modules_layer_norm_parameters_weight_ = (None)
        layer_outputs_10 = (
            mul_9
            + l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_bottleneck_modules_layer_norm_parameters_bias_
        )
        mul_9 = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_bottleneck_modules_layer_norm_parameters_bias_ = (None)
        tensor = torch.tensor(1000)
        tensor = None
        layer_input_4 = torch._C._nn.linear(
            layer_outputs_10,
            l_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_input_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_input_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_input_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_input_modules_dense_parameters_bias_ = (None)
        mul_10 = (
            layer_input_4
            * l_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_input_modules_layer_norm_parameters_weight_
        )
        layer_input_4 = l_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_input_modules_layer_norm_parameters_weight_ = (None)
        layer_input_5 = (
            mul_10
            + l_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_input_modules_layer_norm_parameters_bias_
        )
        mul_10 = l_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_input_modules_layer_norm_parameters_bias_ = (None)
        layer_input_6 = torch._C._nn.linear(
            layer_outputs_10,
            l_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_attention_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_attention_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_attention_modules_dense_parameters_bias_ = (None)
        mul_11 = (
            layer_input_6
            * l_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_attention_modules_layer_norm_parameters_weight_
        )
        layer_input_6 = l_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_attention_modules_layer_norm_parameters_weight_ = (None)
        layer_input_7 = (
            mul_11
            + l_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_attention_modules_layer_norm_parameters_bias_
        )
        mul_11 = l_self_modules_encoder_modules_layer_modules_1_modules_bottleneck_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_18 = torch._C._nn.linear(
            layer_input_7,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_4 = linear_18.view(1, -1, 4, 32)
        linear_18 = None
        query_layer_1 = view_4.transpose(1, 2)
        view_4 = None
        linear_19 = torch._C._nn.linear(
            layer_input_7,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        layer_input_7 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_5 = linear_19.view(1, -1, 4, 32)
        linear_19 = None
        key_layer_1 = view_5.transpose(1, 2)
        view_5 = None
        linear_20 = torch._C._nn.linear(
            layer_outputs_10,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_6 = linear_20.view(1, -1, 4, 32)
        linear_20 = None
        value_layer_1 = view_6.transpose(1, 2)
        view_6 = None
        transpose_7 = key_layer_1.transpose(-1, -2)
        key_layer_1 = None
        attention_scores_3 = torch.matmul(query_layer_1, transpose_7)
        query_layer_1 = transpose_7 = None
        attention_scores_4 = attention_scores_3 / 5.656854249492381
        attention_scores_3 = None
        attention_scores_5 = attention_scores_4 + extended_attention_mask_2
        attention_scores_4 = None
        attention_probs_2 = torch.nn.functional.softmax(attention_scores_5, dim=-1)
        attention_scores_5 = None
        attention_probs_3 = torch.nn.functional.dropout(
            attention_probs_2, 0.1, False, False
        )
        attention_probs_2 = None
        context_layer_3 = torch.matmul(attention_probs_3, value_layer_1)
        attention_probs_3 = value_layer_1 = None
        permute_1 = context_layer_3.permute(0, 2, 1, 3)
        context_layer_3 = None
        context_layer_4 = permute_1.contiguous()
        permute_1 = None
        context_layer_5 = context_layer_4.view((1, 45, 128))
        context_layer_4 = None
        layer_outputs_11 = torch._C._nn.linear(
            context_layer_5,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_5 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        add_21 = layer_outputs_11 + layer_input_5
        layer_outputs_11 = layer_input_5 = None
        mul_12 = (
            add_21
            * l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_weight_
        )
        add_21 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = (None)
        layer_outputs_12 = (
            mul_12
            + l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_12 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_8 = torch._C._nn.linear(
            layer_outputs_12,
            l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_9 = torch._C._nn.gelu(hidden_states_8)
        hidden_states_8 = None
        layer_outputs_13 = torch._C._nn.linear(
            hidden_states_9,
            l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_9 = l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_output_modules_dense_parameters_bias_ = (None)
        add_23 = layer_outputs_13 + layer_outputs_12
        layer_outputs_13 = layer_outputs_12 = None
        mul_13 = (
            add_23
            * l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_weight_
        )
        add_23 = l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_weight_ = (None)
        layer_outputs_14 = (
            mul_13
            + l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_13 = l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_10 = torch._C._nn.linear(
            layer_outputs_14,
            l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_11 = torch._C._nn.gelu(hidden_states_10)
        hidden_states_10 = None
        layer_outputs_15 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_11 = l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_output_modules_dense_parameters_bias_ = (None)
        add_25 = layer_outputs_15 + layer_outputs_14
        layer_outputs_15 = layer_outputs_14 = None
        mul_14 = (
            add_25
            * l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_weight_
        )
        add_25 = l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_weight_ = (None)
        layer_outputs_16 = (
            mul_14
            + l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_14 = l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_12 = torch._C._nn.linear(
            layer_outputs_16,
            l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_13 = torch._C._nn.gelu(hidden_states_12)
        hidden_states_12 = None
        layer_outputs_17 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_13 = l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_output_modules_dense_parameters_bias_ = (None)
        add_27 = layer_outputs_17 + layer_outputs_16
        layer_outputs_17 = layer_outputs_16 = None
        mul_15 = (
            add_27
            * l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_weight_
        )
        add_27 = l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_weight_ = (None)
        layer_outputs_18 = (
            mul_15
            + l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_15 = l_self_modules_encoder_modules_layer_modules_1_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_14 = torch._C._nn.linear(
            layer_outputs_18,
            l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_15 = torch._C._nn.gelu(hidden_states_14)
        hidden_states_14 = None
        layer_output_2 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_15 = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_ = (None)
        add_29 = layer_output_2 + layer_outputs_18
        layer_output_2 = layer_outputs_18 = None
        mul_16 = (
            add_29
            * l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_weight_
        )
        add_29 = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_weight_ = (None)
        layer_output_3 = (
            mul_16
            + l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_16 = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_bias_ = (None)
        layer_outputs_19 = torch._C._nn.linear(
            layer_output_3,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_bottleneck_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_bottleneck_modules_dense_parameters_bias_,
        )
        layer_output_3 = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_bottleneck_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_bottleneck_modules_dense_parameters_bias_ = (None)
        layer_outputs_20 = torch.nn.functional.dropout(
            layer_outputs_19, 0.1, False, False
        )
        layer_outputs_19 = None
        add_31 = layer_outputs_20 + layer_outputs_10
        layer_outputs_20 = layer_outputs_10 = None
        mul_17 = (
            add_31
            * l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_bottleneck_modules_layer_norm_parameters_weight_
        )
        add_31 = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_bottleneck_modules_layer_norm_parameters_weight_ = (None)
        layer_outputs_21 = (
            mul_17
            + l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_bottleneck_modules_layer_norm_parameters_bias_
        )
        mul_17 = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_bottleneck_modules_layer_norm_parameters_bias_ = (None)
        tensor_1 = torch.tensor(1000)
        tensor_1 = None
        layer_input_8 = torch._C._nn.linear(
            layer_outputs_21,
            l_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_input_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_input_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_input_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_input_modules_dense_parameters_bias_ = (None)
        mul_18 = (
            layer_input_8
            * l_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_input_modules_layer_norm_parameters_weight_
        )
        layer_input_8 = l_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_input_modules_layer_norm_parameters_weight_ = (None)
        layer_input_9 = (
            mul_18
            + l_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_input_modules_layer_norm_parameters_bias_
        )
        mul_18 = l_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_input_modules_layer_norm_parameters_bias_ = (None)
        layer_input_10 = torch._C._nn.linear(
            layer_outputs_21,
            l_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_attention_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_attention_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_attention_modules_dense_parameters_bias_ = (None)
        mul_19 = (
            layer_input_10
            * l_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_attention_modules_layer_norm_parameters_weight_
        )
        layer_input_10 = l_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_attention_modules_layer_norm_parameters_weight_ = (None)
        layer_input_11 = (
            mul_19
            + l_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_attention_modules_layer_norm_parameters_bias_
        )
        mul_19 = l_self_modules_encoder_modules_layer_modules_2_modules_bottleneck_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_33 = torch._C._nn.linear(
            layer_input_11,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_8 = linear_33.view(1, -1, 4, 32)
        linear_33 = None
        query_layer_2 = view_8.transpose(1, 2)
        view_8 = None
        linear_34 = torch._C._nn.linear(
            layer_input_11,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        layer_input_11 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_9 = linear_34.view(1, -1, 4, 32)
        linear_34 = None
        key_layer_2 = view_9.transpose(1, 2)
        view_9 = None
        linear_35 = torch._C._nn.linear(
            layer_outputs_21,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_10 = linear_35.view(1, -1, 4, 32)
        linear_35 = None
        value_layer_2 = view_10.transpose(1, 2)
        view_10 = None
        transpose_11 = key_layer_2.transpose(-1, -2)
        key_layer_2 = None
        attention_scores_6 = torch.matmul(query_layer_2, transpose_11)
        query_layer_2 = transpose_11 = None
        attention_scores_7 = attention_scores_6 / 5.656854249492381
        attention_scores_6 = None
        attention_scores_8 = attention_scores_7 + extended_attention_mask_2
        attention_scores_7 = None
        attention_probs_4 = torch.nn.functional.softmax(attention_scores_8, dim=-1)
        attention_scores_8 = None
        attention_probs_5 = torch.nn.functional.dropout(
            attention_probs_4, 0.1, False, False
        )
        attention_probs_4 = None
        context_layer_6 = torch.matmul(attention_probs_5, value_layer_2)
        attention_probs_5 = value_layer_2 = None
        permute_2 = context_layer_6.permute(0, 2, 1, 3)
        context_layer_6 = None
        context_layer_7 = permute_2.contiguous()
        permute_2 = None
        context_layer_8 = context_layer_7.view((1, 45, 128))
        context_layer_7 = None
        layer_outputs_22 = torch._C._nn.linear(
            context_layer_8,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_8 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        add_36 = layer_outputs_22 + layer_input_9
        layer_outputs_22 = layer_input_9 = None
        mul_20 = (
            add_36
            * l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_weight_
        )
        add_36 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = (None)
        layer_outputs_23 = (
            mul_20
            + l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_20 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_16 = torch._C._nn.linear(
            layer_outputs_23,
            l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_17 = torch._C._nn.gelu(hidden_states_16)
        hidden_states_16 = None
        layer_outputs_24 = torch._C._nn.linear(
            hidden_states_17,
            l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_17 = l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_output_modules_dense_parameters_bias_ = (None)
        add_38 = layer_outputs_24 + layer_outputs_23
        layer_outputs_24 = layer_outputs_23 = None
        mul_21 = (
            add_38
            * l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_weight_
        )
        add_38 = l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_weight_ = (None)
        layer_outputs_25 = (
            mul_21
            + l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_21 = l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_18 = torch._C._nn.linear(
            layer_outputs_25,
            l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_19 = torch._C._nn.gelu(hidden_states_18)
        hidden_states_18 = None
        layer_outputs_26 = torch._C._nn.linear(
            hidden_states_19,
            l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_19 = l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_output_modules_dense_parameters_bias_ = (None)
        add_40 = layer_outputs_26 + layer_outputs_25
        layer_outputs_26 = layer_outputs_25 = None
        mul_22 = (
            add_40
            * l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_weight_
        )
        add_40 = l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_weight_ = (None)
        layer_outputs_27 = (
            mul_22
            + l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_22 = l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_20 = torch._C._nn.linear(
            layer_outputs_27,
            l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_21 = torch._C._nn.gelu(hidden_states_20)
        hidden_states_20 = None
        layer_outputs_28 = torch._C._nn.linear(
            hidden_states_21,
            l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_21 = l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_output_modules_dense_parameters_bias_ = (None)
        add_42 = layer_outputs_28 + layer_outputs_27
        layer_outputs_28 = layer_outputs_27 = None
        mul_23 = (
            add_42
            * l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_weight_
        )
        add_42 = l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_weight_ = (None)
        layer_outputs_29 = (
            mul_23
            + l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_23 = l_self_modules_encoder_modules_layer_modules_2_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_22 = torch._C._nn.linear(
            layer_outputs_29,
            l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_23 = torch._C._nn.gelu(hidden_states_22)
        hidden_states_22 = None
        layer_output_4 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_23 = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_ = (None)
        add_44 = layer_output_4 + layer_outputs_29
        layer_output_4 = layer_outputs_29 = None
        mul_24 = (
            add_44
            * l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_weight_
        )
        add_44 = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_weight_ = (None)
        layer_output_5 = (
            mul_24
            + l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_24 = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_bias_ = (None)
        layer_outputs_30 = torch._C._nn.linear(
            layer_output_5,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_bottleneck_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_bottleneck_modules_dense_parameters_bias_,
        )
        layer_output_5 = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_bottleneck_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_bottleneck_modules_dense_parameters_bias_ = (None)
        layer_outputs_31 = torch.nn.functional.dropout(
            layer_outputs_30, 0.1, False, False
        )
        layer_outputs_30 = None
        add_46 = layer_outputs_31 + layer_outputs_21
        layer_outputs_31 = layer_outputs_21 = None
        mul_25 = (
            add_46
            * l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_bottleneck_modules_layer_norm_parameters_weight_
        )
        add_46 = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_bottleneck_modules_layer_norm_parameters_weight_ = (None)
        layer_outputs_32 = (
            mul_25
            + l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_bottleneck_modules_layer_norm_parameters_bias_
        )
        mul_25 = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_bottleneck_modules_layer_norm_parameters_bias_ = (None)
        tensor_2 = torch.tensor(1000)
        tensor_2 = None
        layer_input_12 = torch._C._nn.linear(
            layer_outputs_32,
            l_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_input_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_input_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_input_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_input_modules_dense_parameters_bias_ = (None)
        mul_26 = (
            layer_input_12
            * l_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_input_modules_layer_norm_parameters_weight_
        )
        layer_input_12 = l_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_input_modules_layer_norm_parameters_weight_ = (None)
        layer_input_13 = (
            mul_26
            + l_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_input_modules_layer_norm_parameters_bias_
        )
        mul_26 = l_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_input_modules_layer_norm_parameters_bias_ = (None)
        layer_input_14 = torch._C._nn.linear(
            layer_outputs_32,
            l_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_attention_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_attention_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_attention_modules_dense_parameters_bias_ = (None)
        mul_27 = (
            layer_input_14
            * l_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_attention_modules_layer_norm_parameters_weight_
        )
        layer_input_14 = l_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_attention_modules_layer_norm_parameters_weight_ = (None)
        layer_input_15 = (
            mul_27
            + l_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_attention_modules_layer_norm_parameters_bias_
        )
        mul_27 = l_self_modules_encoder_modules_layer_modules_3_modules_bottleneck_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_48 = torch._C._nn.linear(
            layer_input_15,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_12 = linear_48.view(1, -1, 4, 32)
        linear_48 = None
        query_layer_3 = view_12.transpose(1, 2)
        view_12 = None
        linear_49 = torch._C._nn.linear(
            layer_input_15,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        layer_input_15 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_13 = linear_49.view(1, -1, 4, 32)
        linear_49 = None
        key_layer_3 = view_13.transpose(1, 2)
        view_13 = None
        linear_50 = torch._C._nn.linear(
            layer_outputs_32,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_14 = linear_50.view(1, -1, 4, 32)
        linear_50 = None
        value_layer_3 = view_14.transpose(1, 2)
        view_14 = None
        transpose_15 = key_layer_3.transpose(-1, -2)
        key_layer_3 = None
        attention_scores_9 = torch.matmul(query_layer_3, transpose_15)
        query_layer_3 = transpose_15 = None
        attention_scores_10 = attention_scores_9 / 5.656854249492381
        attention_scores_9 = None
        attention_scores_11 = attention_scores_10 + extended_attention_mask_2
        attention_scores_10 = None
        attention_probs_6 = torch.nn.functional.softmax(attention_scores_11, dim=-1)
        attention_scores_11 = None
        attention_probs_7 = torch.nn.functional.dropout(
            attention_probs_6, 0.1, False, False
        )
        attention_probs_6 = None
        context_layer_9 = torch.matmul(attention_probs_7, value_layer_3)
        attention_probs_7 = value_layer_3 = None
        permute_3 = context_layer_9.permute(0, 2, 1, 3)
        context_layer_9 = None
        context_layer_10 = permute_3.contiguous()
        permute_3 = None
        context_layer_11 = context_layer_10.view((1, 45, 128))
        context_layer_10 = None
        layer_outputs_33 = torch._C._nn.linear(
            context_layer_11,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_11 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        add_51 = layer_outputs_33 + layer_input_13
        layer_outputs_33 = layer_input_13 = None
        mul_28 = (
            add_51
            * l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_weight_
        )
        add_51 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = (None)
        layer_outputs_34 = (
            mul_28
            + l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_28 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_24 = torch._C._nn.linear(
            layer_outputs_34,
            l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_25 = torch._C._nn.gelu(hidden_states_24)
        hidden_states_24 = None
        layer_outputs_35 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_25 = l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_output_modules_dense_parameters_bias_ = (None)
        add_53 = layer_outputs_35 + layer_outputs_34
        layer_outputs_35 = layer_outputs_34 = None
        mul_29 = (
            add_53
            * l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_weight_
        )
        add_53 = l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_weight_ = (None)
        layer_outputs_36 = (
            mul_29
            + l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_29 = l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_26 = torch._C._nn.linear(
            layer_outputs_36,
            l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_27 = torch._C._nn.gelu(hidden_states_26)
        hidden_states_26 = None
        layer_outputs_37 = torch._C._nn.linear(
            hidden_states_27,
            l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_27 = l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_output_modules_dense_parameters_bias_ = (None)
        add_55 = layer_outputs_37 + layer_outputs_36
        layer_outputs_37 = layer_outputs_36 = None
        mul_30 = (
            add_55
            * l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_weight_
        )
        add_55 = l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_weight_ = (None)
        layer_outputs_38 = (
            mul_30
            + l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_30 = l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_28 = torch._C._nn.linear(
            layer_outputs_38,
            l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_29 = torch._C._nn.gelu(hidden_states_28)
        hidden_states_28 = None
        layer_outputs_39 = torch._C._nn.linear(
            hidden_states_29,
            l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_29 = l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_output_modules_dense_parameters_bias_ = (None)
        add_57 = layer_outputs_39 + layer_outputs_38
        layer_outputs_39 = layer_outputs_38 = None
        mul_31 = (
            add_57
            * l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_weight_
        )
        add_57 = l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_weight_ = (None)
        layer_outputs_40 = (
            mul_31
            + l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_31 = l_self_modules_encoder_modules_layer_modules_3_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_30 = torch._C._nn.linear(
            layer_outputs_40,
            l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_31 = torch._C._nn.gelu(hidden_states_30)
        hidden_states_30 = None
        layer_output_6 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_31 = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_ = (None)
        add_59 = layer_output_6 + layer_outputs_40
        layer_output_6 = layer_outputs_40 = None
        mul_32 = (
            add_59
            * l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_weight_
        )
        add_59 = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_weight_ = (None)
        layer_output_7 = (
            mul_32
            + l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_32 = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_bias_ = (None)
        layer_outputs_41 = torch._C._nn.linear(
            layer_output_7,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_bottleneck_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_bottleneck_modules_dense_parameters_bias_,
        )
        layer_output_7 = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_bottleneck_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_bottleneck_modules_dense_parameters_bias_ = (None)
        layer_outputs_42 = torch.nn.functional.dropout(
            layer_outputs_41, 0.1, False, False
        )
        layer_outputs_41 = None
        add_61 = layer_outputs_42 + layer_outputs_32
        layer_outputs_42 = layer_outputs_32 = None
        mul_33 = (
            add_61
            * l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_bottleneck_modules_layer_norm_parameters_weight_
        )
        add_61 = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_bottleneck_modules_layer_norm_parameters_weight_ = (None)
        layer_outputs_43 = (
            mul_33
            + l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_bottleneck_modules_layer_norm_parameters_bias_
        )
        mul_33 = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_bottleneck_modules_layer_norm_parameters_bias_ = (None)
        tensor_3 = torch.tensor(1000)
        tensor_3 = None
        layer_input_16 = torch._C._nn.linear(
            layer_outputs_43,
            l_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_input_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_input_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_input_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_input_modules_dense_parameters_bias_ = (None)
        mul_34 = (
            layer_input_16
            * l_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_input_modules_layer_norm_parameters_weight_
        )
        layer_input_16 = l_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_input_modules_layer_norm_parameters_weight_ = (None)
        layer_input_17 = (
            mul_34
            + l_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_input_modules_layer_norm_parameters_bias_
        )
        mul_34 = l_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_input_modules_layer_norm_parameters_bias_ = (None)
        layer_input_18 = torch._C._nn.linear(
            layer_outputs_43,
            l_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_attention_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_attention_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_attention_modules_dense_parameters_bias_ = (None)
        mul_35 = (
            layer_input_18
            * l_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_attention_modules_layer_norm_parameters_weight_
        )
        layer_input_18 = l_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_attention_modules_layer_norm_parameters_weight_ = (None)
        layer_input_19 = (
            mul_35
            + l_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_attention_modules_layer_norm_parameters_bias_
        )
        mul_35 = l_self_modules_encoder_modules_layer_modules_4_modules_bottleneck_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_63 = torch._C._nn.linear(
            layer_input_19,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_16 = linear_63.view(1, -1, 4, 32)
        linear_63 = None
        query_layer_4 = view_16.transpose(1, 2)
        view_16 = None
        linear_64 = torch._C._nn.linear(
            layer_input_19,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        layer_input_19 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_17 = linear_64.view(1, -1, 4, 32)
        linear_64 = None
        key_layer_4 = view_17.transpose(1, 2)
        view_17 = None
        linear_65 = torch._C._nn.linear(
            layer_outputs_43,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_18 = linear_65.view(1, -1, 4, 32)
        linear_65 = None
        value_layer_4 = view_18.transpose(1, 2)
        view_18 = None
        transpose_19 = key_layer_4.transpose(-1, -2)
        key_layer_4 = None
        attention_scores_12 = torch.matmul(query_layer_4, transpose_19)
        query_layer_4 = transpose_19 = None
        attention_scores_13 = attention_scores_12 / 5.656854249492381
        attention_scores_12 = None
        attention_scores_14 = attention_scores_13 + extended_attention_mask_2
        attention_scores_13 = extended_attention_mask_2 = None
        attention_probs_8 = torch.nn.functional.softmax(attention_scores_14, dim=-1)
        attention_scores_14 = None
        attention_probs_9 = torch.nn.functional.dropout(
            attention_probs_8, 0.1, False, False
        )
        attention_probs_8 = None
        context_layer_12 = torch.matmul(attention_probs_9, value_layer_4)
        attention_probs_9 = value_layer_4 = None
        permute_4 = context_layer_12.permute(0, 2, 1, 3)
        context_layer_12 = None
        context_layer_13 = permute_4.contiguous()
        permute_4 = None
        context_layer_14 = context_layer_13.view((1, 45, 128))
        context_layer_13 = None
        layer_outputs_44 = torch._C._nn.linear(
            context_layer_14,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_14 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        add_66 = layer_outputs_44 + layer_input_17
        layer_outputs_44 = layer_input_17 = None
        mul_36 = (
            add_66
            * l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_weight_
        )
        add_66 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = (None)
        layer_outputs_45 = (
            mul_36
            + l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_36 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_32 = torch._C._nn.linear(
            layer_outputs_45,
            l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_33 = torch._C._nn.gelu(hidden_states_32)
        hidden_states_32 = None
        layer_outputs_46 = torch._C._nn.linear(
            hidden_states_33,
            l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_33 = l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_output_modules_dense_parameters_bias_ = (None)
        add_68 = layer_outputs_46 + layer_outputs_45
        layer_outputs_46 = layer_outputs_45 = None
        mul_37 = (
            add_68
            * l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_weight_
        )
        add_68 = l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_weight_ = (None)
        layer_outputs_47 = (
            mul_37
            + l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_37 = l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_0_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_34 = torch._C._nn.linear(
            layer_outputs_47,
            l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_35 = torch._C._nn.gelu(hidden_states_34)
        hidden_states_34 = None
        layer_outputs_48 = torch._C._nn.linear(
            hidden_states_35,
            l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_35 = l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_output_modules_dense_parameters_bias_ = (None)
        add_70 = layer_outputs_48 + layer_outputs_47
        layer_outputs_48 = layer_outputs_47 = None
        mul_38 = (
            add_70
            * l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_weight_
        )
        add_70 = l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_weight_ = (None)
        layer_outputs_49 = (
            mul_38
            + l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_38 = l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_1_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_36 = torch._C._nn.linear(
            layer_outputs_49,
            l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_37 = torch._C._nn.gelu(hidden_states_36)
        hidden_states_36 = None
        layer_outputs_50 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_37 = l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_output_modules_dense_parameters_bias_ = (None)
        add_72 = layer_outputs_50 + layer_outputs_49
        layer_outputs_50 = layer_outputs_49 = None
        mul_39 = (
            add_72
            * l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_weight_
        )
        add_72 = l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_weight_ = (None)
        layer_outputs_51 = (
            mul_39
            + l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_39 = l_self_modules_encoder_modules_layer_modules_4_modules_ffn_modules_2_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_38 = torch._C._nn.linear(
            layer_outputs_51,
            l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_39 = torch._C._nn.gelu(hidden_states_38)
        hidden_states_38 = None
        layer_output_8 = torch._C._nn.linear(
            hidden_states_39,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_39 = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_ = (None)
        add_74 = layer_output_8 + layer_outputs_51
        layer_output_8 = layer_outputs_51 = None
        mul_40 = (
            add_74
            * l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_weight_
        )
        add_74 = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_weight_ = (None)
        layer_output_9 = (
            mul_40
            + l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_40 = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_bias_ = (None)
        layer_outputs_52 = torch._C._nn.linear(
            layer_output_9,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_bottleneck_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_bottleneck_modules_dense_parameters_bias_,
        )
        layer_output_9 = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_bottleneck_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_bottleneck_modules_dense_parameters_bias_ = (None)
        layer_outputs_53 = torch.nn.functional.dropout(
            layer_outputs_52, 0.1, False, False
        )
        layer_outputs_52 = None
        add_76 = layer_outputs_53 + layer_outputs_43
        layer_outputs_53 = layer_outputs_43 = None
        mul_41 = (
            add_76
            * l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_bottleneck_modules_layer_norm_parameters_weight_
        )
        add_76 = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_bottleneck_modules_layer_norm_parameters_weight_ = (None)
        layer_outputs_54 = (
            mul_41
            + l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_bottleneck_modules_layer_norm_parameters_bias_
        )
        mul_41 = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_bottleneck_modules_layer_norm_parameters_bias_ = (None)
        tensor_4 = torch.tensor(1000)
        tensor_4 = None
        first_token_tensor = layer_outputs_54[(slice(None, None, None), 0)]
        pooled_output = torch._C._nn.linear(
            first_token_tensor,
            l_self_modules_pooler_modules_dense_parameters_weight_,
            l_self_modules_pooler_modules_dense_parameters_bias_,
        )
        first_token_tensor = (
            l_self_modules_pooler_modules_dense_parameters_weight_
        ) = l_self_modules_pooler_modules_dense_parameters_bias_ = None
        pooled_output_1 = torch.tanh(pooled_output)
        pooled_output = None
        return (layer_outputs_54, pooled_output_1)
