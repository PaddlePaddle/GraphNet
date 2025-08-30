import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_attention_mask_: torch.Tensor,
        L_token_type_ids_: torch.Tensor,
        L_self_modules_embeddings_modules_word_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_: torch.Tensor,
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
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_: torch.Tensor,
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
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_: torch.Tensor,
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
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_: torch.Tensor,
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
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_: torch.Tensor,
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
        L_self_modules_pooler_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_pooler_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_input_ids_ = L_input_ids_
        l_attention_mask_ = L_attention_mask_
        l_token_type_ids_ = L_token_type_ids_
        l_self_modules_embeddings_modules_word_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_word_embeddings_parameters_weight_
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
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_
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
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_
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
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_
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
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_
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
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_
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
        embeddings = inputs_embeds + token_type_embeddings
        inputs_embeds = token_type_embeddings = None
        embeddings_1 = torch.nn.functional.layer_norm(
            embeddings,
            (32,),
            l_self_modules_embeddings_modules_layer_norm_parameters_weight_,
            l_self_modules_embeddings_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        embeddings = (
            l_self_modules_embeddings_modules_layer_norm_parameters_weight_
        ) = l_self_modules_embeddings_modules_layer_norm_parameters_bias_ = None
        embeddings_2 = torch.nn.functional.dropout(embeddings_1, 0.1, False, False)
        embeddings_1 = None
        mixed_query_layer = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        linear_1 = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        x = linear_1.view((1, 45, 4, 8))
        linear_1 = None
        key_layer = x.permute(0, 2, 1, 3)
        x = None
        linear_2 = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        x_1 = linear_2.view((1, 45, 4, 8))
        linear_2 = None
        value_layer = x_1.permute(0, 2, 1, 3)
        x_1 = None
        x_2 = mixed_query_layer.view((1, 45, 4, 8))
        mixed_query_layer = None
        query_layer = x_2.permute(0, 2, 1, 3)
        x_2 = None
        transpose = key_layer.transpose(-1, -2)
        key_layer = None
        attention_scores = torch.matmul(query_layer, transpose)
        transpose = None
        relations_keys = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_[
            (slice(None, 45, None), slice(None, 45, None), slice(None, None, None))
        ]
        query_layer_t = query_layer.permute(2, 0, 1, 3)
        query_layer = None
        contiguous = query_layer_t.contiguous()
        query_layer_t = None
        query_layer_r = contiguous.view(45, 4, 8)
        contiguous = None
        permute_4 = relations_keys.permute(0, 2, 1)
        relations_keys = None
        key_position_scores = torch.matmul(query_layer_r, permute_4)
        query_layer_r = permute_4 = None
        key_position_scores_r = key_position_scores.view(45, 1, 4, 45)
        key_position_scores = None
        key_position_scores_r_t = key_position_scores_r.permute(1, 2, 0, 3)
        key_position_scores_r = None
        attention_scores_1 = attention_scores + key_position_scores_r_t
        attention_scores = key_position_scores_r_t = None
        attention_scores_2 = attention_scores_1 / 2.8284271247461903
        attention_scores_1 = None
        attention_scores_3 = attention_scores_2 + extended_attention_mask_2
        attention_scores_2 = None
        attention_probs = torch.nn.functional.softmax(attention_scores_3, dim=-1)
        attention_scores_3 = None
        attention_probs_1 = torch.nn.functional.dropout(
            attention_probs, 0.1, False, False
        )
        attention_probs = None
        context_layer = torch.matmul(attention_probs_1, value_layer)
        value_layer = None
        relations_values = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_[
            (slice(None, 45, None), slice(None, 45, None), slice(None, None, None))
        ]
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_ = (
            None
        )
        attention_probs_t = attention_probs_1.permute(2, 0, 1, 3)
        attention_probs_1 = None
        contiguous_1 = attention_probs_t.contiguous()
        attention_probs_t = None
        attentions_probs_r = contiguous_1.view(45, 4, 45)
        contiguous_1 = None
        value_position_scores = torch.matmul(attentions_probs_r, relations_values)
        attentions_probs_r = relations_values = None
        value_position_scores_r = value_position_scores.view(45, 1, 4, 8)
        value_position_scores = None
        value_position_scores_r_t = value_position_scores_r.permute(1, 2, 0, 3)
        value_position_scores_r = None
        context_layer_1 = context_layer + value_position_scores_r_t
        context_layer = value_position_scores_r_t = None
        permute_8 = context_layer_1.permute(0, 2, 1, 3)
        context_layer_1 = None
        context_layer_2 = permute_8.contiguous()
        permute_8 = None
        context_layer_3 = context_layer_2.view((1, 45, 32))
        context_layer_2 = None
        hidden_states = torch._C._nn.linear(
            context_layer_3,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_3 = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_1 = torch.nn.functional.dropout(hidden_states, 0.1, False, False)
        hidden_states = None
        add_4 = hidden_states_1 + embeddings_2
        hidden_states_1 = embeddings_2 = None
        hidden_states_2 = torch.nn.functional.layer_norm(
            add_4,
            (32,),
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_4 = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
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
        add_5 = hidden_states_6 + hidden_states_2
        hidden_states_6 = hidden_states_2 = None
        hidden_states_7 = torch.nn.functional.layer_norm(
            add_5,
            (32,),
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_5 = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_bias_ = (None)
        mixed_query_layer_1 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        linear_7 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        x_3 = linear_7.view((1, 45, 4, 8))
        linear_7 = None
        key_layer_1 = x_3.permute(0, 2, 1, 3)
        x_3 = None
        linear_8 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        x_4 = linear_8.view((1, 45, 4, 8))
        linear_8 = None
        value_layer_1 = x_4.permute(0, 2, 1, 3)
        x_4 = None
        x_5 = mixed_query_layer_1.view((1, 45, 4, 8))
        mixed_query_layer_1 = None
        query_layer_1 = x_5.permute(0, 2, 1, 3)
        x_5 = None
        transpose_1 = key_layer_1.transpose(-1, -2)
        key_layer_1 = None
        attention_scores_4 = torch.matmul(query_layer_1, transpose_1)
        transpose_1 = None
        relations_keys_1 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_[
            (slice(None, 45, None), slice(None, 45, None), slice(None, None, None))
        ]
        query_layer_t_1 = query_layer_1.permute(2, 0, 1, 3)
        query_layer_1 = None
        contiguous_3 = query_layer_t_1.contiguous()
        query_layer_t_1 = None
        query_layer_r_1 = contiguous_3.view(45, 4, 8)
        contiguous_3 = None
        permute_13 = relations_keys_1.permute(0, 2, 1)
        relations_keys_1 = None
        key_position_scores_1 = torch.matmul(query_layer_r_1, permute_13)
        query_layer_r_1 = permute_13 = None
        key_position_scores_r_1 = key_position_scores_1.view(45, 1, 4, 45)
        key_position_scores_1 = None
        key_position_scores_r_t_1 = key_position_scores_r_1.permute(1, 2, 0, 3)
        key_position_scores_r_1 = None
        attention_scores_5 = attention_scores_4 + key_position_scores_r_t_1
        attention_scores_4 = key_position_scores_r_t_1 = None
        attention_scores_6 = attention_scores_5 / 2.8284271247461903
        attention_scores_5 = None
        attention_scores_7 = attention_scores_6 + extended_attention_mask_2
        attention_scores_6 = None
        attention_probs_2 = torch.nn.functional.softmax(attention_scores_7, dim=-1)
        attention_scores_7 = None
        attention_probs_3 = torch.nn.functional.dropout(
            attention_probs_2, 0.1, False, False
        )
        attention_probs_2 = None
        context_layer_4 = torch.matmul(attention_probs_3, value_layer_1)
        value_layer_1 = None
        relations_values_1 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_[
            (slice(None, 45, None), slice(None, 45, None), slice(None, None, None))
        ]
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_ = (
            None
        )
        attention_probs_t_1 = attention_probs_3.permute(2, 0, 1, 3)
        attention_probs_3 = None
        contiguous_4 = attention_probs_t_1.contiguous()
        attention_probs_t_1 = None
        attentions_probs_r_1 = contiguous_4.view(45, 4, 45)
        contiguous_4 = None
        value_position_scores_1 = torch.matmul(attentions_probs_r_1, relations_values_1)
        attentions_probs_r_1 = relations_values_1 = None
        value_position_scores_r_1 = value_position_scores_1.view(45, 1, 4, 8)
        value_position_scores_1 = None
        value_position_scores_r_t_1 = value_position_scores_r_1.permute(1, 2, 0, 3)
        value_position_scores_r_1 = None
        context_layer_5 = context_layer_4 + value_position_scores_r_t_1
        context_layer_4 = value_position_scores_r_t_1 = None
        permute_17 = context_layer_5.permute(0, 2, 1, 3)
        context_layer_5 = None
        context_layer_6 = permute_17.contiguous()
        permute_17 = None
        context_layer_7 = context_layer_6.view((1, 45, 32))
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
        add_9 = hidden_states_9 + hidden_states_7
        hidden_states_9 = hidden_states_7 = None
        hidden_states_10 = torch.nn.functional.layer_norm(
            add_9,
            (32,),
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_9 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
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
        add_10 = hidden_states_14 + hidden_states_10
        hidden_states_14 = hidden_states_10 = None
        hidden_states_15 = torch.nn.functional.layer_norm(
            add_10,
            (32,),
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_10 = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_bias_ = (None)
        mixed_query_layer_2 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        linear_13 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        x_6 = linear_13.view((1, 45, 4, 8))
        linear_13 = None
        key_layer_2 = x_6.permute(0, 2, 1, 3)
        x_6 = None
        linear_14 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        x_7 = linear_14.view((1, 45, 4, 8))
        linear_14 = None
        value_layer_2 = x_7.permute(0, 2, 1, 3)
        x_7 = None
        x_8 = mixed_query_layer_2.view((1, 45, 4, 8))
        mixed_query_layer_2 = None
        query_layer_2 = x_8.permute(0, 2, 1, 3)
        x_8 = None
        transpose_2 = key_layer_2.transpose(-1, -2)
        key_layer_2 = None
        attention_scores_8 = torch.matmul(query_layer_2, transpose_2)
        transpose_2 = None
        relations_keys_2 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_[
            (slice(None, 45, None), slice(None, 45, None), slice(None, None, None))
        ]
        query_layer_t_2 = query_layer_2.permute(2, 0, 1, 3)
        query_layer_2 = None
        contiguous_6 = query_layer_t_2.contiguous()
        query_layer_t_2 = None
        query_layer_r_2 = contiguous_6.view(45, 4, 8)
        contiguous_6 = None
        permute_22 = relations_keys_2.permute(0, 2, 1)
        relations_keys_2 = None
        key_position_scores_2 = torch.matmul(query_layer_r_2, permute_22)
        query_layer_r_2 = permute_22 = None
        key_position_scores_r_2 = key_position_scores_2.view(45, 1, 4, 45)
        key_position_scores_2 = None
        key_position_scores_r_t_2 = key_position_scores_r_2.permute(1, 2, 0, 3)
        key_position_scores_r_2 = None
        attention_scores_9 = attention_scores_8 + key_position_scores_r_t_2
        attention_scores_8 = key_position_scores_r_t_2 = None
        attention_scores_10 = attention_scores_9 / 2.8284271247461903
        attention_scores_9 = None
        attention_scores_11 = attention_scores_10 + extended_attention_mask_2
        attention_scores_10 = None
        attention_probs_4 = torch.nn.functional.softmax(attention_scores_11, dim=-1)
        attention_scores_11 = None
        attention_probs_5 = torch.nn.functional.dropout(
            attention_probs_4, 0.1, False, False
        )
        attention_probs_4 = None
        context_layer_8 = torch.matmul(attention_probs_5, value_layer_2)
        value_layer_2 = None
        relations_values_2 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_[
            (slice(None, 45, None), slice(None, 45, None), slice(None, None, None))
        ]
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_ = (
            None
        )
        attention_probs_t_2 = attention_probs_5.permute(2, 0, 1, 3)
        attention_probs_5 = None
        contiguous_7 = attention_probs_t_2.contiguous()
        attention_probs_t_2 = None
        attentions_probs_r_2 = contiguous_7.view(45, 4, 45)
        contiguous_7 = None
        value_position_scores_2 = torch.matmul(attentions_probs_r_2, relations_values_2)
        attentions_probs_r_2 = relations_values_2 = None
        value_position_scores_r_2 = value_position_scores_2.view(45, 1, 4, 8)
        value_position_scores_2 = None
        value_position_scores_r_t_2 = value_position_scores_r_2.permute(1, 2, 0, 3)
        value_position_scores_r_2 = None
        context_layer_9 = context_layer_8 + value_position_scores_r_t_2
        context_layer_8 = value_position_scores_r_t_2 = None
        permute_26 = context_layer_9.permute(0, 2, 1, 3)
        context_layer_9 = None
        context_layer_10 = permute_26.contiguous()
        permute_26 = None
        context_layer_11 = context_layer_10.view((1, 45, 32))
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
        add_14 = hidden_states_17 + hidden_states_15
        hidden_states_17 = hidden_states_15 = None
        hidden_states_18 = torch.nn.functional.layer_norm(
            add_14,
            (32,),
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_14 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
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
        add_15 = hidden_states_22 + hidden_states_18
        hidden_states_22 = hidden_states_18 = None
        hidden_states_23 = torch.nn.functional.layer_norm(
            add_15,
            (32,),
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_15 = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_bias_ = (None)
        mixed_query_layer_3 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        linear_19 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        x_9 = linear_19.view((1, 45, 4, 8))
        linear_19 = None
        key_layer_3 = x_9.permute(0, 2, 1, 3)
        x_9 = None
        linear_20 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        x_10 = linear_20.view((1, 45, 4, 8))
        linear_20 = None
        value_layer_3 = x_10.permute(0, 2, 1, 3)
        x_10 = None
        x_11 = mixed_query_layer_3.view((1, 45, 4, 8))
        mixed_query_layer_3 = None
        query_layer_3 = x_11.permute(0, 2, 1, 3)
        x_11 = None
        transpose_3 = key_layer_3.transpose(-1, -2)
        key_layer_3 = None
        attention_scores_12 = torch.matmul(query_layer_3, transpose_3)
        transpose_3 = None
        relations_keys_3 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_[
            (slice(None, 45, None), slice(None, 45, None), slice(None, None, None))
        ]
        query_layer_t_3 = query_layer_3.permute(2, 0, 1, 3)
        query_layer_3 = None
        contiguous_9 = query_layer_t_3.contiguous()
        query_layer_t_3 = None
        query_layer_r_3 = contiguous_9.view(45, 4, 8)
        contiguous_9 = None
        permute_31 = relations_keys_3.permute(0, 2, 1)
        relations_keys_3 = None
        key_position_scores_3 = torch.matmul(query_layer_r_3, permute_31)
        query_layer_r_3 = permute_31 = None
        key_position_scores_r_3 = key_position_scores_3.view(45, 1, 4, 45)
        key_position_scores_3 = None
        key_position_scores_r_t_3 = key_position_scores_r_3.permute(1, 2, 0, 3)
        key_position_scores_r_3 = None
        attention_scores_13 = attention_scores_12 + key_position_scores_r_t_3
        attention_scores_12 = key_position_scores_r_t_3 = None
        attention_scores_14 = attention_scores_13 / 2.8284271247461903
        attention_scores_13 = None
        attention_scores_15 = attention_scores_14 + extended_attention_mask_2
        attention_scores_14 = None
        attention_probs_6 = torch.nn.functional.softmax(attention_scores_15, dim=-1)
        attention_scores_15 = None
        attention_probs_7 = torch.nn.functional.dropout(
            attention_probs_6, 0.1, False, False
        )
        attention_probs_6 = None
        context_layer_12 = torch.matmul(attention_probs_7, value_layer_3)
        value_layer_3 = None
        relations_values_3 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_[
            (slice(None, 45, None), slice(None, 45, None), slice(None, None, None))
        ]
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_ = (
            None
        )
        attention_probs_t_3 = attention_probs_7.permute(2, 0, 1, 3)
        attention_probs_7 = None
        contiguous_10 = attention_probs_t_3.contiguous()
        attention_probs_t_3 = None
        attentions_probs_r_3 = contiguous_10.view(45, 4, 45)
        contiguous_10 = None
        value_position_scores_3 = torch.matmul(attentions_probs_r_3, relations_values_3)
        attentions_probs_r_3 = relations_values_3 = None
        value_position_scores_r_3 = value_position_scores_3.view(45, 1, 4, 8)
        value_position_scores_3 = None
        value_position_scores_r_t_3 = value_position_scores_r_3.permute(1, 2, 0, 3)
        value_position_scores_r_3 = None
        context_layer_13 = context_layer_12 + value_position_scores_r_t_3
        context_layer_12 = value_position_scores_r_t_3 = None
        permute_35 = context_layer_13.permute(0, 2, 1, 3)
        context_layer_13 = None
        context_layer_14 = permute_35.contiguous()
        permute_35 = None
        context_layer_15 = context_layer_14.view((1, 45, 32))
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
        add_19 = hidden_states_25 + hidden_states_23
        hidden_states_25 = hidden_states_23 = None
        hidden_states_26 = torch.nn.functional.layer_norm(
            add_19,
            (32,),
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_19 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
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
        add_20 = hidden_states_30 + hidden_states_26
        hidden_states_30 = hidden_states_26 = None
        hidden_states_31 = torch.nn.functional.layer_norm(
            add_20,
            (32,),
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_20 = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_bias_ = (None)
        mixed_query_layer_4 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        linear_25 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        x_12 = linear_25.view((1, 45, 4, 8))
        linear_25 = None
        key_layer_4 = x_12.permute(0, 2, 1, 3)
        x_12 = None
        linear_26 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        x_13 = linear_26.view((1, 45, 4, 8))
        linear_26 = None
        value_layer_4 = x_13.permute(0, 2, 1, 3)
        x_13 = None
        x_14 = mixed_query_layer_4.view((1, 45, 4, 8))
        mixed_query_layer_4 = None
        query_layer_4 = x_14.permute(0, 2, 1, 3)
        x_14 = None
        transpose_4 = key_layer_4.transpose(-1, -2)
        key_layer_4 = None
        attention_scores_16 = torch.matmul(query_layer_4, transpose_4)
        transpose_4 = None
        relations_keys_4 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_[
            (slice(None, 45, None), slice(None, 45, None), slice(None, None, None))
        ]
        query_layer_t_4 = query_layer_4.permute(2, 0, 1, 3)
        query_layer_4 = None
        contiguous_12 = query_layer_t_4.contiguous()
        query_layer_t_4 = None
        query_layer_r_4 = contiguous_12.view(45, 4, 8)
        contiguous_12 = None
        permute_40 = relations_keys_4.permute(0, 2, 1)
        relations_keys_4 = None
        key_position_scores_4 = torch.matmul(query_layer_r_4, permute_40)
        query_layer_r_4 = permute_40 = None
        key_position_scores_r_4 = key_position_scores_4.view(45, 1, 4, 45)
        key_position_scores_4 = None
        key_position_scores_r_t_4 = key_position_scores_r_4.permute(1, 2, 0, 3)
        key_position_scores_r_4 = None
        attention_scores_17 = attention_scores_16 + key_position_scores_r_t_4
        attention_scores_16 = key_position_scores_r_t_4 = None
        attention_scores_18 = attention_scores_17 / 2.8284271247461903
        attention_scores_17 = None
        attention_scores_19 = attention_scores_18 + extended_attention_mask_2
        attention_scores_18 = extended_attention_mask_2 = None
        attention_probs_8 = torch.nn.functional.softmax(attention_scores_19, dim=-1)
        attention_scores_19 = None
        attention_probs_9 = torch.nn.functional.dropout(
            attention_probs_8, 0.1, False, False
        )
        attention_probs_8 = None
        context_layer_16 = torch.matmul(attention_probs_9, value_layer_4)
        value_layer_4 = None
        relations_values_4 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_[
            (slice(None, 45, None), slice(None, 45, None), slice(None, None, None))
        ]
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_relative_positions_encoding_buffers_positions_encoding_ = (
            None
        )
        attention_probs_t_4 = attention_probs_9.permute(2, 0, 1, 3)
        attention_probs_9 = None
        contiguous_13 = attention_probs_t_4.contiguous()
        attention_probs_t_4 = None
        attentions_probs_r_4 = contiguous_13.view(45, 4, 45)
        contiguous_13 = None
        value_position_scores_4 = torch.matmul(attentions_probs_r_4, relations_values_4)
        attentions_probs_r_4 = relations_values_4 = None
        value_position_scores_r_4 = value_position_scores_4.view(45, 1, 4, 8)
        value_position_scores_4 = None
        value_position_scores_r_t_4 = value_position_scores_r_4.permute(1, 2, 0, 3)
        value_position_scores_r_4 = None
        context_layer_17 = context_layer_16 + value_position_scores_r_t_4
        context_layer_16 = value_position_scores_r_t_4 = None
        permute_44 = context_layer_17.permute(0, 2, 1, 3)
        context_layer_17 = None
        context_layer_18 = permute_44.contiguous()
        permute_44 = None
        context_layer_19 = context_layer_18.view((1, 45, 32))
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
        add_24 = hidden_states_33 + hidden_states_31
        hidden_states_33 = hidden_states_31 = None
        hidden_states_34 = torch.nn.functional.layer_norm(
            add_24,
            (32,),
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_24 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
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
        add_25 = hidden_states_38 + hidden_states_34
        hidden_states_38 = hidden_states_34 = None
        hidden_states_39 = torch.nn.functional.layer_norm(
            add_25,
            (32,),
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_25 = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_bias_ = (None)
        first_token_tensor = hidden_states_39[(slice(None, None, None), 0)]
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
        return (hidden_states_39, pooled_output_1)
