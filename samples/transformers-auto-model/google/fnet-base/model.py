import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_token_type_ids_: torch.Tensor,
        L_self_modules_embeddings_buffers_position_ids_: torch.Tensor,
        L_self_modules_embeddings_modules_word_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_position_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_fourier_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_fourier_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_fourier_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_fourier_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_fourier_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_fourier_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_fourier_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_fourier_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_fourier_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_fourier_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_fourier_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_fourier_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_fourier_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_fourier_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_fourier_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_fourier_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_fourier_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_fourier_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_fourier_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_fourier_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_fourier_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_fourier_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_fourier_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_fourier_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_pooler_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_pooler_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_input_ids_ = L_input_ids_
        l_token_type_ids_ = L_token_type_ids_
        l_self_modules_embeddings_buffers_position_ids_ = (
            L_self_modules_embeddings_buffers_position_ids_
        )
        l_self_modules_embeddings_modules_word_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_word_embeddings_parameters_weight_
        )
        l_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_
        )
        l_self_modules_embeddings_modules_position_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_position_embeddings_parameters_weight_
        )
        l_self_modules_embeddings_modules_layer_norm_parameters_weight_ = (
            L_self_modules_embeddings_modules_LayerNorm_parameters_weight_
        )
        l_self_modules_embeddings_modules_layer_norm_parameters_bias_ = (
            L_self_modules_embeddings_modules_LayerNorm_parameters_bias_
        )
        l_self_modules_embeddings_modules_projection_parameters_weight_ = (
            L_self_modules_embeddings_modules_projection_parameters_weight_
        )
        l_self_modules_embeddings_modules_projection_parameters_bias_ = (
            L_self_modules_embeddings_modules_projection_parameters_bias_
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_fourier_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_fourier_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_fourier_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_fourier_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_fourier_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_fourier_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_fourier_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_fourier_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_fourier_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_fourier_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_fourier_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_fourier_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_fourier_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_fourier_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_fourier_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_fourier_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_fourier_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_fourier_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_fourier_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_fourier_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_fourier_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_fourier_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_fourier_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_fourier_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_fourier_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_fourier_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_fourier_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_fourier_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_fourier_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_fourier_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_fourier_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_fourier_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_fourier_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_fourier_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_fourier_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_fourier_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_fourier_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_fourier_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_fourier_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_fourier_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_fourier_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_fourier_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_fourier_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_fourier_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_fourier_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_fourier_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_fourier_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_fourier_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_pooler_modules_dense_parameters_weight_ = (
            L_self_modules_pooler_modules_dense_parameters_weight_
        )
        l_self_modules_pooler_modules_dense_parameters_bias_ = (
            L_self_modules_pooler_modules_dense_parameters_bias_
        )
        size_1 = l_input_ids_.size()
        getitem_3 = size_1[1]
        size_1 = None
        position_ids = l_self_modules_embeddings_buffers_position_ids_[
            (slice(None, None, None), slice(None, getitem_3, None))
        ]
        l_self_modules_embeddings_buffers_position_ids_ = getitem_3 = None
        inputs_embeds = torch.nn.functional.embedding(
            l_input_ids_,
            l_self_modules_embeddings_modules_word_embeddings_parameters_weight_,
            3,
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
        embeddings += position_embeddings
        embeddings_1 = embeddings
        embeddings = position_embeddings = None
        embeddings_2 = torch.nn.functional.layer_norm(
            embeddings_1,
            (768,),
            l_self_modules_embeddings_modules_layer_norm_parameters_weight_,
            l_self_modules_embeddings_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        embeddings_1 = (
            l_self_modules_embeddings_modules_layer_norm_parameters_weight_
        ) = l_self_modules_embeddings_modules_layer_norm_parameters_bias_ = None
        embeddings_3 = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_embeddings_modules_projection_parameters_weight_,
            l_self_modules_embeddings_modules_projection_parameters_bias_,
        )
        embeddings_2 = (
            l_self_modules_embeddings_modules_projection_parameters_weight_
        ) = l_self_modules_embeddings_modules_projection_parameters_bias_ = None
        embeddings_4 = torch.nn.functional.dropout(embeddings_3, 0.1, False, False)
        embeddings_3 = None
        fft_fftn = torch._C._fft.fft_fftn(embeddings_4, dim=(1, 2))
        outputs = fft_fftn.real
        fft_fftn = None
        add_1 = embeddings_4 + outputs
        embeddings_4 = outputs = None
        hidden_states = torch.nn.functional.layer_norm(
            add_1,
            (768,),
            l_self_modules_encoder_modules_layer_modules_0_modules_fourier_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_fourier_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_1 = l_self_modules_encoder_modules_layer_modules_0_modules_fourier_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_fourier_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_1 = torch._C._nn.linear(
            hidden_states,
            l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul = 0.5 * hidden_states_1
        pow_1 = torch.pow(hidden_states_1, 3.0)
        mul_1 = 0.044715 * pow_1
        pow_1 = None
        add_2 = hidden_states_1 + mul_1
        hidden_states_1 = mul_1 = None
        mul_2 = 0.7978845608028654 * add_2
        add_2 = None
        tanh = torch.tanh(mul_2)
        mul_2 = None
        add_3 = 1.0 + tanh
        tanh = None
        hidden_states_2 = mul * add_3
        mul = add_3 = None
        hidden_states_3 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_2 = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_4 = torch.nn.functional.dropout(
            hidden_states_3, 0.1, False, False
        )
        hidden_states_3 = None
        add_4 = hidden_states_4 + hidden_states
        hidden_states_4 = hidden_states = None
        hidden_states_5 = torch.nn.functional.layer_norm(
            add_4,
            (768,),
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_4 = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_bias_ = (None)
        fft_fftn_1 = torch._C._fft.fft_fftn(hidden_states_5, dim=(1, 2))
        outputs_1 = fft_fftn_1.real
        fft_fftn_1 = None
        add_5 = hidden_states_5 + outputs_1
        hidden_states_5 = outputs_1 = None
        hidden_states_6 = torch.nn.functional.layer_norm(
            add_5,
            (768,),
            l_self_modules_encoder_modules_layer_modules_1_modules_fourier_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_fourier_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_5 = l_self_modules_encoder_modules_layer_modules_1_modules_fourier_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_fourier_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_7 = torch._C._nn.linear(
            hidden_states_6,
            l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_4 = 0.5 * hidden_states_7
        pow_2 = torch.pow(hidden_states_7, 3.0)
        mul_5 = 0.044715 * pow_2
        pow_2 = None
        add_6 = hidden_states_7 + mul_5
        hidden_states_7 = mul_5 = None
        mul_6 = 0.7978845608028654 * add_6
        add_6 = None
        tanh_1 = torch.tanh(mul_6)
        mul_6 = None
        add_7 = 1.0 + tanh_1
        tanh_1 = None
        hidden_states_8 = mul_4 * add_7
        mul_4 = add_7 = None
        hidden_states_9 = torch._C._nn.linear(
            hidden_states_8,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_8 = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_10 = torch.nn.functional.dropout(
            hidden_states_9, 0.1, False, False
        )
        hidden_states_9 = None
        add_8 = hidden_states_10 + hidden_states_6
        hidden_states_10 = hidden_states_6 = None
        hidden_states_11 = torch.nn.functional.layer_norm(
            add_8,
            (768,),
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_8 = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_bias_ = (None)
        fft_fftn_2 = torch._C._fft.fft_fftn(hidden_states_11, dim=(1, 2))
        outputs_2 = fft_fftn_2.real
        fft_fftn_2 = None
        add_9 = hidden_states_11 + outputs_2
        hidden_states_11 = outputs_2 = None
        hidden_states_12 = torch.nn.functional.layer_norm(
            add_9,
            (768,),
            l_self_modules_encoder_modules_layer_modules_2_modules_fourier_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_fourier_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_9 = l_self_modules_encoder_modules_layer_modules_2_modules_fourier_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_fourier_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_13 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_8 = 0.5 * hidden_states_13
        pow_3 = torch.pow(hidden_states_13, 3.0)
        mul_9 = 0.044715 * pow_3
        pow_3 = None
        add_10 = hidden_states_13 + mul_9
        hidden_states_13 = mul_9 = None
        mul_10 = 0.7978845608028654 * add_10
        add_10 = None
        tanh_2 = torch.tanh(mul_10)
        mul_10 = None
        add_11 = 1.0 + tanh_2
        tanh_2 = None
        hidden_states_14 = mul_8 * add_11
        mul_8 = add_11 = None
        hidden_states_15 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_14 = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_16 = torch.nn.functional.dropout(
            hidden_states_15, 0.1, False, False
        )
        hidden_states_15 = None
        add_12 = hidden_states_16 + hidden_states_12
        hidden_states_16 = hidden_states_12 = None
        hidden_states_17 = torch.nn.functional.layer_norm(
            add_12,
            (768,),
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_12 = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_bias_ = (None)
        fft_fftn_3 = torch._C._fft.fft_fftn(hidden_states_17, dim=(1, 2))
        outputs_3 = fft_fftn_3.real
        fft_fftn_3 = None
        add_13 = hidden_states_17 + outputs_3
        hidden_states_17 = outputs_3 = None
        hidden_states_18 = torch.nn.functional.layer_norm(
            add_13,
            (768,),
            l_self_modules_encoder_modules_layer_modules_3_modules_fourier_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_fourier_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_13 = l_self_modules_encoder_modules_layer_modules_3_modules_fourier_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_fourier_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_19 = torch._C._nn.linear(
            hidden_states_18,
            l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_12 = 0.5 * hidden_states_19
        pow_4 = torch.pow(hidden_states_19, 3.0)
        mul_13 = 0.044715 * pow_4
        pow_4 = None
        add_14 = hidden_states_19 + mul_13
        hidden_states_19 = mul_13 = None
        mul_14 = 0.7978845608028654 * add_14
        add_14 = None
        tanh_3 = torch.tanh(mul_14)
        mul_14 = None
        add_15 = 1.0 + tanh_3
        tanh_3 = None
        hidden_states_20 = mul_12 * add_15
        mul_12 = add_15 = None
        hidden_states_21 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_20 = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_22 = torch.nn.functional.dropout(
            hidden_states_21, 0.1, False, False
        )
        hidden_states_21 = None
        add_16 = hidden_states_22 + hidden_states_18
        hidden_states_22 = hidden_states_18 = None
        hidden_states_23 = torch.nn.functional.layer_norm(
            add_16,
            (768,),
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_16 = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_bias_ = (None)
        fft_fftn_4 = torch._C._fft.fft_fftn(hidden_states_23, dim=(1, 2))
        outputs_4 = fft_fftn_4.real
        fft_fftn_4 = None
        add_17 = hidden_states_23 + outputs_4
        hidden_states_23 = outputs_4 = None
        hidden_states_24 = torch.nn.functional.layer_norm(
            add_17,
            (768,),
            l_self_modules_encoder_modules_layer_modules_4_modules_fourier_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_fourier_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_17 = l_self_modules_encoder_modules_layer_modules_4_modules_fourier_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_fourier_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_25 = torch._C._nn.linear(
            hidden_states_24,
            l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_16 = 0.5 * hidden_states_25
        pow_5 = torch.pow(hidden_states_25, 3.0)
        mul_17 = 0.044715 * pow_5
        pow_5 = None
        add_18 = hidden_states_25 + mul_17
        hidden_states_25 = mul_17 = None
        mul_18 = 0.7978845608028654 * add_18
        add_18 = None
        tanh_4 = torch.tanh(mul_18)
        mul_18 = None
        add_19 = 1.0 + tanh_4
        tanh_4 = None
        hidden_states_26 = mul_16 * add_19
        mul_16 = add_19 = None
        hidden_states_27 = torch._C._nn.linear(
            hidden_states_26,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_26 = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_28 = torch.nn.functional.dropout(
            hidden_states_27, 0.1, False, False
        )
        hidden_states_27 = None
        add_20 = hidden_states_28 + hidden_states_24
        hidden_states_28 = hidden_states_24 = None
        hidden_states_29 = torch.nn.functional.layer_norm(
            add_20,
            (768,),
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_20 = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_bias_ = (None)
        fft_fftn_5 = torch._C._fft.fft_fftn(hidden_states_29, dim=(1, 2))
        outputs_5 = fft_fftn_5.real
        fft_fftn_5 = None
        add_21 = hidden_states_29 + outputs_5
        hidden_states_29 = outputs_5 = None
        hidden_states_30 = torch.nn.functional.layer_norm(
            add_21,
            (768,),
            l_self_modules_encoder_modules_layer_modules_5_modules_fourier_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_fourier_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_21 = l_self_modules_encoder_modules_layer_modules_5_modules_fourier_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_fourier_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_31 = torch._C._nn.linear(
            hidden_states_30,
            l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_20 = 0.5 * hidden_states_31
        pow_6 = torch.pow(hidden_states_31, 3.0)
        mul_21 = 0.044715 * pow_6
        pow_6 = None
        add_22 = hidden_states_31 + mul_21
        hidden_states_31 = mul_21 = None
        mul_22 = 0.7978845608028654 * add_22
        add_22 = None
        tanh_5 = torch.tanh(mul_22)
        mul_22 = None
        add_23 = 1.0 + tanh_5
        tanh_5 = None
        hidden_states_32 = mul_20 * add_23
        mul_20 = add_23 = None
        hidden_states_33 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_32 = l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_34 = torch.nn.functional.dropout(
            hidden_states_33, 0.1, False, False
        )
        hidden_states_33 = None
        add_24 = hidden_states_34 + hidden_states_30
        hidden_states_34 = hidden_states_30 = None
        hidden_states_35 = torch.nn.functional.layer_norm(
            add_24,
            (768,),
            l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_24 = l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_parameters_bias_ = (None)
        fft_fftn_6 = torch._C._fft.fft_fftn(hidden_states_35, dim=(1, 2))
        outputs_6 = fft_fftn_6.real
        fft_fftn_6 = None
        add_25 = hidden_states_35 + outputs_6
        hidden_states_35 = outputs_6 = None
        hidden_states_36 = torch.nn.functional.layer_norm(
            add_25,
            (768,),
            l_self_modules_encoder_modules_layer_modules_6_modules_fourier_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_fourier_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_25 = l_self_modules_encoder_modules_layer_modules_6_modules_fourier_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_fourier_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_37 = torch._C._nn.linear(
            hidden_states_36,
            l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_24 = 0.5 * hidden_states_37
        pow_7 = torch.pow(hidden_states_37, 3.0)
        mul_25 = 0.044715 * pow_7
        pow_7 = None
        add_26 = hidden_states_37 + mul_25
        hidden_states_37 = mul_25 = None
        mul_26 = 0.7978845608028654 * add_26
        add_26 = None
        tanh_6 = torch.tanh(mul_26)
        mul_26 = None
        add_27 = 1.0 + tanh_6
        tanh_6 = None
        hidden_states_38 = mul_24 * add_27
        mul_24 = add_27 = None
        hidden_states_39 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_38 = l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_40 = torch.nn.functional.dropout(
            hidden_states_39, 0.1, False, False
        )
        hidden_states_39 = None
        add_28 = hidden_states_40 + hidden_states_36
        hidden_states_40 = hidden_states_36 = None
        hidden_states_41 = torch.nn.functional.layer_norm(
            add_28,
            (768,),
            l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_28 = l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_parameters_bias_ = (None)
        fft_fftn_7 = torch._C._fft.fft_fftn(hidden_states_41, dim=(1, 2))
        outputs_7 = fft_fftn_7.real
        fft_fftn_7 = None
        add_29 = hidden_states_41 + outputs_7
        hidden_states_41 = outputs_7 = None
        hidden_states_42 = torch.nn.functional.layer_norm(
            add_29,
            (768,),
            l_self_modules_encoder_modules_layer_modules_7_modules_fourier_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_fourier_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_29 = l_self_modules_encoder_modules_layer_modules_7_modules_fourier_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_fourier_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_43 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_28 = 0.5 * hidden_states_43
        pow_8 = torch.pow(hidden_states_43, 3.0)
        mul_29 = 0.044715 * pow_8
        pow_8 = None
        add_30 = hidden_states_43 + mul_29
        hidden_states_43 = mul_29 = None
        mul_30 = 0.7978845608028654 * add_30
        add_30 = None
        tanh_7 = torch.tanh(mul_30)
        mul_30 = None
        add_31 = 1.0 + tanh_7
        tanh_7 = None
        hidden_states_44 = mul_28 * add_31
        mul_28 = add_31 = None
        hidden_states_45 = torch._C._nn.linear(
            hidden_states_44,
            l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_44 = l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_46 = torch.nn.functional.dropout(
            hidden_states_45, 0.1, False, False
        )
        hidden_states_45 = None
        add_32 = hidden_states_46 + hidden_states_42
        hidden_states_46 = hidden_states_42 = None
        hidden_states_47 = torch.nn.functional.layer_norm(
            add_32,
            (768,),
            l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_32 = l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_parameters_bias_ = (None)
        fft_fftn_8 = torch._C._fft.fft_fftn(hidden_states_47, dim=(1, 2))
        outputs_8 = fft_fftn_8.real
        fft_fftn_8 = None
        add_33 = hidden_states_47 + outputs_8
        hidden_states_47 = outputs_8 = None
        hidden_states_48 = torch.nn.functional.layer_norm(
            add_33,
            (768,),
            l_self_modules_encoder_modules_layer_modules_8_modules_fourier_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_fourier_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_33 = l_self_modules_encoder_modules_layer_modules_8_modules_fourier_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_fourier_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_49 = torch._C._nn.linear(
            hidden_states_48,
            l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_32 = 0.5 * hidden_states_49
        pow_9 = torch.pow(hidden_states_49, 3.0)
        mul_33 = 0.044715 * pow_9
        pow_9 = None
        add_34 = hidden_states_49 + mul_33
        hidden_states_49 = mul_33 = None
        mul_34 = 0.7978845608028654 * add_34
        add_34 = None
        tanh_8 = torch.tanh(mul_34)
        mul_34 = None
        add_35 = 1.0 + tanh_8
        tanh_8 = None
        hidden_states_50 = mul_32 * add_35
        mul_32 = add_35 = None
        hidden_states_51 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_50 = l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_52 = torch.nn.functional.dropout(
            hidden_states_51, 0.1, False, False
        )
        hidden_states_51 = None
        add_36 = hidden_states_52 + hidden_states_48
        hidden_states_52 = hidden_states_48 = None
        hidden_states_53 = torch.nn.functional.layer_norm(
            add_36,
            (768,),
            l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_36 = l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_parameters_bias_ = (None)
        fft_fftn_9 = torch._C._fft.fft_fftn(hidden_states_53, dim=(1, 2))
        outputs_9 = fft_fftn_9.real
        fft_fftn_9 = None
        add_37 = hidden_states_53 + outputs_9
        hidden_states_53 = outputs_9 = None
        hidden_states_54 = torch.nn.functional.layer_norm(
            add_37,
            (768,),
            l_self_modules_encoder_modules_layer_modules_9_modules_fourier_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_fourier_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_37 = l_self_modules_encoder_modules_layer_modules_9_modules_fourier_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_fourier_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_55 = torch._C._nn.linear(
            hidden_states_54,
            l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_36 = 0.5 * hidden_states_55
        pow_10 = torch.pow(hidden_states_55, 3.0)
        mul_37 = 0.044715 * pow_10
        pow_10 = None
        add_38 = hidden_states_55 + mul_37
        hidden_states_55 = mul_37 = None
        mul_38 = 0.7978845608028654 * add_38
        add_38 = None
        tanh_9 = torch.tanh(mul_38)
        mul_38 = None
        add_39 = 1.0 + tanh_9
        tanh_9 = None
        hidden_states_56 = mul_36 * add_39
        mul_36 = add_39 = None
        hidden_states_57 = torch._C._nn.linear(
            hidden_states_56,
            l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_56 = l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_58 = torch.nn.functional.dropout(
            hidden_states_57, 0.1, False, False
        )
        hidden_states_57 = None
        add_40 = hidden_states_58 + hidden_states_54
        hidden_states_58 = hidden_states_54 = None
        hidden_states_59 = torch.nn.functional.layer_norm(
            add_40,
            (768,),
            l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_40 = l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_parameters_bias_ = (None)
        fft_fftn_10 = torch._C._fft.fft_fftn(hidden_states_59, dim=(1, 2))
        outputs_10 = fft_fftn_10.real
        fft_fftn_10 = None
        add_41 = hidden_states_59 + outputs_10
        hidden_states_59 = outputs_10 = None
        hidden_states_60 = torch.nn.functional.layer_norm(
            add_41,
            (768,),
            l_self_modules_encoder_modules_layer_modules_10_modules_fourier_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_fourier_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_41 = l_self_modules_encoder_modules_layer_modules_10_modules_fourier_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_fourier_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_61 = torch._C._nn.linear(
            hidden_states_60,
            l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_40 = 0.5 * hidden_states_61
        pow_11 = torch.pow(hidden_states_61, 3.0)
        mul_41 = 0.044715 * pow_11
        pow_11 = None
        add_42 = hidden_states_61 + mul_41
        hidden_states_61 = mul_41 = None
        mul_42 = 0.7978845608028654 * add_42
        add_42 = None
        tanh_10 = torch.tanh(mul_42)
        mul_42 = None
        add_43 = 1.0 + tanh_10
        tanh_10 = None
        hidden_states_62 = mul_40 * add_43
        mul_40 = add_43 = None
        hidden_states_63 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_62 = l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_64 = torch.nn.functional.dropout(
            hidden_states_63, 0.1, False, False
        )
        hidden_states_63 = None
        add_44 = hidden_states_64 + hidden_states_60
        hidden_states_64 = hidden_states_60 = None
        hidden_states_65 = torch.nn.functional.layer_norm(
            add_44,
            (768,),
            l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_44 = l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_parameters_bias_ = (None)
        fft_fftn_11 = torch._C._fft.fft_fftn(hidden_states_65, dim=(1, 2))
        outputs_11 = fft_fftn_11.real
        fft_fftn_11 = None
        add_45 = hidden_states_65 + outputs_11
        hidden_states_65 = outputs_11 = None
        hidden_states_66 = torch.nn.functional.layer_norm(
            add_45,
            (768,),
            l_self_modules_encoder_modules_layer_modules_11_modules_fourier_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_fourier_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_45 = l_self_modules_encoder_modules_layer_modules_11_modules_fourier_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_fourier_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_67 = torch._C._nn.linear(
            hidden_states_66,
            l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_44 = 0.5 * hidden_states_67
        pow_12 = torch.pow(hidden_states_67, 3.0)
        mul_45 = 0.044715 * pow_12
        pow_12 = None
        add_46 = hidden_states_67 + mul_45
        hidden_states_67 = mul_45 = None
        mul_46 = 0.7978845608028654 * add_46
        add_46 = None
        tanh_11 = torch.tanh(mul_46)
        mul_46 = None
        add_47 = 1.0 + tanh_11
        tanh_11 = None
        hidden_states_68 = mul_44 * add_47
        mul_44 = add_47 = None
        hidden_states_69 = torch._C._nn.linear(
            hidden_states_68,
            l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_68 = l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_70 = torch.nn.functional.dropout(
            hidden_states_69, 0.1, False, False
        )
        hidden_states_69 = None
        add_48 = hidden_states_70 + hidden_states_66
        hidden_states_70 = hidden_states_66 = None
        hidden_states_71 = torch.nn.functional.layer_norm(
            add_48,
            (768,),
            l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_48 = l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_parameters_bias_ = (None)
        first_token_tensor = hidden_states_71[(slice(None, None, None), 0)]
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
        return (hidden_states_71, pooled_output_1)
