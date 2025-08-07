import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_self_modules_embeddings_modules_word_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_buffers_position_ids_: torch.Tensor,
        L_self_modules_embeddings_modules_position_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_attention_mask_: torch.Tensor,
        L_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_q_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_q_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_k_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_k_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_v_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_v_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_out_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_out_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_0_modules_sa_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_0_modules_sa_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_0_modules_ffn_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_0_modules_ffn_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_0_modules_ffn_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_0_modules_ffn_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_0_modules_output_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_0_modules_output_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_q_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_q_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_k_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_k_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_v_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_v_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_out_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_out_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_1_modules_sa_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_1_modules_sa_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_1_modules_ffn_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_1_modules_ffn_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_1_modules_ffn_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_1_modules_ffn_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_1_modules_output_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_1_modules_output_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_q_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_q_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_k_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_k_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_v_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_v_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_out_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_out_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_2_modules_sa_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_2_modules_sa_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_2_modules_ffn_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_2_modules_ffn_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_2_modules_ffn_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_2_modules_ffn_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_2_modules_output_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_2_modules_output_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_q_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_q_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_k_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_k_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_v_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_v_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_out_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_out_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_3_modules_sa_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_3_modules_sa_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_3_modules_ffn_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_3_modules_ffn_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_3_modules_ffn_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_3_modules_ffn_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_3_modules_output_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_3_modules_output_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_q_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_q_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_k_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_k_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_v_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_v_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_out_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_out_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_4_modules_sa_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_4_modules_sa_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_4_modules_ffn_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_4_modules_ffn_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_4_modules_ffn_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_4_modules_ffn_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_4_modules_output_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_4_modules_output_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_q_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_q_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_k_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_k_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_v_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_v_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_out_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_out_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_5_modules_sa_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_5_modules_sa_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_5_modules_ffn_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_5_modules_ffn_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_5_modules_ffn_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_5_modules_ffn_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_5_modules_output_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_layer_modules_5_modules_output_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_input_ids_ = L_input_ids_
        l_self_modules_embeddings_modules_word_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_word_embeddings_parameters_weight_
        )
        l_self_modules_embeddings_buffers_position_ids_ = (
            L_self_modules_embeddings_buffers_position_ids_
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
        l_attention_mask_ = L_attention_mask_
        l_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_q_lin_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_q_lin_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_q_lin_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_q_lin_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_k_lin_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_k_lin_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_k_lin_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_k_lin_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_v_lin_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_v_lin_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_v_lin_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_v_lin_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_out_lin_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_out_lin_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_out_lin_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_out_lin_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_0_modules_sa_layer_norm_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_0_modules_sa_layer_norm_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_0_modules_sa_layer_norm_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_0_modules_sa_layer_norm_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_0_modules_ffn_modules_lin1_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_0_modules_ffn_modules_lin1_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_0_modules_ffn_modules_lin1_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_0_modules_ffn_modules_lin1_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_0_modules_ffn_modules_lin2_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_0_modules_ffn_modules_lin2_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_0_modules_ffn_modules_lin2_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_0_modules_ffn_modules_lin2_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_0_modules_output_layer_norm_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_0_modules_output_layer_norm_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_0_modules_output_layer_norm_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_0_modules_output_layer_norm_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_q_lin_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_q_lin_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_q_lin_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_q_lin_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_k_lin_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_k_lin_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_k_lin_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_k_lin_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_v_lin_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_v_lin_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_v_lin_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_v_lin_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_out_lin_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_out_lin_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_out_lin_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_out_lin_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_1_modules_sa_layer_norm_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_1_modules_sa_layer_norm_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_1_modules_sa_layer_norm_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_1_modules_sa_layer_norm_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_1_modules_ffn_modules_lin1_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_1_modules_ffn_modules_lin1_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_1_modules_ffn_modules_lin1_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_1_modules_ffn_modules_lin1_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_1_modules_ffn_modules_lin2_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_1_modules_ffn_modules_lin2_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_1_modules_ffn_modules_lin2_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_1_modules_ffn_modules_lin2_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_1_modules_output_layer_norm_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_1_modules_output_layer_norm_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_1_modules_output_layer_norm_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_1_modules_output_layer_norm_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_q_lin_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_q_lin_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_q_lin_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_q_lin_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_k_lin_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_k_lin_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_k_lin_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_k_lin_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_v_lin_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_v_lin_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_v_lin_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_v_lin_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_out_lin_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_out_lin_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_out_lin_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_out_lin_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_2_modules_sa_layer_norm_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_2_modules_sa_layer_norm_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_2_modules_sa_layer_norm_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_2_modules_sa_layer_norm_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_2_modules_ffn_modules_lin1_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_2_modules_ffn_modules_lin1_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_2_modules_ffn_modules_lin1_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_2_modules_ffn_modules_lin1_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_2_modules_ffn_modules_lin2_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_2_modules_ffn_modules_lin2_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_2_modules_ffn_modules_lin2_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_2_modules_ffn_modules_lin2_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_2_modules_output_layer_norm_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_2_modules_output_layer_norm_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_2_modules_output_layer_norm_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_2_modules_output_layer_norm_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_q_lin_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_q_lin_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_q_lin_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_q_lin_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_k_lin_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_k_lin_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_k_lin_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_k_lin_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_v_lin_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_v_lin_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_v_lin_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_v_lin_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_out_lin_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_out_lin_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_out_lin_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_out_lin_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_3_modules_sa_layer_norm_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_3_modules_sa_layer_norm_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_3_modules_sa_layer_norm_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_3_modules_sa_layer_norm_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_3_modules_ffn_modules_lin1_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_3_modules_ffn_modules_lin1_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_3_modules_ffn_modules_lin1_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_3_modules_ffn_modules_lin1_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_3_modules_ffn_modules_lin2_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_3_modules_ffn_modules_lin2_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_3_modules_ffn_modules_lin2_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_3_modules_ffn_modules_lin2_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_3_modules_output_layer_norm_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_3_modules_output_layer_norm_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_3_modules_output_layer_norm_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_3_modules_output_layer_norm_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_q_lin_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_q_lin_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_q_lin_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_q_lin_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_k_lin_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_k_lin_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_k_lin_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_k_lin_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_v_lin_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_v_lin_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_v_lin_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_v_lin_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_out_lin_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_out_lin_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_out_lin_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_out_lin_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_4_modules_sa_layer_norm_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_4_modules_sa_layer_norm_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_4_modules_sa_layer_norm_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_4_modules_sa_layer_norm_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_4_modules_ffn_modules_lin1_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_4_modules_ffn_modules_lin1_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_4_modules_ffn_modules_lin1_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_4_modules_ffn_modules_lin1_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_4_modules_ffn_modules_lin2_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_4_modules_ffn_modules_lin2_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_4_modules_ffn_modules_lin2_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_4_modules_ffn_modules_lin2_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_4_modules_output_layer_norm_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_4_modules_output_layer_norm_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_4_modules_output_layer_norm_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_4_modules_output_layer_norm_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_q_lin_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_q_lin_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_q_lin_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_q_lin_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_k_lin_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_k_lin_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_k_lin_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_k_lin_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_v_lin_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_v_lin_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_v_lin_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_v_lin_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_out_lin_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_out_lin_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_out_lin_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_out_lin_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_5_modules_sa_layer_norm_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_5_modules_sa_layer_norm_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_5_modules_sa_layer_norm_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_5_modules_sa_layer_norm_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_5_modules_ffn_modules_lin1_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_5_modules_ffn_modules_lin1_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_5_modules_ffn_modules_lin1_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_5_modules_ffn_modules_lin1_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_5_modules_ffn_modules_lin2_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_5_modules_ffn_modules_lin2_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_5_modules_ffn_modules_lin2_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_5_modules_ffn_modules_lin2_parameters_bias_
        l_self_modules_transformer_modules_layer_modules_5_modules_output_layer_norm_parameters_weight_ = L_self_modules_transformer_modules_layer_modules_5_modules_output_layer_norm_parameters_weight_
        l_self_modules_transformer_modules_layer_modules_5_modules_output_layer_norm_parameters_bias_ = L_self_modules_transformer_modules_layer_modules_5_modules_output_layer_norm_parameters_bias_
        input_embeds = torch.nn.functional.embedding(
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
        position_ids = l_self_modules_embeddings_buffers_position_ids_[
            (slice(None, None, None), slice(None, 22, None))
        ]
        l_self_modules_embeddings_buffers_position_ids_ = None
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
        embeddings = input_embeds + position_embeddings
        input_embeds = position_embeddings = None
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
        getitem_1 = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        l_attention_mask_ = None
        expand = getitem_1.expand(1, 1, 22, 22)
        getitem_1 = None
        expanded_mask = expand.to(torch.float32)
        expand = None
        tensor = torch.tensor(1.0, dtype=torch.float32)
        inverted_mask = tensor - expanded_mask
        tensor = expanded_mask = None
        to_1 = inverted_mask.to(torch.bool)
        attention_mask = inverted_mask.masked_fill(to_1, -3.4028234663852886e38)
        inverted_mask = to_1 = None
        linear = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_q_lin_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_q_lin_parameters_bias_,
        )
        l_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_q_lin_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_q_lin_parameters_bias_ = (None)
        view = linear.view(1, -1, 12, 64)
        linear = None
        q = view.transpose(1, 2)
        view = None
        linear_1 = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_k_lin_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_k_lin_parameters_bias_,
        )
        l_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_k_lin_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_k_lin_parameters_bias_ = (None)
        view_1 = linear_1.view(1, -1, 12, 64)
        linear_1 = None
        k = view_1.transpose(1, 2)
        view_1 = None
        linear_2 = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_v_lin_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_v_lin_parameters_bias_,
        )
        l_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_v_lin_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_v_lin_parameters_bias_ = (None)
        view_2 = linear_2.view(1, -1, 12, 64)
        linear_2 = None
        v = view_2.transpose(1, 2)
        view_2 = None
        attn_output = torch._C._nn.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        q = k = v = None
        transpose_3 = attn_output.transpose(1, 2)
        attn_output = None
        contiguous = transpose_3.contiguous()
        transpose_3 = None
        attn_output_1 = contiguous.view(1, -1, 768)
        contiguous = None
        attn_output_2 = torch._C._nn.linear(
            attn_output_1,
            l_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_out_lin_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_out_lin_parameters_bias_,
        )
        attn_output_1 = l_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_out_lin_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_out_lin_parameters_bias_ = (None)
        add_1 = attn_output_2 + embeddings_2
        attn_output_2 = embeddings_2 = None
        sa_output = torch.nn.functional.layer_norm(
            add_1,
            (768,),
            l_self_modules_transformer_modules_layer_modules_0_modules_sa_layer_norm_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_0_modules_sa_layer_norm_parameters_bias_,
            1e-12,
        )
        add_1 = l_self_modules_transformer_modules_layer_modules_0_modules_sa_layer_norm_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_0_modules_sa_layer_norm_parameters_bias_ = (None)
        x = torch._C._nn.linear(
            sa_output,
            l_self_modules_transformer_modules_layer_modules_0_modules_ffn_modules_lin1_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_0_modules_ffn_modules_lin1_parameters_bias_,
        )
        l_self_modules_transformer_modules_layer_modules_0_modules_ffn_modules_lin1_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_0_modules_ffn_modules_lin1_parameters_bias_ = (None)
        x_1 = torch._C._nn.gelu(x)
        x = None
        x_2 = torch._C._nn.linear(
            x_1,
            l_self_modules_transformer_modules_layer_modules_0_modules_ffn_modules_lin2_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_0_modules_ffn_modules_lin2_parameters_bias_,
        )
        x_1 = l_self_modules_transformer_modules_layer_modules_0_modules_ffn_modules_lin2_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_0_modules_ffn_modules_lin2_parameters_bias_ = (None)
        x_3 = torch.nn.functional.dropout(x_2, 0.1, False, False)
        x_2 = None
        add_2 = x_3 + sa_output
        x_3 = sa_output = None
        ffn_output = torch.nn.functional.layer_norm(
            add_2,
            (768,),
            l_self_modules_transformer_modules_layer_modules_0_modules_output_layer_norm_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_0_modules_output_layer_norm_parameters_bias_,
            1e-12,
        )
        add_2 = l_self_modules_transformer_modules_layer_modules_0_modules_output_layer_norm_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_0_modules_output_layer_norm_parameters_bias_ = (None)
        linear_6 = torch._C._nn.linear(
            ffn_output,
            l_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_q_lin_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_q_lin_parameters_bias_,
        )
        l_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_q_lin_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_q_lin_parameters_bias_ = (None)
        view_4 = linear_6.view(1, -1, 12, 64)
        linear_6 = None
        q_1 = view_4.transpose(1, 2)
        view_4 = None
        linear_7 = torch._C._nn.linear(
            ffn_output,
            l_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_k_lin_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_k_lin_parameters_bias_,
        )
        l_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_k_lin_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_k_lin_parameters_bias_ = (None)
        view_5 = linear_7.view(1, -1, 12, 64)
        linear_7 = None
        k_1 = view_5.transpose(1, 2)
        view_5 = None
        linear_8 = torch._C._nn.linear(
            ffn_output,
            l_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_v_lin_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_v_lin_parameters_bias_,
        )
        l_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_v_lin_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_v_lin_parameters_bias_ = (None)
        view_6 = linear_8.view(1, -1, 12, 64)
        linear_8 = None
        v_1 = view_6.transpose(1, 2)
        view_6 = None
        attn_output_3 = torch._C._nn.scaled_dot_product_attention(
            q_1, k_1, v_1, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        q_1 = k_1 = v_1 = None
        transpose_7 = attn_output_3.transpose(1, 2)
        attn_output_3 = None
        contiguous_1 = transpose_7.contiguous()
        transpose_7 = None
        attn_output_4 = contiguous_1.view(1, -1, 768)
        contiguous_1 = None
        attn_output_5 = torch._C._nn.linear(
            attn_output_4,
            l_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_out_lin_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_out_lin_parameters_bias_,
        )
        attn_output_4 = l_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_out_lin_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_out_lin_parameters_bias_ = (None)
        add_3 = attn_output_5 + ffn_output
        attn_output_5 = ffn_output = None
        sa_output_1 = torch.nn.functional.layer_norm(
            add_3,
            (768,),
            l_self_modules_transformer_modules_layer_modules_1_modules_sa_layer_norm_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_1_modules_sa_layer_norm_parameters_bias_,
            1e-12,
        )
        add_3 = l_self_modules_transformer_modules_layer_modules_1_modules_sa_layer_norm_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_1_modules_sa_layer_norm_parameters_bias_ = (None)
        x_4 = torch._C._nn.linear(
            sa_output_1,
            l_self_modules_transformer_modules_layer_modules_1_modules_ffn_modules_lin1_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_1_modules_ffn_modules_lin1_parameters_bias_,
        )
        l_self_modules_transformer_modules_layer_modules_1_modules_ffn_modules_lin1_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_1_modules_ffn_modules_lin1_parameters_bias_ = (None)
        x_5 = torch._C._nn.gelu(x_4)
        x_4 = None
        x_6 = torch._C._nn.linear(
            x_5,
            l_self_modules_transformer_modules_layer_modules_1_modules_ffn_modules_lin2_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_1_modules_ffn_modules_lin2_parameters_bias_,
        )
        x_5 = l_self_modules_transformer_modules_layer_modules_1_modules_ffn_modules_lin2_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_1_modules_ffn_modules_lin2_parameters_bias_ = (None)
        x_7 = torch.nn.functional.dropout(x_6, 0.1, False, False)
        x_6 = None
        add_4 = x_7 + sa_output_1
        x_7 = sa_output_1 = None
        ffn_output_1 = torch.nn.functional.layer_norm(
            add_4,
            (768,),
            l_self_modules_transformer_modules_layer_modules_1_modules_output_layer_norm_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_1_modules_output_layer_norm_parameters_bias_,
            1e-12,
        )
        add_4 = l_self_modules_transformer_modules_layer_modules_1_modules_output_layer_norm_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_1_modules_output_layer_norm_parameters_bias_ = (None)
        linear_12 = torch._C._nn.linear(
            ffn_output_1,
            l_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_q_lin_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_q_lin_parameters_bias_,
        )
        l_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_q_lin_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_q_lin_parameters_bias_ = (None)
        view_8 = linear_12.view(1, -1, 12, 64)
        linear_12 = None
        q_2 = view_8.transpose(1, 2)
        view_8 = None
        linear_13 = torch._C._nn.linear(
            ffn_output_1,
            l_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_k_lin_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_k_lin_parameters_bias_,
        )
        l_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_k_lin_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_k_lin_parameters_bias_ = (None)
        view_9 = linear_13.view(1, -1, 12, 64)
        linear_13 = None
        k_2 = view_9.transpose(1, 2)
        view_9 = None
        linear_14 = torch._C._nn.linear(
            ffn_output_1,
            l_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_v_lin_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_v_lin_parameters_bias_,
        )
        l_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_v_lin_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_v_lin_parameters_bias_ = (None)
        view_10 = linear_14.view(1, -1, 12, 64)
        linear_14 = None
        v_2 = view_10.transpose(1, 2)
        view_10 = None
        attn_output_6 = torch._C._nn.scaled_dot_product_attention(
            q_2, k_2, v_2, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        q_2 = k_2 = v_2 = None
        transpose_11 = attn_output_6.transpose(1, 2)
        attn_output_6 = None
        contiguous_2 = transpose_11.contiguous()
        transpose_11 = None
        attn_output_7 = contiguous_2.view(1, -1, 768)
        contiguous_2 = None
        attn_output_8 = torch._C._nn.linear(
            attn_output_7,
            l_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_out_lin_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_out_lin_parameters_bias_,
        )
        attn_output_7 = l_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_out_lin_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_out_lin_parameters_bias_ = (None)
        add_5 = attn_output_8 + ffn_output_1
        attn_output_8 = ffn_output_1 = None
        sa_output_2 = torch.nn.functional.layer_norm(
            add_5,
            (768,),
            l_self_modules_transformer_modules_layer_modules_2_modules_sa_layer_norm_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_2_modules_sa_layer_norm_parameters_bias_,
            1e-12,
        )
        add_5 = l_self_modules_transformer_modules_layer_modules_2_modules_sa_layer_norm_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_2_modules_sa_layer_norm_parameters_bias_ = (None)
        x_8 = torch._C._nn.linear(
            sa_output_2,
            l_self_modules_transformer_modules_layer_modules_2_modules_ffn_modules_lin1_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_2_modules_ffn_modules_lin1_parameters_bias_,
        )
        l_self_modules_transformer_modules_layer_modules_2_modules_ffn_modules_lin1_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_2_modules_ffn_modules_lin1_parameters_bias_ = (None)
        x_9 = torch._C._nn.gelu(x_8)
        x_8 = None
        x_10 = torch._C._nn.linear(
            x_9,
            l_self_modules_transformer_modules_layer_modules_2_modules_ffn_modules_lin2_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_2_modules_ffn_modules_lin2_parameters_bias_,
        )
        x_9 = l_self_modules_transformer_modules_layer_modules_2_modules_ffn_modules_lin2_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_2_modules_ffn_modules_lin2_parameters_bias_ = (None)
        x_11 = torch.nn.functional.dropout(x_10, 0.1, False, False)
        x_10 = None
        add_6 = x_11 + sa_output_2
        x_11 = sa_output_2 = None
        ffn_output_2 = torch.nn.functional.layer_norm(
            add_6,
            (768,),
            l_self_modules_transformer_modules_layer_modules_2_modules_output_layer_norm_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_2_modules_output_layer_norm_parameters_bias_,
            1e-12,
        )
        add_6 = l_self_modules_transformer_modules_layer_modules_2_modules_output_layer_norm_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_2_modules_output_layer_norm_parameters_bias_ = (None)
        linear_18 = torch._C._nn.linear(
            ffn_output_2,
            l_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_q_lin_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_q_lin_parameters_bias_,
        )
        l_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_q_lin_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_q_lin_parameters_bias_ = (None)
        view_12 = linear_18.view(1, -1, 12, 64)
        linear_18 = None
        q_3 = view_12.transpose(1, 2)
        view_12 = None
        linear_19 = torch._C._nn.linear(
            ffn_output_2,
            l_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_k_lin_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_k_lin_parameters_bias_,
        )
        l_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_k_lin_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_k_lin_parameters_bias_ = (None)
        view_13 = linear_19.view(1, -1, 12, 64)
        linear_19 = None
        k_3 = view_13.transpose(1, 2)
        view_13 = None
        linear_20 = torch._C._nn.linear(
            ffn_output_2,
            l_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_v_lin_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_v_lin_parameters_bias_,
        )
        l_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_v_lin_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_v_lin_parameters_bias_ = (None)
        view_14 = linear_20.view(1, -1, 12, 64)
        linear_20 = None
        v_3 = view_14.transpose(1, 2)
        view_14 = None
        attn_output_9 = torch._C._nn.scaled_dot_product_attention(
            q_3, k_3, v_3, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        q_3 = k_3 = v_3 = None
        transpose_15 = attn_output_9.transpose(1, 2)
        attn_output_9 = None
        contiguous_3 = transpose_15.contiguous()
        transpose_15 = None
        attn_output_10 = contiguous_3.view(1, -1, 768)
        contiguous_3 = None
        attn_output_11 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_out_lin_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_out_lin_parameters_bias_,
        )
        attn_output_10 = l_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_out_lin_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_out_lin_parameters_bias_ = (None)
        add_7 = attn_output_11 + ffn_output_2
        attn_output_11 = ffn_output_2 = None
        sa_output_3 = torch.nn.functional.layer_norm(
            add_7,
            (768,),
            l_self_modules_transformer_modules_layer_modules_3_modules_sa_layer_norm_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_3_modules_sa_layer_norm_parameters_bias_,
            1e-12,
        )
        add_7 = l_self_modules_transformer_modules_layer_modules_3_modules_sa_layer_norm_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_3_modules_sa_layer_norm_parameters_bias_ = (None)
        x_12 = torch._C._nn.linear(
            sa_output_3,
            l_self_modules_transformer_modules_layer_modules_3_modules_ffn_modules_lin1_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_3_modules_ffn_modules_lin1_parameters_bias_,
        )
        l_self_modules_transformer_modules_layer_modules_3_modules_ffn_modules_lin1_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_3_modules_ffn_modules_lin1_parameters_bias_ = (None)
        x_13 = torch._C._nn.gelu(x_12)
        x_12 = None
        x_14 = torch._C._nn.linear(
            x_13,
            l_self_modules_transformer_modules_layer_modules_3_modules_ffn_modules_lin2_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_3_modules_ffn_modules_lin2_parameters_bias_,
        )
        x_13 = l_self_modules_transformer_modules_layer_modules_3_modules_ffn_modules_lin2_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_3_modules_ffn_modules_lin2_parameters_bias_ = (None)
        x_15 = torch.nn.functional.dropout(x_14, 0.1, False, False)
        x_14 = None
        add_8 = x_15 + sa_output_3
        x_15 = sa_output_3 = None
        ffn_output_3 = torch.nn.functional.layer_norm(
            add_8,
            (768,),
            l_self_modules_transformer_modules_layer_modules_3_modules_output_layer_norm_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_3_modules_output_layer_norm_parameters_bias_,
            1e-12,
        )
        add_8 = l_self_modules_transformer_modules_layer_modules_3_modules_output_layer_norm_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_3_modules_output_layer_norm_parameters_bias_ = (None)
        linear_24 = torch._C._nn.linear(
            ffn_output_3,
            l_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_q_lin_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_q_lin_parameters_bias_,
        )
        l_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_q_lin_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_q_lin_parameters_bias_ = (None)
        view_16 = linear_24.view(1, -1, 12, 64)
        linear_24 = None
        q_4 = view_16.transpose(1, 2)
        view_16 = None
        linear_25 = torch._C._nn.linear(
            ffn_output_3,
            l_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_k_lin_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_k_lin_parameters_bias_,
        )
        l_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_k_lin_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_k_lin_parameters_bias_ = (None)
        view_17 = linear_25.view(1, -1, 12, 64)
        linear_25 = None
        k_4 = view_17.transpose(1, 2)
        view_17 = None
        linear_26 = torch._C._nn.linear(
            ffn_output_3,
            l_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_v_lin_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_v_lin_parameters_bias_,
        )
        l_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_v_lin_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_v_lin_parameters_bias_ = (None)
        view_18 = linear_26.view(1, -1, 12, 64)
        linear_26 = None
        v_4 = view_18.transpose(1, 2)
        view_18 = None
        attn_output_12 = torch._C._nn.scaled_dot_product_attention(
            q_4, k_4, v_4, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        q_4 = k_4 = v_4 = None
        transpose_19 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        contiguous_4 = transpose_19.contiguous()
        transpose_19 = None
        attn_output_13 = contiguous_4.view(1, -1, 768)
        contiguous_4 = None
        attn_output_14 = torch._C._nn.linear(
            attn_output_13,
            l_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_out_lin_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_out_lin_parameters_bias_,
        )
        attn_output_13 = l_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_out_lin_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_out_lin_parameters_bias_ = (None)
        add_9 = attn_output_14 + ffn_output_3
        attn_output_14 = ffn_output_3 = None
        sa_output_4 = torch.nn.functional.layer_norm(
            add_9,
            (768,),
            l_self_modules_transformer_modules_layer_modules_4_modules_sa_layer_norm_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_4_modules_sa_layer_norm_parameters_bias_,
            1e-12,
        )
        add_9 = l_self_modules_transformer_modules_layer_modules_4_modules_sa_layer_norm_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_4_modules_sa_layer_norm_parameters_bias_ = (None)
        x_16 = torch._C._nn.linear(
            sa_output_4,
            l_self_modules_transformer_modules_layer_modules_4_modules_ffn_modules_lin1_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_4_modules_ffn_modules_lin1_parameters_bias_,
        )
        l_self_modules_transformer_modules_layer_modules_4_modules_ffn_modules_lin1_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_4_modules_ffn_modules_lin1_parameters_bias_ = (None)
        x_17 = torch._C._nn.gelu(x_16)
        x_16 = None
        x_18 = torch._C._nn.linear(
            x_17,
            l_self_modules_transformer_modules_layer_modules_4_modules_ffn_modules_lin2_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_4_modules_ffn_modules_lin2_parameters_bias_,
        )
        x_17 = l_self_modules_transformer_modules_layer_modules_4_modules_ffn_modules_lin2_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_4_modules_ffn_modules_lin2_parameters_bias_ = (None)
        x_19 = torch.nn.functional.dropout(x_18, 0.1, False, False)
        x_18 = None
        add_10 = x_19 + sa_output_4
        x_19 = sa_output_4 = None
        ffn_output_4 = torch.nn.functional.layer_norm(
            add_10,
            (768,),
            l_self_modules_transformer_modules_layer_modules_4_modules_output_layer_norm_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_4_modules_output_layer_norm_parameters_bias_,
            1e-12,
        )
        add_10 = l_self_modules_transformer_modules_layer_modules_4_modules_output_layer_norm_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_4_modules_output_layer_norm_parameters_bias_ = (None)
        linear_30 = torch._C._nn.linear(
            ffn_output_4,
            l_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_q_lin_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_q_lin_parameters_bias_,
        )
        l_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_q_lin_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_q_lin_parameters_bias_ = (None)
        view_20 = linear_30.view(1, -1, 12, 64)
        linear_30 = None
        q_5 = view_20.transpose(1, 2)
        view_20 = None
        linear_31 = torch._C._nn.linear(
            ffn_output_4,
            l_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_k_lin_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_k_lin_parameters_bias_,
        )
        l_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_k_lin_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_k_lin_parameters_bias_ = (None)
        view_21 = linear_31.view(1, -1, 12, 64)
        linear_31 = None
        k_5 = view_21.transpose(1, 2)
        view_21 = None
        linear_32 = torch._C._nn.linear(
            ffn_output_4,
            l_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_v_lin_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_v_lin_parameters_bias_,
        )
        l_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_v_lin_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_v_lin_parameters_bias_ = (None)
        view_22 = linear_32.view(1, -1, 12, 64)
        linear_32 = None
        v_5 = view_22.transpose(1, 2)
        view_22 = None
        attn_output_15 = torch._C._nn.scaled_dot_product_attention(
            q_5, k_5, v_5, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        q_5 = k_5 = v_5 = attention_mask = None
        transpose_23 = attn_output_15.transpose(1, 2)
        attn_output_15 = None
        contiguous_5 = transpose_23.contiguous()
        transpose_23 = None
        attn_output_16 = contiguous_5.view(1, -1, 768)
        contiguous_5 = None
        attn_output_17 = torch._C._nn.linear(
            attn_output_16,
            l_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_out_lin_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_out_lin_parameters_bias_,
        )
        attn_output_16 = l_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_out_lin_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_5_modules_attention_modules_out_lin_parameters_bias_ = (None)
        add_11 = attn_output_17 + ffn_output_4
        attn_output_17 = ffn_output_4 = None
        sa_output_5 = torch.nn.functional.layer_norm(
            add_11,
            (768,),
            l_self_modules_transformer_modules_layer_modules_5_modules_sa_layer_norm_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_5_modules_sa_layer_norm_parameters_bias_,
            1e-12,
        )
        add_11 = l_self_modules_transformer_modules_layer_modules_5_modules_sa_layer_norm_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_5_modules_sa_layer_norm_parameters_bias_ = (None)
        x_20 = torch._C._nn.linear(
            sa_output_5,
            l_self_modules_transformer_modules_layer_modules_5_modules_ffn_modules_lin1_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_5_modules_ffn_modules_lin1_parameters_bias_,
        )
        l_self_modules_transformer_modules_layer_modules_5_modules_ffn_modules_lin1_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_5_modules_ffn_modules_lin1_parameters_bias_ = (None)
        x_21 = torch._C._nn.gelu(x_20)
        x_20 = None
        x_22 = torch._C._nn.linear(
            x_21,
            l_self_modules_transformer_modules_layer_modules_5_modules_ffn_modules_lin2_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_5_modules_ffn_modules_lin2_parameters_bias_,
        )
        x_21 = l_self_modules_transformer_modules_layer_modules_5_modules_ffn_modules_lin2_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_5_modules_ffn_modules_lin2_parameters_bias_ = (None)
        x_23 = torch.nn.functional.dropout(x_22, 0.1, False, False)
        x_22 = None
        add_12 = x_23 + sa_output_5
        x_23 = sa_output_5 = None
        ffn_output_5 = torch.nn.functional.layer_norm(
            add_12,
            (768,),
            l_self_modules_transformer_modules_layer_modules_5_modules_output_layer_norm_parameters_weight_,
            l_self_modules_transformer_modules_layer_modules_5_modules_output_layer_norm_parameters_bias_,
            1e-12,
        )
        add_12 = l_self_modules_transformer_modules_layer_modules_5_modules_output_layer_norm_parameters_weight_ = l_self_modules_transformer_modules_layer_modules_5_modules_output_layer_norm_parameters_bias_ = (None)
        return (ffn_output_5,)
