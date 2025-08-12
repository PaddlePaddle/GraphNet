import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_attention_mask_: torch.Tensor,
        L_self_buffers_position_ids_: torch.Tensor,
        L_self_modules_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_position_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_token_type_ids_: torch.Tensor,
        L_self_modules_layer_norm_emb_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm_emb_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_0_modules_q_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_0_modules_q_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_0_modules_k_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_0_modules_k_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_0_modules_v_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_0_modules_v_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_0_modules_out_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_0_modules_out_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_0_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_0_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_0_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_0_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_1_modules_q_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_1_modules_q_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_1_modules_k_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_1_modules_k_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_1_modules_v_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_1_modules_v_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_1_modules_out_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_1_modules_out_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_1_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_1_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_1_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_1_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_2_modules_q_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_2_modules_q_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_2_modules_k_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_2_modules_k_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_2_modules_v_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_2_modules_v_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_2_modules_out_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_2_modules_out_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_2_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_2_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_2_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_2_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_3_modules_q_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_3_modules_q_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_3_modules_k_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_3_modules_k_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_3_modules_v_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_3_modules_v_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_3_modules_out_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_3_modules_out_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_3_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_3_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_3_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_3_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_4_modules_q_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_4_modules_q_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_4_modules_k_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_4_modules_k_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_4_modules_v_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_4_modules_v_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_4_modules_out_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_4_modules_out_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_4_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_4_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_4_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_4_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_5_modules_q_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_5_modules_q_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_5_modules_k_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_5_modules_k_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_5_modules_v_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_5_modules_v_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_5_modules_out_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_5_modules_out_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_5_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_5_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_5_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_5_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_6_modules_q_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_6_modules_q_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_6_modules_k_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_6_modules_k_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_6_modules_v_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_6_modules_v_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_6_modules_out_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_6_modules_out_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_6_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_6_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_6_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_6_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_7_modules_q_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_7_modules_q_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_7_modules_k_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_7_modules_k_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_7_modules_v_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_7_modules_v_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_7_modules_out_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_7_modules_out_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_7_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_7_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_7_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_7_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_8_modules_q_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_8_modules_q_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_8_modules_k_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_8_modules_k_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_8_modules_v_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_8_modules_v_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_8_modules_out_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_8_modules_out_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_8_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_8_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_8_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_8_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_9_modules_q_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_9_modules_q_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_9_modules_k_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_9_modules_k_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_9_modules_v_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_9_modules_v_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_9_modules_out_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_9_modules_out_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_9_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_9_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_9_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_9_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_9_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_9_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_9_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_9_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_10_modules_q_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_10_modules_q_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_10_modules_k_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_10_modules_k_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_10_modules_v_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_10_modules_v_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_10_modules_out_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_10_modules_out_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_10_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_10_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_10_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_10_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_10_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_10_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_10_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_10_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_11_modules_q_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_11_modules_q_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_11_modules_k_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_11_modules_k_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_11_modules_v_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_11_modules_v_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_11_modules_out_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_11_modules_out_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_11_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_11_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_11_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_11_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_11_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_11_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_11_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_11_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_input_ids_ = L_input_ids_
        l_attention_mask_ = L_attention_mask_
        l_self_buffers_position_ids_ = L_self_buffers_position_ids_
        l_self_modules_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_parameters_weight_
        )
        l_self_modules_position_embeddings_parameters_weight_ = (
            L_self_modules_position_embeddings_parameters_weight_
        )
        l_token_type_ids_ = L_token_type_ids_
        l_self_modules_layer_norm_emb_parameters_weight_ = (
            L_self_modules_layer_norm_emb_parameters_weight_
        )
        l_self_modules_layer_norm_emb_parameters_bias_ = (
            L_self_modules_layer_norm_emb_parameters_bias_
        )
        l_self_modules_attentions_modules_0_modules_q_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_0_modules_q_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_0_modules_q_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_0_modules_q_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_0_modules_k_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_0_modules_k_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_0_modules_k_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_0_modules_k_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_0_modules_v_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_0_modules_v_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_0_modules_v_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_0_modules_v_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_0_modules_out_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_0_modules_out_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_0_modules_out_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_0_modules_out_lin_parameters_bias_
        )
        l_self_modules_layer_norm1_modules_0_parameters_weight_ = (
            L_self_modules_layer_norm1_modules_0_parameters_weight_
        )
        l_self_modules_layer_norm1_modules_0_parameters_bias_ = (
            L_self_modules_layer_norm1_modules_0_parameters_bias_
        )
        l_self_modules_ffns_modules_0_modules_lin1_parameters_weight_ = (
            L_self_modules_ffns_modules_0_modules_lin1_parameters_weight_
        )
        l_self_modules_ffns_modules_0_modules_lin1_parameters_bias_ = (
            L_self_modules_ffns_modules_0_modules_lin1_parameters_bias_
        )
        l_self_modules_ffns_modules_0_modules_lin2_parameters_weight_ = (
            L_self_modules_ffns_modules_0_modules_lin2_parameters_weight_
        )
        l_self_modules_ffns_modules_0_modules_lin2_parameters_bias_ = (
            L_self_modules_ffns_modules_0_modules_lin2_parameters_bias_
        )
        l_self_modules_layer_norm2_modules_0_parameters_weight_ = (
            L_self_modules_layer_norm2_modules_0_parameters_weight_
        )
        l_self_modules_layer_norm2_modules_0_parameters_bias_ = (
            L_self_modules_layer_norm2_modules_0_parameters_bias_
        )
        l_self_modules_attentions_modules_1_modules_q_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_1_modules_q_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_1_modules_q_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_1_modules_q_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_1_modules_k_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_1_modules_k_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_1_modules_k_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_1_modules_k_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_1_modules_v_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_1_modules_v_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_1_modules_v_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_1_modules_v_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_1_modules_out_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_1_modules_out_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_1_modules_out_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_1_modules_out_lin_parameters_bias_
        )
        l_self_modules_layer_norm1_modules_1_parameters_weight_ = (
            L_self_modules_layer_norm1_modules_1_parameters_weight_
        )
        l_self_modules_layer_norm1_modules_1_parameters_bias_ = (
            L_self_modules_layer_norm1_modules_1_parameters_bias_
        )
        l_self_modules_ffns_modules_1_modules_lin1_parameters_weight_ = (
            L_self_modules_ffns_modules_1_modules_lin1_parameters_weight_
        )
        l_self_modules_ffns_modules_1_modules_lin1_parameters_bias_ = (
            L_self_modules_ffns_modules_1_modules_lin1_parameters_bias_
        )
        l_self_modules_ffns_modules_1_modules_lin2_parameters_weight_ = (
            L_self_modules_ffns_modules_1_modules_lin2_parameters_weight_
        )
        l_self_modules_ffns_modules_1_modules_lin2_parameters_bias_ = (
            L_self_modules_ffns_modules_1_modules_lin2_parameters_bias_
        )
        l_self_modules_layer_norm2_modules_1_parameters_weight_ = (
            L_self_modules_layer_norm2_modules_1_parameters_weight_
        )
        l_self_modules_layer_norm2_modules_1_parameters_bias_ = (
            L_self_modules_layer_norm2_modules_1_parameters_bias_
        )
        l_self_modules_attentions_modules_2_modules_q_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_2_modules_q_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_2_modules_q_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_2_modules_q_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_2_modules_k_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_2_modules_k_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_2_modules_k_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_2_modules_k_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_2_modules_v_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_2_modules_v_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_2_modules_v_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_2_modules_v_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_2_modules_out_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_2_modules_out_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_2_modules_out_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_2_modules_out_lin_parameters_bias_
        )
        l_self_modules_layer_norm1_modules_2_parameters_weight_ = (
            L_self_modules_layer_norm1_modules_2_parameters_weight_
        )
        l_self_modules_layer_norm1_modules_2_parameters_bias_ = (
            L_self_modules_layer_norm1_modules_2_parameters_bias_
        )
        l_self_modules_ffns_modules_2_modules_lin1_parameters_weight_ = (
            L_self_modules_ffns_modules_2_modules_lin1_parameters_weight_
        )
        l_self_modules_ffns_modules_2_modules_lin1_parameters_bias_ = (
            L_self_modules_ffns_modules_2_modules_lin1_parameters_bias_
        )
        l_self_modules_ffns_modules_2_modules_lin2_parameters_weight_ = (
            L_self_modules_ffns_modules_2_modules_lin2_parameters_weight_
        )
        l_self_modules_ffns_modules_2_modules_lin2_parameters_bias_ = (
            L_self_modules_ffns_modules_2_modules_lin2_parameters_bias_
        )
        l_self_modules_layer_norm2_modules_2_parameters_weight_ = (
            L_self_modules_layer_norm2_modules_2_parameters_weight_
        )
        l_self_modules_layer_norm2_modules_2_parameters_bias_ = (
            L_self_modules_layer_norm2_modules_2_parameters_bias_
        )
        l_self_modules_attentions_modules_3_modules_q_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_3_modules_q_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_3_modules_q_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_3_modules_q_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_3_modules_k_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_3_modules_k_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_3_modules_k_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_3_modules_k_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_3_modules_v_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_3_modules_v_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_3_modules_v_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_3_modules_v_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_3_modules_out_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_3_modules_out_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_3_modules_out_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_3_modules_out_lin_parameters_bias_
        )
        l_self_modules_layer_norm1_modules_3_parameters_weight_ = (
            L_self_modules_layer_norm1_modules_3_parameters_weight_
        )
        l_self_modules_layer_norm1_modules_3_parameters_bias_ = (
            L_self_modules_layer_norm1_modules_3_parameters_bias_
        )
        l_self_modules_ffns_modules_3_modules_lin1_parameters_weight_ = (
            L_self_modules_ffns_modules_3_modules_lin1_parameters_weight_
        )
        l_self_modules_ffns_modules_3_modules_lin1_parameters_bias_ = (
            L_self_modules_ffns_modules_3_modules_lin1_parameters_bias_
        )
        l_self_modules_ffns_modules_3_modules_lin2_parameters_weight_ = (
            L_self_modules_ffns_modules_3_modules_lin2_parameters_weight_
        )
        l_self_modules_ffns_modules_3_modules_lin2_parameters_bias_ = (
            L_self_modules_ffns_modules_3_modules_lin2_parameters_bias_
        )
        l_self_modules_layer_norm2_modules_3_parameters_weight_ = (
            L_self_modules_layer_norm2_modules_3_parameters_weight_
        )
        l_self_modules_layer_norm2_modules_3_parameters_bias_ = (
            L_self_modules_layer_norm2_modules_3_parameters_bias_
        )
        l_self_modules_attentions_modules_4_modules_q_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_4_modules_q_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_4_modules_q_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_4_modules_q_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_4_modules_k_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_4_modules_k_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_4_modules_k_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_4_modules_k_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_4_modules_v_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_4_modules_v_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_4_modules_v_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_4_modules_v_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_4_modules_out_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_4_modules_out_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_4_modules_out_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_4_modules_out_lin_parameters_bias_
        )
        l_self_modules_layer_norm1_modules_4_parameters_weight_ = (
            L_self_modules_layer_norm1_modules_4_parameters_weight_
        )
        l_self_modules_layer_norm1_modules_4_parameters_bias_ = (
            L_self_modules_layer_norm1_modules_4_parameters_bias_
        )
        l_self_modules_ffns_modules_4_modules_lin1_parameters_weight_ = (
            L_self_modules_ffns_modules_4_modules_lin1_parameters_weight_
        )
        l_self_modules_ffns_modules_4_modules_lin1_parameters_bias_ = (
            L_self_modules_ffns_modules_4_modules_lin1_parameters_bias_
        )
        l_self_modules_ffns_modules_4_modules_lin2_parameters_weight_ = (
            L_self_modules_ffns_modules_4_modules_lin2_parameters_weight_
        )
        l_self_modules_ffns_modules_4_modules_lin2_parameters_bias_ = (
            L_self_modules_ffns_modules_4_modules_lin2_parameters_bias_
        )
        l_self_modules_layer_norm2_modules_4_parameters_weight_ = (
            L_self_modules_layer_norm2_modules_4_parameters_weight_
        )
        l_self_modules_layer_norm2_modules_4_parameters_bias_ = (
            L_self_modules_layer_norm2_modules_4_parameters_bias_
        )
        l_self_modules_attentions_modules_5_modules_q_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_5_modules_q_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_5_modules_q_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_5_modules_q_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_5_modules_k_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_5_modules_k_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_5_modules_k_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_5_modules_k_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_5_modules_v_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_5_modules_v_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_5_modules_v_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_5_modules_v_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_5_modules_out_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_5_modules_out_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_5_modules_out_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_5_modules_out_lin_parameters_bias_
        )
        l_self_modules_layer_norm1_modules_5_parameters_weight_ = (
            L_self_modules_layer_norm1_modules_5_parameters_weight_
        )
        l_self_modules_layer_norm1_modules_5_parameters_bias_ = (
            L_self_modules_layer_norm1_modules_5_parameters_bias_
        )
        l_self_modules_ffns_modules_5_modules_lin1_parameters_weight_ = (
            L_self_modules_ffns_modules_5_modules_lin1_parameters_weight_
        )
        l_self_modules_ffns_modules_5_modules_lin1_parameters_bias_ = (
            L_self_modules_ffns_modules_5_modules_lin1_parameters_bias_
        )
        l_self_modules_ffns_modules_5_modules_lin2_parameters_weight_ = (
            L_self_modules_ffns_modules_5_modules_lin2_parameters_weight_
        )
        l_self_modules_ffns_modules_5_modules_lin2_parameters_bias_ = (
            L_self_modules_ffns_modules_5_modules_lin2_parameters_bias_
        )
        l_self_modules_layer_norm2_modules_5_parameters_weight_ = (
            L_self_modules_layer_norm2_modules_5_parameters_weight_
        )
        l_self_modules_layer_norm2_modules_5_parameters_bias_ = (
            L_self_modules_layer_norm2_modules_5_parameters_bias_
        )
        l_self_modules_attentions_modules_6_modules_q_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_6_modules_q_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_6_modules_q_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_6_modules_q_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_6_modules_k_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_6_modules_k_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_6_modules_k_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_6_modules_k_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_6_modules_v_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_6_modules_v_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_6_modules_v_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_6_modules_v_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_6_modules_out_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_6_modules_out_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_6_modules_out_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_6_modules_out_lin_parameters_bias_
        )
        l_self_modules_layer_norm1_modules_6_parameters_weight_ = (
            L_self_modules_layer_norm1_modules_6_parameters_weight_
        )
        l_self_modules_layer_norm1_modules_6_parameters_bias_ = (
            L_self_modules_layer_norm1_modules_6_parameters_bias_
        )
        l_self_modules_ffns_modules_6_modules_lin1_parameters_weight_ = (
            L_self_modules_ffns_modules_6_modules_lin1_parameters_weight_
        )
        l_self_modules_ffns_modules_6_modules_lin1_parameters_bias_ = (
            L_self_modules_ffns_modules_6_modules_lin1_parameters_bias_
        )
        l_self_modules_ffns_modules_6_modules_lin2_parameters_weight_ = (
            L_self_modules_ffns_modules_6_modules_lin2_parameters_weight_
        )
        l_self_modules_ffns_modules_6_modules_lin2_parameters_bias_ = (
            L_self_modules_ffns_modules_6_modules_lin2_parameters_bias_
        )
        l_self_modules_layer_norm2_modules_6_parameters_weight_ = (
            L_self_modules_layer_norm2_modules_6_parameters_weight_
        )
        l_self_modules_layer_norm2_modules_6_parameters_bias_ = (
            L_self_modules_layer_norm2_modules_6_parameters_bias_
        )
        l_self_modules_attentions_modules_7_modules_q_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_7_modules_q_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_7_modules_q_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_7_modules_q_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_7_modules_k_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_7_modules_k_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_7_modules_k_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_7_modules_k_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_7_modules_v_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_7_modules_v_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_7_modules_v_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_7_modules_v_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_7_modules_out_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_7_modules_out_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_7_modules_out_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_7_modules_out_lin_parameters_bias_
        )
        l_self_modules_layer_norm1_modules_7_parameters_weight_ = (
            L_self_modules_layer_norm1_modules_7_parameters_weight_
        )
        l_self_modules_layer_norm1_modules_7_parameters_bias_ = (
            L_self_modules_layer_norm1_modules_7_parameters_bias_
        )
        l_self_modules_ffns_modules_7_modules_lin1_parameters_weight_ = (
            L_self_modules_ffns_modules_7_modules_lin1_parameters_weight_
        )
        l_self_modules_ffns_modules_7_modules_lin1_parameters_bias_ = (
            L_self_modules_ffns_modules_7_modules_lin1_parameters_bias_
        )
        l_self_modules_ffns_modules_7_modules_lin2_parameters_weight_ = (
            L_self_modules_ffns_modules_7_modules_lin2_parameters_weight_
        )
        l_self_modules_ffns_modules_7_modules_lin2_parameters_bias_ = (
            L_self_modules_ffns_modules_7_modules_lin2_parameters_bias_
        )
        l_self_modules_layer_norm2_modules_7_parameters_weight_ = (
            L_self_modules_layer_norm2_modules_7_parameters_weight_
        )
        l_self_modules_layer_norm2_modules_7_parameters_bias_ = (
            L_self_modules_layer_norm2_modules_7_parameters_bias_
        )
        l_self_modules_attentions_modules_8_modules_q_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_8_modules_q_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_8_modules_q_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_8_modules_q_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_8_modules_k_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_8_modules_k_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_8_modules_k_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_8_modules_k_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_8_modules_v_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_8_modules_v_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_8_modules_v_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_8_modules_v_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_8_modules_out_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_8_modules_out_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_8_modules_out_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_8_modules_out_lin_parameters_bias_
        )
        l_self_modules_layer_norm1_modules_8_parameters_weight_ = (
            L_self_modules_layer_norm1_modules_8_parameters_weight_
        )
        l_self_modules_layer_norm1_modules_8_parameters_bias_ = (
            L_self_modules_layer_norm1_modules_8_parameters_bias_
        )
        l_self_modules_ffns_modules_8_modules_lin1_parameters_weight_ = (
            L_self_modules_ffns_modules_8_modules_lin1_parameters_weight_
        )
        l_self_modules_ffns_modules_8_modules_lin1_parameters_bias_ = (
            L_self_modules_ffns_modules_8_modules_lin1_parameters_bias_
        )
        l_self_modules_ffns_modules_8_modules_lin2_parameters_weight_ = (
            L_self_modules_ffns_modules_8_modules_lin2_parameters_weight_
        )
        l_self_modules_ffns_modules_8_modules_lin2_parameters_bias_ = (
            L_self_modules_ffns_modules_8_modules_lin2_parameters_bias_
        )
        l_self_modules_layer_norm2_modules_8_parameters_weight_ = (
            L_self_modules_layer_norm2_modules_8_parameters_weight_
        )
        l_self_modules_layer_norm2_modules_8_parameters_bias_ = (
            L_self_modules_layer_norm2_modules_8_parameters_bias_
        )
        l_self_modules_attentions_modules_9_modules_q_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_9_modules_q_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_9_modules_q_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_9_modules_q_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_9_modules_k_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_9_modules_k_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_9_modules_k_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_9_modules_k_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_9_modules_v_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_9_modules_v_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_9_modules_v_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_9_modules_v_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_9_modules_out_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_9_modules_out_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_9_modules_out_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_9_modules_out_lin_parameters_bias_
        )
        l_self_modules_layer_norm1_modules_9_parameters_weight_ = (
            L_self_modules_layer_norm1_modules_9_parameters_weight_
        )
        l_self_modules_layer_norm1_modules_9_parameters_bias_ = (
            L_self_modules_layer_norm1_modules_9_parameters_bias_
        )
        l_self_modules_ffns_modules_9_modules_lin1_parameters_weight_ = (
            L_self_modules_ffns_modules_9_modules_lin1_parameters_weight_
        )
        l_self_modules_ffns_modules_9_modules_lin1_parameters_bias_ = (
            L_self_modules_ffns_modules_9_modules_lin1_parameters_bias_
        )
        l_self_modules_ffns_modules_9_modules_lin2_parameters_weight_ = (
            L_self_modules_ffns_modules_9_modules_lin2_parameters_weight_
        )
        l_self_modules_ffns_modules_9_modules_lin2_parameters_bias_ = (
            L_self_modules_ffns_modules_9_modules_lin2_parameters_bias_
        )
        l_self_modules_layer_norm2_modules_9_parameters_weight_ = (
            L_self_modules_layer_norm2_modules_9_parameters_weight_
        )
        l_self_modules_layer_norm2_modules_9_parameters_bias_ = (
            L_self_modules_layer_norm2_modules_9_parameters_bias_
        )
        l_self_modules_attentions_modules_10_modules_q_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_10_modules_q_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_10_modules_q_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_10_modules_q_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_10_modules_k_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_10_modules_k_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_10_modules_k_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_10_modules_k_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_10_modules_v_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_10_modules_v_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_10_modules_v_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_10_modules_v_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_10_modules_out_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_10_modules_out_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_10_modules_out_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_10_modules_out_lin_parameters_bias_
        )
        l_self_modules_layer_norm1_modules_10_parameters_weight_ = (
            L_self_modules_layer_norm1_modules_10_parameters_weight_
        )
        l_self_modules_layer_norm1_modules_10_parameters_bias_ = (
            L_self_modules_layer_norm1_modules_10_parameters_bias_
        )
        l_self_modules_ffns_modules_10_modules_lin1_parameters_weight_ = (
            L_self_modules_ffns_modules_10_modules_lin1_parameters_weight_
        )
        l_self_modules_ffns_modules_10_modules_lin1_parameters_bias_ = (
            L_self_modules_ffns_modules_10_modules_lin1_parameters_bias_
        )
        l_self_modules_ffns_modules_10_modules_lin2_parameters_weight_ = (
            L_self_modules_ffns_modules_10_modules_lin2_parameters_weight_
        )
        l_self_modules_ffns_modules_10_modules_lin2_parameters_bias_ = (
            L_self_modules_ffns_modules_10_modules_lin2_parameters_bias_
        )
        l_self_modules_layer_norm2_modules_10_parameters_weight_ = (
            L_self_modules_layer_norm2_modules_10_parameters_weight_
        )
        l_self_modules_layer_norm2_modules_10_parameters_bias_ = (
            L_self_modules_layer_norm2_modules_10_parameters_bias_
        )
        l_self_modules_attentions_modules_11_modules_q_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_11_modules_q_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_11_modules_q_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_11_modules_q_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_11_modules_k_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_11_modules_k_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_11_modules_k_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_11_modules_k_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_11_modules_v_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_11_modules_v_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_11_modules_v_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_11_modules_v_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_11_modules_out_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_11_modules_out_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_11_modules_out_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_11_modules_out_lin_parameters_bias_
        )
        l_self_modules_layer_norm1_modules_11_parameters_weight_ = (
            L_self_modules_layer_norm1_modules_11_parameters_weight_
        )
        l_self_modules_layer_norm1_modules_11_parameters_bias_ = (
            L_self_modules_layer_norm1_modules_11_parameters_bias_
        )
        l_self_modules_ffns_modules_11_modules_lin1_parameters_weight_ = (
            L_self_modules_ffns_modules_11_modules_lin1_parameters_weight_
        )
        l_self_modules_ffns_modules_11_modules_lin1_parameters_bias_ = (
            L_self_modules_ffns_modules_11_modules_lin1_parameters_bias_
        )
        l_self_modules_ffns_modules_11_modules_lin2_parameters_weight_ = (
            L_self_modules_ffns_modules_11_modules_lin2_parameters_weight_
        )
        l_self_modules_ffns_modules_11_modules_lin2_parameters_bias_ = (
            L_self_modules_ffns_modules_11_modules_lin2_parameters_bias_
        )
        l_self_modules_layer_norm2_modules_11_parameters_weight_ = (
            L_self_modules_layer_norm2_modules_11_parameters_weight_
        )
        l_self_modules_layer_norm2_modules_11_parameters_bias_ = (
            L_self_modules_layer_norm2_modules_11_parameters_bias_
        )
        ne = l_input_ids_ != 2
        sum_1 = ne.sum(dim=1)
        ne = None
        lengths = sum_1.long()
        sum_1 = None
        max_1 = lengths.max()
        lengths = None
        item = max_1.item()
        max_1 = None
        le_1 = item <= 20
        item = None
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(
            le_1, "Runtime assertion failed for expression u0 <= 20 on node 'le_1'"
        )
        le_1 = _assert_scalar_default = None
        alen = torch.arange(20, dtype=torch.int64, device=device(type="cuda", index=0))
        alen = None
        position_ids = l_self_buffers_position_ids_[
            (slice(None, None, None), slice(None, 20, None))
        ]
        l_self_buffers_position_ids_ = None
        position_ids_1 = position_ids.expand((1, 20))
        position_ids = None
        input_ids = l_input_ids_[(slice(None, None, None), slice(-20, None, None))]
        l_input_ids_ = None
        position_ids_2 = position_ids_1[
            (slice(None, None, None), slice(-20, None, None))
        ]
        position_ids_1 = None
        mask = l_attention_mask_[(slice(None, None, None), slice(-20, None, None))]
        attn_mask = l_attention_mask_[(slice(None, None, None), slice(-20, None, None))]
        l_attention_mask_ = None
        inputs_embeds = torch.nn.functional.embedding(
            input_ids,
            l_self_modules_embeddings_parameters_weight_,
            2,
            None,
            2.0,
            False,
            False,
        )
        input_ids = None
        embedding_1 = torch.nn.functional.embedding(
            position_ids_2,
            l_self_modules_position_embeddings_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        position_ids_2 = l_self_modules_position_embeddings_parameters_weight_ = None
        expand_as = embedding_1.expand_as(inputs_embeds)
        embedding_1 = None
        tensor = inputs_embeds + expand_as
        inputs_embeds = expand_as = None
        embedding_2 = torch.nn.functional.embedding(
            l_token_type_ids_,
            l_self_modules_embeddings_parameters_weight_,
            2,
            None,
            2.0,
            False,
            False,
        )
        l_token_type_ids_ = l_self_modules_embeddings_parameters_weight_ = None
        tensor_1 = tensor + embedding_2
        tensor = embedding_2 = None
        tensor_2 = torch.nn.functional.layer_norm(
            tensor_1,
            (768,),
            l_self_modules_layer_norm_emb_parameters_weight_,
            l_self_modules_layer_norm_emb_parameters_bias_,
            1e-06,
        )
        tensor_1 = (
            l_self_modules_layer_norm_emb_parameters_weight_
        ) = l_self_modules_layer_norm_emb_parameters_bias_ = None
        tensor_3 = torch.nn.functional.dropout(tensor_2, p=0.1, training=False)
        tensor_2 = None
        unsqueeze = mask.unsqueeze(-1)
        to = unsqueeze.to(torch.float32)
        unsqueeze = None
        tensor_3 *= to
        tensor_4 = tensor_3
        tensor_3 = to = None
        linear = torch._C._nn.linear(
            tensor_4,
            l_self_modules_attentions_modules_0_modules_q_lin_parameters_weight_,
            l_self_modules_attentions_modules_0_modules_q_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_0_modules_q_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_0_modules_q_lin_parameters_bias_
        ) = None
        view = linear.view(1, -1, 12, 64)
        linear = None
        q = view.transpose(1, 2)
        view = None
        k = torch._C._nn.linear(
            tensor_4,
            l_self_modules_attentions_modules_0_modules_k_lin_parameters_weight_,
            l_self_modules_attentions_modules_0_modules_k_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_0_modules_k_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_0_modules_k_lin_parameters_bias_
        ) = None
        v = torch._C._nn.linear(
            tensor_4,
            l_self_modules_attentions_modules_0_modules_v_lin_parameters_weight_,
            l_self_modules_attentions_modules_0_modules_v_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_0_modules_v_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_0_modules_v_lin_parameters_bias_
        ) = None
        view_1 = k.view(1, -1, 12, 64)
        k = None
        k_1 = view_1.transpose(1, 2)
        view_1 = None
        view_2 = v.view(1, -1, 12, 64)
        v = None
        v_1 = view_2.transpose(1, 2)
        view_2 = None
        q_1 = q / 8.0
        q = None
        transpose_3 = k_1.transpose(2, 3)
        scores = torch.matmul(q_1, transpose_3)
        q_1 = transpose_3 = None
        eq = attn_mask.__eq__(0)
        view_3 = eq.view((1, 1, 1, -1))
        eq = None
        mask_1 = view_3.expand_as(scores)
        view_3 = None
        masked_fill_ = scores.masked_fill_(mask_1, -3.4028234663852886e38)
        mask_1 = masked_fill_ = None
        float_1 = scores.float()
        softmax = torch.nn.functional.softmax(float_1, dim=-1)
        float_1 = None
        weights = softmax.type_as(scores)
        softmax = scores = None
        weights_1 = torch.nn.functional.dropout(weights, p=0.1, training=False)
        weights = None
        context = torch.matmul(weights_1, v_1)
        weights_1 = None
        transpose_4 = context.transpose(1, 2)
        context = None
        contiguous = transpose_4.contiguous()
        transpose_4 = None
        context_1 = contiguous.view(1, -1, 768)
        contiguous = None
        attn = torch._C._nn.linear(
            context_1,
            l_self_modules_attentions_modules_0_modules_out_lin_parameters_weight_,
            l_self_modules_attentions_modules_0_modules_out_lin_parameters_bias_,
        )
        context_1 = (
            l_self_modules_attentions_modules_0_modules_out_lin_parameters_weight_
        ) = l_self_modules_attentions_modules_0_modules_out_lin_parameters_bias_ = None
        attn_1 = torch.nn.functional.dropout(attn, p=0.1, training=False)
        attn = None
        tensor_5 = tensor_4 + attn_1
        tensor_4 = attn_1 = None
        tensor_6 = torch.nn.functional.layer_norm(
            tensor_5,
            (768,),
            l_self_modules_layer_norm1_modules_0_parameters_weight_,
            l_self_modules_layer_norm1_modules_0_parameters_bias_,
            1e-06,
        )
        tensor_5 = (
            l_self_modules_layer_norm1_modules_0_parameters_weight_
        ) = l_self_modules_layer_norm1_modules_0_parameters_bias_ = None
        x = torch._C._nn.linear(
            tensor_6,
            l_self_modules_ffns_modules_0_modules_lin1_parameters_weight_,
            l_self_modules_ffns_modules_0_modules_lin1_parameters_bias_,
        )
        l_self_modules_ffns_modules_0_modules_lin1_parameters_weight_ = (
            l_self_modules_ffns_modules_0_modules_lin1_parameters_bias_
        ) = None
        x_1 = torch._C._nn.gelu(x)
        x = None
        x_2 = torch._C._nn.linear(
            x_1,
            l_self_modules_ffns_modules_0_modules_lin2_parameters_weight_,
            l_self_modules_ffns_modules_0_modules_lin2_parameters_bias_,
        )
        x_1 = (
            l_self_modules_ffns_modules_0_modules_lin2_parameters_weight_
        ) = l_self_modules_ffns_modules_0_modules_lin2_parameters_bias_ = None
        x_3 = torch.nn.functional.dropout(x_2, p=0.1, training=False)
        x_2 = None
        tensor_7 = tensor_6 + x_3
        tensor_6 = x_3 = None
        tensor_8 = torch.nn.functional.layer_norm(
            tensor_7,
            (768,),
            l_self_modules_layer_norm2_modules_0_parameters_weight_,
            l_self_modules_layer_norm2_modules_0_parameters_bias_,
            1e-06,
        )
        tensor_7 = (
            l_self_modules_layer_norm2_modules_0_parameters_weight_
        ) = l_self_modules_layer_norm2_modules_0_parameters_bias_ = None
        unsqueeze_1 = mask.unsqueeze(-1)
        to_1 = unsqueeze_1.to(torch.float32)
        unsqueeze_1 = None
        tensor_8 *= to_1
        tensor_9 = tensor_8
        tensor_8 = to_1 = None
        linear_6 = torch._C._nn.linear(
            tensor_9,
            l_self_modules_attentions_modules_1_modules_q_lin_parameters_weight_,
            l_self_modules_attentions_modules_1_modules_q_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_1_modules_q_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_1_modules_q_lin_parameters_bias_
        ) = None
        view_5 = linear_6.view(1, -1, 12, 64)
        linear_6 = None
        q_2 = view_5.transpose(1, 2)
        view_5 = None
        k_2 = torch._C._nn.linear(
            tensor_9,
            l_self_modules_attentions_modules_1_modules_k_lin_parameters_weight_,
            l_self_modules_attentions_modules_1_modules_k_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_1_modules_k_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_1_modules_k_lin_parameters_bias_
        ) = None
        v_2 = torch._C._nn.linear(
            tensor_9,
            l_self_modules_attentions_modules_1_modules_v_lin_parameters_weight_,
            l_self_modules_attentions_modules_1_modules_v_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_1_modules_v_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_1_modules_v_lin_parameters_bias_
        ) = None
        view_6 = k_2.view(1, -1, 12, 64)
        k_2 = None
        k_3 = view_6.transpose(1, 2)
        view_6 = None
        view_7 = v_2.view(1, -1, 12, 64)
        v_2 = None
        v_3 = view_7.transpose(1, 2)
        view_7 = None
        q_3 = q_2 / 8.0
        q_2 = None
        transpose_8 = k_3.transpose(2, 3)
        scores_1 = torch.matmul(q_3, transpose_8)
        q_3 = transpose_8 = None
        eq_1 = attn_mask.__eq__(0)
        view_8 = eq_1.view((1, 1, 1, -1))
        eq_1 = None
        mask_2 = view_8.expand_as(scores_1)
        view_8 = None
        masked_fill__1 = scores_1.masked_fill_(mask_2, -3.4028234663852886e38)
        mask_2 = masked_fill__1 = None
        float_2 = scores_1.float()
        softmax_1 = torch.nn.functional.softmax(float_2, dim=-1)
        float_2 = None
        weights_2 = softmax_1.type_as(scores_1)
        softmax_1 = scores_1 = None
        weights_3 = torch.nn.functional.dropout(weights_2, p=0.1, training=False)
        weights_2 = None
        context_2 = torch.matmul(weights_3, v_3)
        weights_3 = None
        transpose_9 = context_2.transpose(1, 2)
        context_2 = None
        contiguous_1 = transpose_9.contiguous()
        transpose_9 = None
        context_3 = contiguous_1.view(1, -1, 768)
        contiguous_1 = None
        attn_2 = torch._C._nn.linear(
            context_3,
            l_self_modules_attentions_modules_1_modules_out_lin_parameters_weight_,
            l_self_modules_attentions_modules_1_modules_out_lin_parameters_bias_,
        )
        context_3 = (
            l_self_modules_attentions_modules_1_modules_out_lin_parameters_weight_
        ) = l_self_modules_attentions_modules_1_modules_out_lin_parameters_bias_ = None
        attn_3 = torch.nn.functional.dropout(attn_2, p=0.1, training=False)
        attn_2 = None
        tensor_10 = tensor_9 + attn_3
        tensor_9 = attn_3 = None
        tensor_11 = torch.nn.functional.layer_norm(
            tensor_10,
            (768,),
            l_self_modules_layer_norm1_modules_1_parameters_weight_,
            l_self_modules_layer_norm1_modules_1_parameters_bias_,
            1e-06,
        )
        tensor_10 = (
            l_self_modules_layer_norm1_modules_1_parameters_weight_
        ) = l_self_modules_layer_norm1_modules_1_parameters_bias_ = None
        x_4 = torch._C._nn.linear(
            tensor_11,
            l_self_modules_ffns_modules_1_modules_lin1_parameters_weight_,
            l_self_modules_ffns_modules_1_modules_lin1_parameters_bias_,
        )
        l_self_modules_ffns_modules_1_modules_lin1_parameters_weight_ = (
            l_self_modules_ffns_modules_1_modules_lin1_parameters_bias_
        ) = None
        x_5 = torch._C._nn.gelu(x_4)
        x_4 = None
        x_6 = torch._C._nn.linear(
            x_5,
            l_self_modules_ffns_modules_1_modules_lin2_parameters_weight_,
            l_self_modules_ffns_modules_1_modules_lin2_parameters_bias_,
        )
        x_5 = (
            l_self_modules_ffns_modules_1_modules_lin2_parameters_weight_
        ) = l_self_modules_ffns_modules_1_modules_lin2_parameters_bias_ = None
        x_7 = torch.nn.functional.dropout(x_6, p=0.1, training=False)
        x_6 = None
        tensor_12 = tensor_11 + x_7
        tensor_11 = x_7 = None
        tensor_13 = torch.nn.functional.layer_norm(
            tensor_12,
            (768,),
            l_self_modules_layer_norm2_modules_1_parameters_weight_,
            l_self_modules_layer_norm2_modules_1_parameters_bias_,
            1e-06,
        )
        tensor_12 = (
            l_self_modules_layer_norm2_modules_1_parameters_weight_
        ) = l_self_modules_layer_norm2_modules_1_parameters_bias_ = None
        unsqueeze_2 = mask.unsqueeze(-1)
        to_2 = unsqueeze_2.to(torch.float32)
        unsqueeze_2 = None
        tensor_13 *= to_2
        tensor_14 = tensor_13
        tensor_13 = to_2 = None
        linear_12 = torch._C._nn.linear(
            tensor_14,
            l_self_modules_attentions_modules_2_modules_q_lin_parameters_weight_,
            l_self_modules_attentions_modules_2_modules_q_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_2_modules_q_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_2_modules_q_lin_parameters_bias_
        ) = None
        view_10 = linear_12.view(1, -1, 12, 64)
        linear_12 = None
        q_4 = view_10.transpose(1, 2)
        view_10 = None
        k_4 = torch._C._nn.linear(
            tensor_14,
            l_self_modules_attentions_modules_2_modules_k_lin_parameters_weight_,
            l_self_modules_attentions_modules_2_modules_k_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_2_modules_k_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_2_modules_k_lin_parameters_bias_
        ) = None
        v_4 = torch._C._nn.linear(
            tensor_14,
            l_self_modules_attentions_modules_2_modules_v_lin_parameters_weight_,
            l_self_modules_attentions_modules_2_modules_v_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_2_modules_v_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_2_modules_v_lin_parameters_bias_
        ) = None
        view_11 = k_4.view(1, -1, 12, 64)
        k_4 = None
        k_5 = view_11.transpose(1, 2)
        view_11 = None
        view_12 = v_4.view(1, -1, 12, 64)
        v_4 = None
        v_5 = view_12.transpose(1, 2)
        view_12 = None
        q_5 = q_4 / 8.0
        q_4 = None
        transpose_13 = k_5.transpose(2, 3)
        scores_2 = torch.matmul(q_5, transpose_13)
        q_5 = transpose_13 = None
        eq_2 = attn_mask.__eq__(0)
        view_13 = eq_2.view((1, 1, 1, -1))
        eq_2 = None
        mask_3 = view_13.expand_as(scores_2)
        view_13 = None
        masked_fill__2 = scores_2.masked_fill_(mask_3, -3.4028234663852886e38)
        mask_3 = masked_fill__2 = None
        float_3 = scores_2.float()
        softmax_2 = torch.nn.functional.softmax(float_3, dim=-1)
        float_3 = None
        weights_4 = softmax_2.type_as(scores_2)
        softmax_2 = scores_2 = None
        weights_5 = torch.nn.functional.dropout(weights_4, p=0.1, training=False)
        weights_4 = None
        context_4 = torch.matmul(weights_5, v_5)
        weights_5 = None
        transpose_14 = context_4.transpose(1, 2)
        context_4 = None
        contiguous_2 = transpose_14.contiguous()
        transpose_14 = None
        context_5 = contiguous_2.view(1, -1, 768)
        contiguous_2 = None
        attn_4 = torch._C._nn.linear(
            context_5,
            l_self_modules_attentions_modules_2_modules_out_lin_parameters_weight_,
            l_self_modules_attentions_modules_2_modules_out_lin_parameters_bias_,
        )
        context_5 = (
            l_self_modules_attentions_modules_2_modules_out_lin_parameters_weight_
        ) = l_self_modules_attentions_modules_2_modules_out_lin_parameters_bias_ = None
        attn_5 = torch.nn.functional.dropout(attn_4, p=0.1, training=False)
        attn_4 = None
        tensor_15 = tensor_14 + attn_5
        tensor_14 = attn_5 = None
        tensor_16 = torch.nn.functional.layer_norm(
            tensor_15,
            (768,),
            l_self_modules_layer_norm1_modules_2_parameters_weight_,
            l_self_modules_layer_norm1_modules_2_parameters_bias_,
            1e-06,
        )
        tensor_15 = (
            l_self_modules_layer_norm1_modules_2_parameters_weight_
        ) = l_self_modules_layer_norm1_modules_2_parameters_bias_ = None
        x_8 = torch._C._nn.linear(
            tensor_16,
            l_self_modules_ffns_modules_2_modules_lin1_parameters_weight_,
            l_self_modules_ffns_modules_2_modules_lin1_parameters_bias_,
        )
        l_self_modules_ffns_modules_2_modules_lin1_parameters_weight_ = (
            l_self_modules_ffns_modules_2_modules_lin1_parameters_bias_
        ) = None
        x_9 = torch._C._nn.gelu(x_8)
        x_8 = None
        x_10 = torch._C._nn.linear(
            x_9,
            l_self_modules_ffns_modules_2_modules_lin2_parameters_weight_,
            l_self_modules_ffns_modules_2_modules_lin2_parameters_bias_,
        )
        x_9 = (
            l_self_modules_ffns_modules_2_modules_lin2_parameters_weight_
        ) = l_self_modules_ffns_modules_2_modules_lin2_parameters_bias_ = None
        x_11 = torch.nn.functional.dropout(x_10, p=0.1, training=False)
        x_10 = None
        tensor_17 = tensor_16 + x_11
        tensor_16 = x_11 = None
        tensor_18 = torch.nn.functional.layer_norm(
            tensor_17,
            (768,),
            l_self_modules_layer_norm2_modules_2_parameters_weight_,
            l_self_modules_layer_norm2_modules_2_parameters_bias_,
            1e-06,
        )
        tensor_17 = (
            l_self_modules_layer_norm2_modules_2_parameters_weight_
        ) = l_self_modules_layer_norm2_modules_2_parameters_bias_ = None
        unsqueeze_3 = mask.unsqueeze(-1)
        to_3 = unsqueeze_3.to(torch.float32)
        unsqueeze_3 = None
        tensor_18 *= to_3
        tensor_19 = tensor_18
        tensor_18 = to_3 = None
        linear_18 = torch._C._nn.linear(
            tensor_19,
            l_self_modules_attentions_modules_3_modules_q_lin_parameters_weight_,
            l_self_modules_attentions_modules_3_modules_q_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_3_modules_q_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_3_modules_q_lin_parameters_bias_
        ) = None
        view_15 = linear_18.view(1, -1, 12, 64)
        linear_18 = None
        q_6 = view_15.transpose(1, 2)
        view_15 = None
        k_6 = torch._C._nn.linear(
            tensor_19,
            l_self_modules_attentions_modules_3_modules_k_lin_parameters_weight_,
            l_self_modules_attentions_modules_3_modules_k_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_3_modules_k_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_3_modules_k_lin_parameters_bias_
        ) = None
        v_6 = torch._C._nn.linear(
            tensor_19,
            l_self_modules_attentions_modules_3_modules_v_lin_parameters_weight_,
            l_self_modules_attentions_modules_3_modules_v_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_3_modules_v_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_3_modules_v_lin_parameters_bias_
        ) = None
        view_16 = k_6.view(1, -1, 12, 64)
        k_6 = None
        k_7 = view_16.transpose(1, 2)
        view_16 = None
        view_17 = v_6.view(1, -1, 12, 64)
        v_6 = None
        v_7 = view_17.transpose(1, 2)
        view_17 = None
        q_7 = q_6 / 8.0
        q_6 = None
        transpose_18 = k_7.transpose(2, 3)
        scores_3 = torch.matmul(q_7, transpose_18)
        q_7 = transpose_18 = None
        eq_3 = attn_mask.__eq__(0)
        view_18 = eq_3.view((1, 1, 1, -1))
        eq_3 = None
        mask_4 = view_18.expand_as(scores_3)
        view_18 = None
        masked_fill__3 = scores_3.masked_fill_(mask_4, -3.4028234663852886e38)
        mask_4 = masked_fill__3 = None
        float_4 = scores_3.float()
        softmax_3 = torch.nn.functional.softmax(float_4, dim=-1)
        float_4 = None
        weights_6 = softmax_3.type_as(scores_3)
        softmax_3 = scores_3 = None
        weights_7 = torch.nn.functional.dropout(weights_6, p=0.1, training=False)
        weights_6 = None
        context_6 = torch.matmul(weights_7, v_7)
        weights_7 = None
        transpose_19 = context_6.transpose(1, 2)
        context_6 = None
        contiguous_3 = transpose_19.contiguous()
        transpose_19 = None
        context_7 = contiguous_3.view(1, -1, 768)
        contiguous_3 = None
        attn_6 = torch._C._nn.linear(
            context_7,
            l_self_modules_attentions_modules_3_modules_out_lin_parameters_weight_,
            l_self_modules_attentions_modules_3_modules_out_lin_parameters_bias_,
        )
        context_7 = (
            l_self_modules_attentions_modules_3_modules_out_lin_parameters_weight_
        ) = l_self_modules_attentions_modules_3_modules_out_lin_parameters_bias_ = None
        attn_7 = torch.nn.functional.dropout(attn_6, p=0.1, training=False)
        attn_6 = None
        tensor_20 = tensor_19 + attn_7
        tensor_19 = attn_7 = None
        tensor_21 = torch.nn.functional.layer_norm(
            tensor_20,
            (768,),
            l_self_modules_layer_norm1_modules_3_parameters_weight_,
            l_self_modules_layer_norm1_modules_3_parameters_bias_,
            1e-06,
        )
        tensor_20 = (
            l_self_modules_layer_norm1_modules_3_parameters_weight_
        ) = l_self_modules_layer_norm1_modules_3_parameters_bias_ = None
        x_12 = torch._C._nn.linear(
            tensor_21,
            l_self_modules_ffns_modules_3_modules_lin1_parameters_weight_,
            l_self_modules_ffns_modules_3_modules_lin1_parameters_bias_,
        )
        l_self_modules_ffns_modules_3_modules_lin1_parameters_weight_ = (
            l_self_modules_ffns_modules_3_modules_lin1_parameters_bias_
        ) = None
        x_13 = torch._C._nn.gelu(x_12)
        x_12 = None
        x_14 = torch._C._nn.linear(
            x_13,
            l_self_modules_ffns_modules_3_modules_lin2_parameters_weight_,
            l_self_modules_ffns_modules_3_modules_lin2_parameters_bias_,
        )
        x_13 = (
            l_self_modules_ffns_modules_3_modules_lin2_parameters_weight_
        ) = l_self_modules_ffns_modules_3_modules_lin2_parameters_bias_ = None
        x_15 = torch.nn.functional.dropout(x_14, p=0.1, training=False)
        x_14 = None
        tensor_22 = tensor_21 + x_15
        tensor_21 = x_15 = None
        tensor_23 = torch.nn.functional.layer_norm(
            tensor_22,
            (768,),
            l_self_modules_layer_norm2_modules_3_parameters_weight_,
            l_self_modules_layer_norm2_modules_3_parameters_bias_,
            1e-06,
        )
        tensor_22 = (
            l_self_modules_layer_norm2_modules_3_parameters_weight_
        ) = l_self_modules_layer_norm2_modules_3_parameters_bias_ = None
        unsqueeze_4 = mask.unsqueeze(-1)
        to_4 = unsqueeze_4.to(torch.float32)
        unsqueeze_4 = None
        tensor_23 *= to_4
        tensor_24 = tensor_23
        tensor_23 = to_4 = None
        linear_24 = torch._C._nn.linear(
            tensor_24,
            l_self_modules_attentions_modules_4_modules_q_lin_parameters_weight_,
            l_self_modules_attentions_modules_4_modules_q_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_4_modules_q_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_4_modules_q_lin_parameters_bias_
        ) = None
        view_20 = linear_24.view(1, -1, 12, 64)
        linear_24 = None
        q_8 = view_20.transpose(1, 2)
        view_20 = None
        k_8 = torch._C._nn.linear(
            tensor_24,
            l_self_modules_attentions_modules_4_modules_k_lin_parameters_weight_,
            l_self_modules_attentions_modules_4_modules_k_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_4_modules_k_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_4_modules_k_lin_parameters_bias_
        ) = None
        v_8 = torch._C._nn.linear(
            tensor_24,
            l_self_modules_attentions_modules_4_modules_v_lin_parameters_weight_,
            l_self_modules_attentions_modules_4_modules_v_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_4_modules_v_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_4_modules_v_lin_parameters_bias_
        ) = None
        view_21 = k_8.view(1, -1, 12, 64)
        k_8 = None
        k_9 = view_21.transpose(1, 2)
        view_21 = None
        view_22 = v_8.view(1, -1, 12, 64)
        v_8 = None
        v_9 = view_22.transpose(1, 2)
        view_22 = None
        q_9 = q_8 / 8.0
        q_8 = None
        transpose_23 = k_9.transpose(2, 3)
        scores_4 = torch.matmul(q_9, transpose_23)
        q_9 = transpose_23 = None
        eq_4 = attn_mask.__eq__(0)
        view_23 = eq_4.view((1, 1, 1, -1))
        eq_4 = None
        mask_5 = view_23.expand_as(scores_4)
        view_23 = None
        masked_fill__4 = scores_4.masked_fill_(mask_5, -3.4028234663852886e38)
        mask_5 = masked_fill__4 = None
        float_5 = scores_4.float()
        softmax_4 = torch.nn.functional.softmax(float_5, dim=-1)
        float_5 = None
        weights_8 = softmax_4.type_as(scores_4)
        softmax_4 = scores_4 = None
        weights_9 = torch.nn.functional.dropout(weights_8, p=0.1, training=False)
        weights_8 = None
        context_8 = torch.matmul(weights_9, v_9)
        weights_9 = None
        transpose_24 = context_8.transpose(1, 2)
        context_8 = None
        contiguous_4 = transpose_24.contiguous()
        transpose_24 = None
        context_9 = contiguous_4.view(1, -1, 768)
        contiguous_4 = None
        attn_8 = torch._C._nn.linear(
            context_9,
            l_self_modules_attentions_modules_4_modules_out_lin_parameters_weight_,
            l_self_modules_attentions_modules_4_modules_out_lin_parameters_bias_,
        )
        context_9 = (
            l_self_modules_attentions_modules_4_modules_out_lin_parameters_weight_
        ) = l_self_modules_attentions_modules_4_modules_out_lin_parameters_bias_ = None
        attn_9 = torch.nn.functional.dropout(attn_8, p=0.1, training=False)
        attn_8 = None
        tensor_25 = tensor_24 + attn_9
        tensor_24 = attn_9 = None
        tensor_26 = torch.nn.functional.layer_norm(
            tensor_25,
            (768,),
            l_self_modules_layer_norm1_modules_4_parameters_weight_,
            l_self_modules_layer_norm1_modules_4_parameters_bias_,
            1e-06,
        )
        tensor_25 = (
            l_self_modules_layer_norm1_modules_4_parameters_weight_
        ) = l_self_modules_layer_norm1_modules_4_parameters_bias_ = None
        x_16 = torch._C._nn.linear(
            tensor_26,
            l_self_modules_ffns_modules_4_modules_lin1_parameters_weight_,
            l_self_modules_ffns_modules_4_modules_lin1_parameters_bias_,
        )
        l_self_modules_ffns_modules_4_modules_lin1_parameters_weight_ = (
            l_self_modules_ffns_modules_4_modules_lin1_parameters_bias_
        ) = None
        x_17 = torch._C._nn.gelu(x_16)
        x_16 = None
        x_18 = torch._C._nn.linear(
            x_17,
            l_self_modules_ffns_modules_4_modules_lin2_parameters_weight_,
            l_self_modules_ffns_modules_4_modules_lin2_parameters_bias_,
        )
        x_17 = (
            l_self_modules_ffns_modules_4_modules_lin2_parameters_weight_
        ) = l_self_modules_ffns_modules_4_modules_lin2_parameters_bias_ = None
        x_19 = torch.nn.functional.dropout(x_18, p=0.1, training=False)
        x_18 = None
        tensor_27 = tensor_26 + x_19
        tensor_26 = x_19 = None
        tensor_28 = torch.nn.functional.layer_norm(
            tensor_27,
            (768,),
            l_self_modules_layer_norm2_modules_4_parameters_weight_,
            l_self_modules_layer_norm2_modules_4_parameters_bias_,
            1e-06,
        )
        tensor_27 = (
            l_self_modules_layer_norm2_modules_4_parameters_weight_
        ) = l_self_modules_layer_norm2_modules_4_parameters_bias_ = None
        unsqueeze_5 = mask.unsqueeze(-1)
        to_5 = unsqueeze_5.to(torch.float32)
        unsqueeze_5 = None
        tensor_28 *= to_5
        tensor_29 = tensor_28
        tensor_28 = to_5 = None
        linear_30 = torch._C._nn.linear(
            tensor_29,
            l_self_modules_attentions_modules_5_modules_q_lin_parameters_weight_,
            l_self_modules_attentions_modules_5_modules_q_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_5_modules_q_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_5_modules_q_lin_parameters_bias_
        ) = None
        view_25 = linear_30.view(1, -1, 12, 64)
        linear_30 = None
        q_10 = view_25.transpose(1, 2)
        view_25 = None
        k_10 = torch._C._nn.linear(
            tensor_29,
            l_self_modules_attentions_modules_5_modules_k_lin_parameters_weight_,
            l_self_modules_attentions_modules_5_modules_k_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_5_modules_k_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_5_modules_k_lin_parameters_bias_
        ) = None
        v_10 = torch._C._nn.linear(
            tensor_29,
            l_self_modules_attentions_modules_5_modules_v_lin_parameters_weight_,
            l_self_modules_attentions_modules_5_modules_v_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_5_modules_v_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_5_modules_v_lin_parameters_bias_
        ) = None
        view_26 = k_10.view(1, -1, 12, 64)
        k_10 = None
        k_11 = view_26.transpose(1, 2)
        view_26 = None
        view_27 = v_10.view(1, -1, 12, 64)
        v_10 = None
        v_11 = view_27.transpose(1, 2)
        view_27 = None
        q_11 = q_10 / 8.0
        q_10 = None
        transpose_28 = k_11.transpose(2, 3)
        scores_5 = torch.matmul(q_11, transpose_28)
        q_11 = transpose_28 = None
        eq_5 = attn_mask.__eq__(0)
        view_28 = eq_5.view((1, 1, 1, -1))
        eq_5 = None
        mask_6 = view_28.expand_as(scores_5)
        view_28 = None
        masked_fill__5 = scores_5.masked_fill_(mask_6, -3.4028234663852886e38)
        mask_6 = masked_fill__5 = None
        float_6 = scores_5.float()
        softmax_5 = torch.nn.functional.softmax(float_6, dim=-1)
        float_6 = None
        weights_10 = softmax_5.type_as(scores_5)
        softmax_5 = scores_5 = None
        weights_11 = torch.nn.functional.dropout(weights_10, p=0.1, training=False)
        weights_10 = None
        context_10 = torch.matmul(weights_11, v_11)
        weights_11 = None
        transpose_29 = context_10.transpose(1, 2)
        context_10 = None
        contiguous_5 = transpose_29.contiguous()
        transpose_29 = None
        context_11 = contiguous_5.view(1, -1, 768)
        contiguous_5 = None
        attn_10 = torch._C._nn.linear(
            context_11,
            l_self_modules_attentions_modules_5_modules_out_lin_parameters_weight_,
            l_self_modules_attentions_modules_5_modules_out_lin_parameters_bias_,
        )
        context_11 = (
            l_self_modules_attentions_modules_5_modules_out_lin_parameters_weight_
        ) = l_self_modules_attentions_modules_5_modules_out_lin_parameters_bias_ = None
        attn_11 = torch.nn.functional.dropout(attn_10, p=0.1, training=False)
        attn_10 = None
        tensor_30 = tensor_29 + attn_11
        tensor_29 = attn_11 = None
        tensor_31 = torch.nn.functional.layer_norm(
            tensor_30,
            (768,),
            l_self_modules_layer_norm1_modules_5_parameters_weight_,
            l_self_modules_layer_norm1_modules_5_parameters_bias_,
            1e-06,
        )
        tensor_30 = (
            l_self_modules_layer_norm1_modules_5_parameters_weight_
        ) = l_self_modules_layer_norm1_modules_5_parameters_bias_ = None
        x_20 = torch._C._nn.linear(
            tensor_31,
            l_self_modules_ffns_modules_5_modules_lin1_parameters_weight_,
            l_self_modules_ffns_modules_5_modules_lin1_parameters_bias_,
        )
        l_self_modules_ffns_modules_5_modules_lin1_parameters_weight_ = (
            l_self_modules_ffns_modules_5_modules_lin1_parameters_bias_
        ) = None
        x_21 = torch._C._nn.gelu(x_20)
        x_20 = None
        x_22 = torch._C._nn.linear(
            x_21,
            l_self_modules_ffns_modules_5_modules_lin2_parameters_weight_,
            l_self_modules_ffns_modules_5_modules_lin2_parameters_bias_,
        )
        x_21 = (
            l_self_modules_ffns_modules_5_modules_lin2_parameters_weight_
        ) = l_self_modules_ffns_modules_5_modules_lin2_parameters_bias_ = None
        x_23 = torch.nn.functional.dropout(x_22, p=0.1, training=False)
        x_22 = None
        tensor_32 = tensor_31 + x_23
        tensor_31 = x_23 = None
        tensor_33 = torch.nn.functional.layer_norm(
            tensor_32,
            (768,),
            l_self_modules_layer_norm2_modules_5_parameters_weight_,
            l_self_modules_layer_norm2_modules_5_parameters_bias_,
            1e-06,
        )
        tensor_32 = (
            l_self_modules_layer_norm2_modules_5_parameters_weight_
        ) = l_self_modules_layer_norm2_modules_5_parameters_bias_ = None
        unsqueeze_6 = mask.unsqueeze(-1)
        to_6 = unsqueeze_6.to(torch.float32)
        unsqueeze_6 = None
        tensor_33 *= to_6
        tensor_34 = tensor_33
        tensor_33 = to_6 = None
        linear_36 = torch._C._nn.linear(
            tensor_34,
            l_self_modules_attentions_modules_6_modules_q_lin_parameters_weight_,
            l_self_modules_attentions_modules_6_modules_q_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_6_modules_q_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_6_modules_q_lin_parameters_bias_
        ) = None
        view_30 = linear_36.view(1, -1, 12, 64)
        linear_36 = None
        q_12 = view_30.transpose(1, 2)
        view_30 = None
        k_12 = torch._C._nn.linear(
            tensor_34,
            l_self_modules_attentions_modules_6_modules_k_lin_parameters_weight_,
            l_self_modules_attentions_modules_6_modules_k_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_6_modules_k_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_6_modules_k_lin_parameters_bias_
        ) = None
        v_12 = torch._C._nn.linear(
            tensor_34,
            l_self_modules_attentions_modules_6_modules_v_lin_parameters_weight_,
            l_self_modules_attentions_modules_6_modules_v_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_6_modules_v_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_6_modules_v_lin_parameters_bias_
        ) = None
        view_31 = k_12.view(1, -1, 12, 64)
        k_12 = None
        k_13 = view_31.transpose(1, 2)
        view_31 = None
        view_32 = v_12.view(1, -1, 12, 64)
        v_12 = None
        v_13 = view_32.transpose(1, 2)
        view_32 = None
        q_13 = q_12 / 8.0
        q_12 = None
        transpose_33 = k_13.transpose(2, 3)
        scores_6 = torch.matmul(q_13, transpose_33)
        q_13 = transpose_33 = None
        eq_6 = attn_mask.__eq__(0)
        view_33 = eq_6.view((1, 1, 1, -1))
        eq_6 = None
        mask_7 = view_33.expand_as(scores_6)
        view_33 = None
        masked_fill__6 = scores_6.masked_fill_(mask_7, -3.4028234663852886e38)
        mask_7 = masked_fill__6 = None
        float_7 = scores_6.float()
        softmax_6 = torch.nn.functional.softmax(float_7, dim=-1)
        float_7 = None
        weights_12 = softmax_6.type_as(scores_6)
        softmax_6 = scores_6 = None
        weights_13 = torch.nn.functional.dropout(weights_12, p=0.1, training=False)
        weights_12 = None
        context_12 = torch.matmul(weights_13, v_13)
        weights_13 = None
        transpose_34 = context_12.transpose(1, 2)
        context_12 = None
        contiguous_6 = transpose_34.contiguous()
        transpose_34 = None
        context_13 = contiguous_6.view(1, -1, 768)
        contiguous_6 = None
        attn_12 = torch._C._nn.linear(
            context_13,
            l_self_modules_attentions_modules_6_modules_out_lin_parameters_weight_,
            l_self_modules_attentions_modules_6_modules_out_lin_parameters_bias_,
        )
        context_13 = (
            l_self_modules_attentions_modules_6_modules_out_lin_parameters_weight_
        ) = l_self_modules_attentions_modules_6_modules_out_lin_parameters_bias_ = None
        attn_13 = torch.nn.functional.dropout(attn_12, p=0.1, training=False)
        attn_12 = None
        tensor_35 = tensor_34 + attn_13
        tensor_34 = attn_13 = None
        tensor_36 = torch.nn.functional.layer_norm(
            tensor_35,
            (768,),
            l_self_modules_layer_norm1_modules_6_parameters_weight_,
            l_self_modules_layer_norm1_modules_6_parameters_bias_,
            1e-06,
        )
        tensor_35 = (
            l_self_modules_layer_norm1_modules_6_parameters_weight_
        ) = l_self_modules_layer_norm1_modules_6_parameters_bias_ = None
        x_24 = torch._C._nn.linear(
            tensor_36,
            l_self_modules_ffns_modules_6_modules_lin1_parameters_weight_,
            l_self_modules_ffns_modules_6_modules_lin1_parameters_bias_,
        )
        l_self_modules_ffns_modules_6_modules_lin1_parameters_weight_ = (
            l_self_modules_ffns_modules_6_modules_lin1_parameters_bias_
        ) = None
        x_25 = torch._C._nn.gelu(x_24)
        x_24 = None
        x_26 = torch._C._nn.linear(
            x_25,
            l_self_modules_ffns_modules_6_modules_lin2_parameters_weight_,
            l_self_modules_ffns_modules_6_modules_lin2_parameters_bias_,
        )
        x_25 = (
            l_self_modules_ffns_modules_6_modules_lin2_parameters_weight_
        ) = l_self_modules_ffns_modules_6_modules_lin2_parameters_bias_ = None
        x_27 = torch.nn.functional.dropout(x_26, p=0.1, training=False)
        x_26 = None
        tensor_37 = tensor_36 + x_27
        tensor_36 = x_27 = None
        tensor_38 = torch.nn.functional.layer_norm(
            tensor_37,
            (768,),
            l_self_modules_layer_norm2_modules_6_parameters_weight_,
            l_self_modules_layer_norm2_modules_6_parameters_bias_,
            1e-06,
        )
        tensor_37 = (
            l_self_modules_layer_norm2_modules_6_parameters_weight_
        ) = l_self_modules_layer_norm2_modules_6_parameters_bias_ = None
        unsqueeze_7 = mask.unsqueeze(-1)
        to_7 = unsqueeze_7.to(torch.float32)
        unsqueeze_7 = None
        tensor_38 *= to_7
        tensor_39 = tensor_38
        tensor_38 = to_7 = None
        linear_42 = torch._C._nn.linear(
            tensor_39,
            l_self_modules_attentions_modules_7_modules_q_lin_parameters_weight_,
            l_self_modules_attentions_modules_7_modules_q_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_7_modules_q_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_7_modules_q_lin_parameters_bias_
        ) = None
        view_35 = linear_42.view(1, -1, 12, 64)
        linear_42 = None
        q_14 = view_35.transpose(1, 2)
        view_35 = None
        k_14 = torch._C._nn.linear(
            tensor_39,
            l_self_modules_attentions_modules_7_modules_k_lin_parameters_weight_,
            l_self_modules_attentions_modules_7_modules_k_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_7_modules_k_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_7_modules_k_lin_parameters_bias_
        ) = None
        v_14 = torch._C._nn.linear(
            tensor_39,
            l_self_modules_attentions_modules_7_modules_v_lin_parameters_weight_,
            l_self_modules_attentions_modules_7_modules_v_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_7_modules_v_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_7_modules_v_lin_parameters_bias_
        ) = None
        view_36 = k_14.view(1, -1, 12, 64)
        k_14 = None
        k_15 = view_36.transpose(1, 2)
        view_36 = None
        view_37 = v_14.view(1, -1, 12, 64)
        v_14 = None
        v_15 = view_37.transpose(1, 2)
        view_37 = None
        q_15 = q_14 / 8.0
        q_14 = None
        transpose_38 = k_15.transpose(2, 3)
        scores_7 = torch.matmul(q_15, transpose_38)
        q_15 = transpose_38 = None
        eq_7 = attn_mask.__eq__(0)
        view_38 = eq_7.view((1, 1, 1, -1))
        eq_7 = None
        mask_8 = view_38.expand_as(scores_7)
        view_38 = None
        masked_fill__7 = scores_7.masked_fill_(mask_8, -3.4028234663852886e38)
        mask_8 = masked_fill__7 = None
        float_8 = scores_7.float()
        softmax_7 = torch.nn.functional.softmax(float_8, dim=-1)
        float_8 = None
        weights_14 = softmax_7.type_as(scores_7)
        softmax_7 = scores_7 = None
        weights_15 = torch.nn.functional.dropout(weights_14, p=0.1, training=False)
        weights_14 = None
        context_14 = torch.matmul(weights_15, v_15)
        weights_15 = None
        transpose_39 = context_14.transpose(1, 2)
        context_14 = None
        contiguous_7 = transpose_39.contiguous()
        transpose_39 = None
        context_15 = contiguous_7.view(1, -1, 768)
        contiguous_7 = None
        attn_14 = torch._C._nn.linear(
            context_15,
            l_self_modules_attentions_modules_7_modules_out_lin_parameters_weight_,
            l_self_modules_attentions_modules_7_modules_out_lin_parameters_bias_,
        )
        context_15 = (
            l_self_modules_attentions_modules_7_modules_out_lin_parameters_weight_
        ) = l_self_modules_attentions_modules_7_modules_out_lin_parameters_bias_ = None
        attn_15 = torch.nn.functional.dropout(attn_14, p=0.1, training=False)
        attn_14 = None
        tensor_40 = tensor_39 + attn_15
        tensor_39 = attn_15 = None
        tensor_41 = torch.nn.functional.layer_norm(
            tensor_40,
            (768,),
            l_self_modules_layer_norm1_modules_7_parameters_weight_,
            l_self_modules_layer_norm1_modules_7_parameters_bias_,
            1e-06,
        )
        tensor_40 = (
            l_self_modules_layer_norm1_modules_7_parameters_weight_
        ) = l_self_modules_layer_norm1_modules_7_parameters_bias_ = None
        x_28 = torch._C._nn.linear(
            tensor_41,
            l_self_modules_ffns_modules_7_modules_lin1_parameters_weight_,
            l_self_modules_ffns_modules_7_modules_lin1_parameters_bias_,
        )
        l_self_modules_ffns_modules_7_modules_lin1_parameters_weight_ = (
            l_self_modules_ffns_modules_7_modules_lin1_parameters_bias_
        ) = None
        x_29 = torch._C._nn.gelu(x_28)
        x_28 = None
        x_30 = torch._C._nn.linear(
            x_29,
            l_self_modules_ffns_modules_7_modules_lin2_parameters_weight_,
            l_self_modules_ffns_modules_7_modules_lin2_parameters_bias_,
        )
        x_29 = (
            l_self_modules_ffns_modules_7_modules_lin2_parameters_weight_
        ) = l_self_modules_ffns_modules_7_modules_lin2_parameters_bias_ = None
        x_31 = torch.nn.functional.dropout(x_30, p=0.1, training=False)
        x_30 = None
        tensor_42 = tensor_41 + x_31
        tensor_41 = x_31 = None
        tensor_43 = torch.nn.functional.layer_norm(
            tensor_42,
            (768,),
            l_self_modules_layer_norm2_modules_7_parameters_weight_,
            l_self_modules_layer_norm2_modules_7_parameters_bias_,
            1e-06,
        )
        tensor_42 = (
            l_self_modules_layer_norm2_modules_7_parameters_weight_
        ) = l_self_modules_layer_norm2_modules_7_parameters_bias_ = None
        unsqueeze_8 = mask.unsqueeze(-1)
        to_8 = unsqueeze_8.to(torch.float32)
        unsqueeze_8 = None
        tensor_43 *= to_8
        tensor_44 = tensor_43
        tensor_43 = to_8 = None
        linear_48 = torch._C._nn.linear(
            tensor_44,
            l_self_modules_attentions_modules_8_modules_q_lin_parameters_weight_,
            l_self_modules_attentions_modules_8_modules_q_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_8_modules_q_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_8_modules_q_lin_parameters_bias_
        ) = None
        view_40 = linear_48.view(1, -1, 12, 64)
        linear_48 = None
        q_16 = view_40.transpose(1, 2)
        view_40 = None
        k_16 = torch._C._nn.linear(
            tensor_44,
            l_self_modules_attentions_modules_8_modules_k_lin_parameters_weight_,
            l_self_modules_attentions_modules_8_modules_k_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_8_modules_k_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_8_modules_k_lin_parameters_bias_
        ) = None
        v_16 = torch._C._nn.linear(
            tensor_44,
            l_self_modules_attentions_modules_8_modules_v_lin_parameters_weight_,
            l_self_modules_attentions_modules_8_modules_v_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_8_modules_v_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_8_modules_v_lin_parameters_bias_
        ) = None
        view_41 = k_16.view(1, -1, 12, 64)
        k_16 = None
        k_17 = view_41.transpose(1, 2)
        view_41 = None
        view_42 = v_16.view(1, -1, 12, 64)
        v_16 = None
        v_17 = view_42.transpose(1, 2)
        view_42 = None
        q_17 = q_16 / 8.0
        q_16 = None
        transpose_43 = k_17.transpose(2, 3)
        scores_8 = torch.matmul(q_17, transpose_43)
        q_17 = transpose_43 = None
        eq_8 = attn_mask.__eq__(0)
        view_43 = eq_8.view((1, 1, 1, -1))
        eq_8 = None
        mask_9 = view_43.expand_as(scores_8)
        view_43 = None
        masked_fill__8 = scores_8.masked_fill_(mask_9, -3.4028234663852886e38)
        mask_9 = masked_fill__8 = None
        float_9 = scores_8.float()
        softmax_8 = torch.nn.functional.softmax(float_9, dim=-1)
        float_9 = None
        weights_16 = softmax_8.type_as(scores_8)
        softmax_8 = scores_8 = None
        weights_17 = torch.nn.functional.dropout(weights_16, p=0.1, training=False)
        weights_16 = None
        context_16 = torch.matmul(weights_17, v_17)
        weights_17 = None
        transpose_44 = context_16.transpose(1, 2)
        context_16 = None
        contiguous_8 = transpose_44.contiguous()
        transpose_44 = None
        context_17 = contiguous_8.view(1, -1, 768)
        contiguous_8 = None
        attn_16 = torch._C._nn.linear(
            context_17,
            l_self_modules_attentions_modules_8_modules_out_lin_parameters_weight_,
            l_self_modules_attentions_modules_8_modules_out_lin_parameters_bias_,
        )
        context_17 = (
            l_self_modules_attentions_modules_8_modules_out_lin_parameters_weight_
        ) = l_self_modules_attentions_modules_8_modules_out_lin_parameters_bias_ = None
        attn_17 = torch.nn.functional.dropout(attn_16, p=0.1, training=False)
        attn_16 = None
        tensor_45 = tensor_44 + attn_17
        tensor_44 = attn_17 = None
        tensor_46 = torch.nn.functional.layer_norm(
            tensor_45,
            (768,),
            l_self_modules_layer_norm1_modules_8_parameters_weight_,
            l_self_modules_layer_norm1_modules_8_parameters_bias_,
            1e-06,
        )
        tensor_45 = (
            l_self_modules_layer_norm1_modules_8_parameters_weight_
        ) = l_self_modules_layer_norm1_modules_8_parameters_bias_ = None
        x_32 = torch._C._nn.linear(
            tensor_46,
            l_self_modules_ffns_modules_8_modules_lin1_parameters_weight_,
            l_self_modules_ffns_modules_8_modules_lin1_parameters_bias_,
        )
        l_self_modules_ffns_modules_8_modules_lin1_parameters_weight_ = (
            l_self_modules_ffns_modules_8_modules_lin1_parameters_bias_
        ) = None
        x_33 = torch._C._nn.gelu(x_32)
        x_32 = None
        x_34 = torch._C._nn.linear(
            x_33,
            l_self_modules_ffns_modules_8_modules_lin2_parameters_weight_,
            l_self_modules_ffns_modules_8_modules_lin2_parameters_bias_,
        )
        x_33 = (
            l_self_modules_ffns_modules_8_modules_lin2_parameters_weight_
        ) = l_self_modules_ffns_modules_8_modules_lin2_parameters_bias_ = None
        x_35 = torch.nn.functional.dropout(x_34, p=0.1, training=False)
        x_34 = None
        tensor_47 = tensor_46 + x_35
        tensor_46 = x_35 = None
        tensor_48 = torch.nn.functional.layer_norm(
            tensor_47,
            (768,),
            l_self_modules_layer_norm2_modules_8_parameters_weight_,
            l_self_modules_layer_norm2_modules_8_parameters_bias_,
            1e-06,
        )
        tensor_47 = (
            l_self_modules_layer_norm2_modules_8_parameters_weight_
        ) = l_self_modules_layer_norm2_modules_8_parameters_bias_ = None
        unsqueeze_9 = mask.unsqueeze(-1)
        to_9 = unsqueeze_9.to(torch.float32)
        unsqueeze_9 = None
        tensor_48 *= to_9
        tensor_49 = tensor_48
        tensor_48 = to_9 = None
        linear_54 = torch._C._nn.linear(
            tensor_49,
            l_self_modules_attentions_modules_9_modules_q_lin_parameters_weight_,
            l_self_modules_attentions_modules_9_modules_q_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_9_modules_q_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_9_modules_q_lin_parameters_bias_
        ) = None
        view_45 = linear_54.view(1, -1, 12, 64)
        linear_54 = None
        q_18 = view_45.transpose(1, 2)
        view_45 = None
        k_18 = torch._C._nn.linear(
            tensor_49,
            l_self_modules_attentions_modules_9_modules_k_lin_parameters_weight_,
            l_self_modules_attentions_modules_9_modules_k_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_9_modules_k_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_9_modules_k_lin_parameters_bias_
        ) = None
        v_18 = torch._C._nn.linear(
            tensor_49,
            l_self_modules_attentions_modules_9_modules_v_lin_parameters_weight_,
            l_self_modules_attentions_modules_9_modules_v_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_9_modules_v_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_9_modules_v_lin_parameters_bias_
        ) = None
        view_46 = k_18.view(1, -1, 12, 64)
        k_18 = None
        k_19 = view_46.transpose(1, 2)
        view_46 = None
        view_47 = v_18.view(1, -1, 12, 64)
        v_18 = None
        v_19 = view_47.transpose(1, 2)
        view_47 = None
        q_19 = q_18 / 8.0
        q_18 = None
        transpose_48 = k_19.transpose(2, 3)
        scores_9 = torch.matmul(q_19, transpose_48)
        q_19 = transpose_48 = None
        eq_9 = attn_mask.__eq__(0)
        view_48 = eq_9.view((1, 1, 1, -1))
        eq_9 = None
        mask_10 = view_48.expand_as(scores_9)
        view_48 = None
        masked_fill__9 = scores_9.masked_fill_(mask_10, -3.4028234663852886e38)
        mask_10 = masked_fill__9 = None
        float_10 = scores_9.float()
        softmax_9 = torch.nn.functional.softmax(float_10, dim=-1)
        float_10 = None
        weights_18 = softmax_9.type_as(scores_9)
        softmax_9 = scores_9 = None
        weights_19 = torch.nn.functional.dropout(weights_18, p=0.1, training=False)
        weights_18 = None
        context_18 = torch.matmul(weights_19, v_19)
        weights_19 = None
        transpose_49 = context_18.transpose(1, 2)
        context_18 = None
        contiguous_9 = transpose_49.contiguous()
        transpose_49 = None
        context_19 = contiguous_9.view(1, -1, 768)
        contiguous_9 = None
        attn_18 = torch._C._nn.linear(
            context_19,
            l_self_modules_attentions_modules_9_modules_out_lin_parameters_weight_,
            l_self_modules_attentions_modules_9_modules_out_lin_parameters_bias_,
        )
        context_19 = (
            l_self_modules_attentions_modules_9_modules_out_lin_parameters_weight_
        ) = l_self_modules_attentions_modules_9_modules_out_lin_parameters_bias_ = None
        attn_19 = torch.nn.functional.dropout(attn_18, p=0.1, training=False)
        attn_18 = None
        tensor_50 = tensor_49 + attn_19
        tensor_49 = attn_19 = None
        tensor_51 = torch.nn.functional.layer_norm(
            tensor_50,
            (768,),
            l_self_modules_layer_norm1_modules_9_parameters_weight_,
            l_self_modules_layer_norm1_modules_9_parameters_bias_,
            1e-06,
        )
        tensor_50 = (
            l_self_modules_layer_norm1_modules_9_parameters_weight_
        ) = l_self_modules_layer_norm1_modules_9_parameters_bias_ = None
        x_36 = torch._C._nn.linear(
            tensor_51,
            l_self_modules_ffns_modules_9_modules_lin1_parameters_weight_,
            l_self_modules_ffns_modules_9_modules_lin1_parameters_bias_,
        )
        l_self_modules_ffns_modules_9_modules_lin1_parameters_weight_ = (
            l_self_modules_ffns_modules_9_modules_lin1_parameters_bias_
        ) = None
        x_37 = torch._C._nn.gelu(x_36)
        x_36 = None
        x_38 = torch._C._nn.linear(
            x_37,
            l_self_modules_ffns_modules_9_modules_lin2_parameters_weight_,
            l_self_modules_ffns_modules_9_modules_lin2_parameters_bias_,
        )
        x_37 = (
            l_self_modules_ffns_modules_9_modules_lin2_parameters_weight_
        ) = l_self_modules_ffns_modules_9_modules_lin2_parameters_bias_ = None
        x_39 = torch.nn.functional.dropout(x_38, p=0.1, training=False)
        x_38 = None
        tensor_52 = tensor_51 + x_39
        tensor_51 = x_39 = None
        tensor_53 = torch.nn.functional.layer_norm(
            tensor_52,
            (768,),
            l_self_modules_layer_norm2_modules_9_parameters_weight_,
            l_self_modules_layer_norm2_modules_9_parameters_bias_,
            1e-06,
        )
        tensor_52 = (
            l_self_modules_layer_norm2_modules_9_parameters_weight_
        ) = l_self_modules_layer_norm2_modules_9_parameters_bias_ = None
        unsqueeze_10 = mask.unsqueeze(-1)
        to_10 = unsqueeze_10.to(torch.float32)
        unsqueeze_10 = None
        tensor_53 *= to_10
        tensor_54 = tensor_53
        tensor_53 = to_10 = None
        linear_60 = torch._C._nn.linear(
            tensor_54,
            l_self_modules_attentions_modules_10_modules_q_lin_parameters_weight_,
            l_self_modules_attentions_modules_10_modules_q_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_10_modules_q_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_10_modules_q_lin_parameters_bias_
        ) = None
        view_50 = linear_60.view(1, -1, 12, 64)
        linear_60 = None
        q_20 = view_50.transpose(1, 2)
        view_50 = None
        k_20 = torch._C._nn.linear(
            tensor_54,
            l_self_modules_attentions_modules_10_modules_k_lin_parameters_weight_,
            l_self_modules_attentions_modules_10_modules_k_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_10_modules_k_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_10_modules_k_lin_parameters_bias_
        ) = None
        v_20 = torch._C._nn.linear(
            tensor_54,
            l_self_modules_attentions_modules_10_modules_v_lin_parameters_weight_,
            l_self_modules_attentions_modules_10_modules_v_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_10_modules_v_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_10_modules_v_lin_parameters_bias_
        ) = None
        view_51 = k_20.view(1, -1, 12, 64)
        k_20 = None
        k_21 = view_51.transpose(1, 2)
        view_51 = None
        view_52 = v_20.view(1, -1, 12, 64)
        v_20 = None
        v_21 = view_52.transpose(1, 2)
        view_52 = None
        q_21 = q_20 / 8.0
        q_20 = None
        transpose_53 = k_21.transpose(2, 3)
        scores_10 = torch.matmul(q_21, transpose_53)
        q_21 = transpose_53 = None
        eq_10 = attn_mask.__eq__(0)
        view_53 = eq_10.view((1, 1, 1, -1))
        eq_10 = None
        mask_11 = view_53.expand_as(scores_10)
        view_53 = None
        masked_fill__10 = scores_10.masked_fill_(mask_11, -3.4028234663852886e38)
        mask_11 = masked_fill__10 = None
        float_11 = scores_10.float()
        softmax_10 = torch.nn.functional.softmax(float_11, dim=-1)
        float_11 = None
        weights_20 = softmax_10.type_as(scores_10)
        softmax_10 = scores_10 = None
        weights_21 = torch.nn.functional.dropout(weights_20, p=0.1, training=False)
        weights_20 = None
        context_20 = torch.matmul(weights_21, v_21)
        weights_21 = None
        transpose_54 = context_20.transpose(1, 2)
        context_20 = None
        contiguous_10 = transpose_54.contiguous()
        transpose_54 = None
        context_21 = contiguous_10.view(1, -1, 768)
        contiguous_10 = None
        attn_20 = torch._C._nn.linear(
            context_21,
            l_self_modules_attentions_modules_10_modules_out_lin_parameters_weight_,
            l_self_modules_attentions_modules_10_modules_out_lin_parameters_bias_,
        )
        context_21 = (
            l_self_modules_attentions_modules_10_modules_out_lin_parameters_weight_
        ) = l_self_modules_attentions_modules_10_modules_out_lin_parameters_bias_ = None
        attn_21 = torch.nn.functional.dropout(attn_20, p=0.1, training=False)
        attn_20 = None
        tensor_55 = tensor_54 + attn_21
        tensor_54 = attn_21 = None
        tensor_56 = torch.nn.functional.layer_norm(
            tensor_55,
            (768,),
            l_self_modules_layer_norm1_modules_10_parameters_weight_,
            l_self_modules_layer_norm1_modules_10_parameters_bias_,
            1e-06,
        )
        tensor_55 = (
            l_self_modules_layer_norm1_modules_10_parameters_weight_
        ) = l_self_modules_layer_norm1_modules_10_parameters_bias_ = None
        x_40 = torch._C._nn.linear(
            tensor_56,
            l_self_modules_ffns_modules_10_modules_lin1_parameters_weight_,
            l_self_modules_ffns_modules_10_modules_lin1_parameters_bias_,
        )
        l_self_modules_ffns_modules_10_modules_lin1_parameters_weight_ = (
            l_self_modules_ffns_modules_10_modules_lin1_parameters_bias_
        ) = None
        x_41 = torch._C._nn.gelu(x_40)
        x_40 = None
        x_42 = torch._C._nn.linear(
            x_41,
            l_self_modules_ffns_modules_10_modules_lin2_parameters_weight_,
            l_self_modules_ffns_modules_10_modules_lin2_parameters_bias_,
        )
        x_41 = (
            l_self_modules_ffns_modules_10_modules_lin2_parameters_weight_
        ) = l_self_modules_ffns_modules_10_modules_lin2_parameters_bias_ = None
        x_43 = torch.nn.functional.dropout(x_42, p=0.1, training=False)
        x_42 = None
        tensor_57 = tensor_56 + x_43
        tensor_56 = x_43 = None
        tensor_58 = torch.nn.functional.layer_norm(
            tensor_57,
            (768,),
            l_self_modules_layer_norm2_modules_10_parameters_weight_,
            l_self_modules_layer_norm2_modules_10_parameters_bias_,
            1e-06,
        )
        tensor_57 = (
            l_self_modules_layer_norm2_modules_10_parameters_weight_
        ) = l_self_modules_layer_norm2_modules_10_parameters_bias_ = None
        unsqueeze_11 = mask.unsqueeze(-1)
        to_11 = unsqueeze_11.to(torch.float32)
        unsqueeze_11 = None
        tensor_58 *= to_11
        tensor_59 = tensor_58
        tensor_58 = to_11 = None
        linear_66 = torch._C._nn.linear(
            tensor_59,
            l_self_modules_attentions_modules_11_modules_q_lin_parameters_weight_,
            l_self_modules_attentions_modules_11_modules_q_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_11_modules_q_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_11_modules_q_lin_parameters_bias_
        ) = None
        view_55 = linear_66.view(1, -1, 12, 64)
        linear_66 = None
        q_22 = view_55.transpose(1, 2)
        view_55 = None
        k_22 = torch._C._nn.linear(
            tensor_59,
            l_self_modules_attentions_modules_11_modules_k_lin_parameters_weight_,
            l_self_modules_attentions_modules_11_modules_k_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_11_modules_k_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_11_modules_k_lin_parameters_bias_
        ) = None
        v_22 = torch._C._nn.linear(
            tensor_59,
            l_self_modules_attentions_modules_11_modules_v_lin_parameters_weight_,
            l_self_modules_attentions_modules_11_modules_v_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_11_modules_v_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_11_modules_v_lin_parameters_bias_
        ) = None
        view_56 = k_22.view(1, -1, 12, 64)
        k_22 = None
        k_23 = view_56.transpose(1, 2)
        view_56 = None
        view_57 = v_22.view(1, -1, 12, 64)
        v_22 = None
        v_23 = view_57.transpose(1, 2)
        view_57 = None
        q_23 = q_22 / 8.0
        q_22 = None
        transpose_58 = k_23.transpose(2, 3)
        scores_11 = torch.matmul(q_23, transpose_58)
        q_23 = transpose_58 = None
        eq_11 = attn_mask.__eq__(0)
        attn_mask = None
        view_58 = eq_11.view((1, 1, 1, -1))
        eq_11 = None
        mask_12 = view_58.expand_as(scores_11)
        view_58 = None
        masked_fill__11 = scores_11.masked_fill_(mask_12, -3.4028234663852886e38)
        mask_12 = masked_fill__11 = None
        float_12 = scores_11.float()
        softmax_11 = torch.nn.functional.softmax(float_12, dim=-1)
        float_12 = None
        weights_22 = softmax_11.type_as(scores_11)
        softmax_11 = scores_11 = None
        weights_23 = torch.nn.functional.dropout(weights_22, p=0.1, training=False)
        weights_22 = None
        context_22 = torch.matmul(weights_23, v_23)
        weights_23 = None
        transpose_59 = context_22.transpose(1, 2)
        context_22 = None
        contiguous_11 = transpose_59.contiguous()
        transpose_59 = None
        context_23 = contiguous_11.view(1, -1, 768)
        contiguous_11 = None
        attn_22 = torch._C._nn.linear(
            context_23,
            l_self_modules_attentions_modules_11_modules_out_lin_parameters_weight_,
            l_self_modules_attentions_modules_11_modules_out_lin_parameters_bias_,
        )
        context_23 = (
            l_self_modules_attentions_modules_11_modules_out_lin_parameters_weight_
        ) = l_self_modules_attentions_modules_11_modules_out_lin_parameters_bias_ = None
        attn_23 = torch.nn.functional.dropout(attn_22, p=0.1, training=False)
        attn_22 = None
        tensor_60 = tensor_59 + attn_23
        tensor_59 = attn_23 = None
        tensor_61 = torch.nn.functional.layer_norm(
            tensor_60,
            (768,),
            l_self_modules_layer_norm1_modules_11_parameters_weight_,
            l_self_modules_layer_norm1_modules_11_parameters_bias_,
            1e-06,
        )
        tensor_60 = (
            l_self_modules_layer_norm1_modules_11_parameters_weight_
        ) = l_self_modules_layer_norm1_modules_11_parameters_bias_ = None
        x_44 = torch._C._nn.linear(
            tensor_61,
            l_self_modules_ffns_modules_11_modules_lin1_parameters_weight_,
            l_self_modules_ffns_modules_11_modules_lin1_parameters_bias_,
        )
        l_self_modules_ffns_modules_11_modules_lin1_parameters_weight_ = (
            l_self_modules_ffns_modules_11_modules_lin1_parameters_bias_
        ) = None
        x_45 = torch._C._nn.gelu(x_44)
        x_44 = None
        x_46 = torch._C._nn.linear(
            x_45,
            l_self_modules_ffns_modules_11_modules_lin2_parameters_weight_,
            l_self_modules_ffns_modules_11_modules_lin2_parameters_bias_,
        )
        x_45 = (
            l_self_modules_ffns_modules_11_modules_lin2_parameters_weight_
        ) = l_self_modules_ffns_modules_11_modules_lin2_parameters_bias_ = None
        x_47 = torch.nn.functional.dropout(x_46, p=0.1, training=False)
        x_46 = None
        tensor_62 = tensor_61 + x_47
        tensor_61 = x_47 = None
        tensor_63 = torch.nn.functional.layer_norm(
            tensor_62,
            (768,),
            l_self_modules_layer_norm2_modules_11_parameters_weight_,
            l_self_modules_layer_norm2_modules_11_parameters_bias_,
            1e-06,
        )
        tensor_62 = (
            l_self_modules_layer_norm2_modules_11_parameters_weight_
        ) = l_self_modules_layer_norm2_modules_11_parameters_bias_ = None
        unsqueeze_12 = mask.unsqueeze(-1)
        mask = None
        to_12 = unsqueeze_12.to(torch.float32)
        unsqueeze_12 = None
        tensor_63 *= to_12
        tensor_64 = tensor_63
        tensor_63 = to_12 = None
        return (
            v_1,
            k_1,
            v_3,
            k_3,
            v_5,
            k_5,
            v_7,
            k_7,
            v_9,
            k_9,
            v_11,
            k_11,
            v_13,
            k_13,
            v_15,
            k_15,
            v_17,
            k_17,
            v_19,
            k_19,
            v_21,
            k_21,
            v_23,
            k_23,
            tensor_64,
        )
