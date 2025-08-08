import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_self_modules_embeddings_modules_tok_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_attention_mask_: torch.Tensor,
        L_self_modules_global_rotary_emb_buffers_inv_freq_: torch.Tensor,
        L_self_modules_local_rotary_emb_buffers_inv_freq_: torch.Tensor,
        L_self_modules_layers_modules_0_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_attn_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mlp_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mlp_modules_Wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mlp_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_attn_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_attn_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mlp_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mlp_modules_Wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mlp_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_attn_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_attn_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mlp_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mlp_modules_Wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mlp_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_attn_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_attn_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mlp_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mlp_modules_Wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mlp_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_attn_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_attn_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mlp_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mlp_modules_Wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mlp_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_attn_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_attn_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_mlp_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_mlp_modules_Wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_mlp_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_attn_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_attn_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_mlp_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_mlp_modules_Wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_mlp_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_attn_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_attn_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_mlp_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_mlp_modules_Wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_mlp_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_attn_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_attn_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_mlp_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_mlp_modules_Wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_mlp_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_attn_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_attn_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_mlp_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_mlp_modules_Wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_mlp_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_attn_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_attn_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_mlp_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_mlp_modules_Wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_mlp_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_attn_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_attn_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_mlp_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_mlp_modules_Wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_mlp_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_attn_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_attn_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_mlp_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_mlp_modules_Wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_mlp_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_attn_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_attn_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_mlp_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_mlp_modules_Wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_mlp_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_attn_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_attn_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_mlp_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_mlp_modules_Wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_mlp_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_attn_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_attn_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_mlp_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_mlp_modules_Wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_mlp_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_attn_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_attn_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_mlp_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_mlp_modules_Wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_mlp_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_attn_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_attn_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_mlp_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_mlp_modules_Wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_mlp_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_attn_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_attn_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_mlp_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_mlp_modules_Wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_mlp_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_attn_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_attn_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_mlp_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_mlp_modules_Wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_mlp_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_attn_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_attn_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_mlp_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_mlp_modules_Wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_mlp_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_attn_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_attn_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_mlp_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_mlp_modules_Wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_mlp_modules_Wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_final_norm_parameters_weight_: torch.nn.parameter.Parameter,
    ):
        l_input_ids_ = L_input_ids_
        l_self_modules_embeddings_modules_tok_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_tok_embeddings_parameters_weight_
        )
        l_self_modules_embeddings_modules_norm_parameters_weight_ = (
            L_self_modules_embeddings_modules_norm_parameters_weight_
        )
        l_attention_mask_ = L_attention_mask_
        l_self_modules_global_rotary_emb_buffers_inv_freq_ = (
            L_self_modules_global_rotary_emb_buffers_inv_freq_
        )
        l_self_modules_local_rotary_emb_buffers_inv_freq_ = (
            L_self_modules_local_rotary_emb_buffers_inv_freq_
        )
        l_self_modules_layers_modules_0_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_attn_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_0_modules_attn_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_0_modules_mlp_norm_parameters_weight_ = (
            L_self_modules_layers_modules_0_modules_mlp_norm_parameters_weight_
        )
        l_self_modules_layers_modules_0_modules_mlp_modules_wi_parameters_weight_ = (
            L_self_modules_layers_modules_0_modules_mlp_modules_Wi_parameters_weight_
        )
        l_self_modules_layers_modules_0_modules_mlp_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_0_modules_mlp_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_1_modules_attn_norm_parameters_weight_ = (
            L_self_modules_layers_modules_1_modules_attn_norm_parameters_weight_
        )
        l_self_modules_layers_modules_1_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_attn_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_1_modules_attn_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_1_modules_mlp_norm_parameters_weight_ = (
            L_self_modules_layers_modules_1_modules_mlp_norm_parameters_weight_
        )
        l_self_modules_layers_modules_1_modules_mlp_modules_wi_parameters_weight_ = (
            L_self_modules_layers_modules_1_modules_mlp_modules_Wi_parameters_weight_
        )
        l_self_modules_layers_modules_1_modules_mlp_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_1_modules_mlp_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_2_modules_attn_norm_parameters_weight_ = (
            L_self_modules_layers_modules_2_modules_attn_norm_parameters_weight_
        )
        l_self_modules_layers_modules_2_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_attn_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_2_modules_attn_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_2_modules_mlp_norm_parameters_weight_ = (
            L_self_modules_layers_modules_2_modules_mlp_norm_parameters_weight_
        )
        l_self_modules_layers_modules_2_modules_mlp_modules_wi_parameters_weight_ = (
            L_self_modules_layers_modules_2_modules_mlp_modules_Wi_parameters_weight_
        )
        l_self_modules_layers_modules_2_modules_mlp_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_2_modules_mlp_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_3_modules_attn_norm_parameters_weight_ = (
            L_self_modules_layers_modules_3_modules_attn_norm_parameters_weight_
        )
        l_self_modules_layers_modules_3_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_attn_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_3_modules_attn_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_3_modules_mlp_norm_parameters_weight_ = (
            L_self_modules_layers_modules_3_modules_mlp_norm_parameters_weight_
        )
        l_self_modules_layers_modules_3_modules_mlp_modules_wi_parameters_weight_ = (
            L_self_modules_layers_modules_3_modules_mlp_modules_Wi_parameters_weight_
        )
        l_self_modules_layers_modules_3_modules_mlp_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_3_modules_mlp_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_4_modules_attn_norm_parameters_weight_ = (
            L_self_modules_layers_modules_4_modules_attn_norm_parameters_weight_
        )
        l_self_modules_layers_modules_4_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_attn_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_4_modules_attn_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_4_modules_mlp_norm_parameters_weight_ = (
            L_self_modules_layers_modules_4_modules_mlp_norm_parameters_weight_
        )
        l_self_modules_layers_modules_4_modules_mlp_modules_wi_parameters_weight_ = (
            L_self_modules_layers_modules_4_modules_mlp_modules_Wi_parameters_weight_
        )
        l_self_modules_layers_modules_4_modules_mlp_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_4_modules_mlp_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_5_modules_attn_norm_parameters_weight_ = (
            L_self_modules_layers_modules_5_modules_attn_norm_parameters_weight_
        )
        l_self_modules_layers_modules_5_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_attn_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_5_modules_attn_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_5_modules_mlp_norm_parameters_weight_ = (
            L_self_modules_layers_modules_5_modules_mlp_norm_parameters_weight_
        )
        l_self_modules_layers_modules_5_modules_mlp_modules_wi_parameters_weight_ = (
            L_self_modules_layers_modules_5_modules_mlp_modules_Wi_parameters_weight_
        )
        l_self_modules_layers_modules_5_modules_mlp_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_5_modules_mlp_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_6_modules_attn_norm_parameters_weight_ = (
            L_self_modules_layers_modules_6_modules_attn_norm_parameters_weight_
        )
        l_self_modules_layers_modules_6_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_attn_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_6_modules_attn_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_6_modules_mlp_norm_parameters_weight_ = (
            L_self_modules_layers_modules_6_modules_mlp_norm_parameters_weight_
        )
        l_self_modules_layers_modules_6_modules_mlp_modules_wi_parameters_weight_ = (
            L_self_modules_layers_modules_6_modules_mlp_modules_Wi_parameters_weight_
        )
        l_self_modules_layers_modules_6_modules_mlp_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_6_modules_mlp_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_7_modules_attn_norm_parameters_weight_ = (
            L_self_modules_layers_modules_7_modules_attn_norm_parameters_weight_
        )
        l_self_modules_layers_modules_7_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_attn_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_7_modules_attn_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_7_modules_mlp_norm_parameters_weight_ = (
            L_self_modules_layers_modules_7_modules_mlp_norm_parameters_weight_
        )
        l_self_modules_layers_modules_7_modules_mlp_modules_wi_parameters_weight_ = (
            L_self_modules_layers_modules_7_modules_mlp_modules_Wi_parameters_weight_
        )
        l_self_modules_layers_modules_7_modules_mlp_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_7_modules_mlp_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_8_modules_attn_norm_parameters_weight_ = (
            L_self_modules_layers_modules_8_modules_attn_norm_parameters_weight_
        )
        l_self_modules_layers_modules_8_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_attn_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_8_modules_attn_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_8_modules_mlp_norm_parameters_weight_ = (
            L_self_modules_layers_modules_8_modules_mlp_norm_parameters_weight_
        )
        l_self_modules_layers_modules_8_modules_mlp_modules_wi_parameters_weight_ = (
            L_self_modules_layers_modules_8_modules_mlp_modules_Wi_parameters_weight_
        )
        l_self_modules_layers_modules_8_modules_mlp_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_8_modules_mlp_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_9_modules_attn_norm_parameters_weight_ = (
            L_self_modules_layers_modules_9_modules_attn_norm_parameters_weight_
        )
        l_self_modules_layers_modules_9_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_attn_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_9_modules_attn_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_9_modules_mlp_norm_parameters_weight_ = (
            L_self_modules_layers_modules_9_modules_mlp_norm_parameters_weight_
        )
        l_self_modules_layers_modules_9_modules_mlp_modules_wi_parameters_weight_ = (
            L_self_modules_layers_modules_9_modules_mlp_modules_Wi_parameters_weight_
        )
        l_self_modules_layers_modules_9_modules_mlp_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_9_modules_mlp_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_10_modules_attn_norm_parameters_weight_ = (
            L_self_modules_layers_modules_10_modules_attn_norm_parameters_weight_
        )
        l_self_modules_layers_modules_10_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_attn_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_10_modules_attn_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_10_modules_mlp_norm_parameters_weight_ = (
            L_self_modules_layers_modules_10_modules_mlp_norm_parameters_weight_
        )
        l_self_modules_layers_modules_10_modules_mlp_modules_wi_parameters_weight_ = (
            L_self_modules_layers_modules_10_modules_mlp_modules_Wi_parameters_weight_
        )
        l_self_modules_layers_modules_10_modules_mlp_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_10_modules_mlp_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_11_modules_attn_norm_parameters_weight_ = (
            L_self_modules_layers_modules_11_modules_attn_norm_parameters_weight_
        )
        l_self_modules_layers_modules_11_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_attn_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_11_modules_attn_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_11_modules_mlp_norm_parameters_weight_ = (
            L_self_modules_layers_modules_11_modules_mlp_norm_parameters_weight_
        )
        l_self_modules_layers_modules_11_modules_mlp_modules_wi_parameters_weight_ = (
            L_self_modules_layers_modules_11_modules_mlp_modules_Wi_parameters_weight_
        )
        l_self_modules_layers_modules_11_modules_mlp_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_11_modules_mlp_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_12_modules_attn_norm_parameters_weight_ = (
            L_self_modules_layers_modules_12_modules_attn_norm_parameters_weight_
        )
        l_self_modules_layers_modules_12_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_attn_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_12_modules_attn_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_12_modules_mlp_norm_parameters_weight_ = (
            L_self_modules_layers_modules_12_modules_mlp_norm_parameters_weight_
        )
        l_self_modules_layers_modules_12_modules_mlp_modules_wi_parameters_weight_ = (
            L_self_modules_layers_modules_12_modules_mlp_modules_Wi_parameters_weight_
        )
        l_self_modules_layers_modules_12_modules_mlp_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_12_modules_mlp_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_13_modules_attn_norm_parameters_weight_ = (
            L_self_modules_layers_modules_13_modules_attn_norm_parameters_weight_
        )
        l_self_modules_layers_modules_13_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_attn_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_13_modules_attn_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_13_modules_mlp_norm_parameters_weight_ = (
            L_self_modules_layers_modules_13_modules_mlp_norm_parameters_weight_
        )
        l_self_modules_layers_modules_13_modules_mlp_modules_wi_parameters_weight_ = (
            L_self_modules_layers_modules_13_modules_mlp_modules_Wi_parameters_weight_
        )
        l_self_modules_layers_modules_13_modules_mlp_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_13_modules_mlp_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_14_modules_attn_norm_parameters_weight_ = (
            L_self_modules_layers_modules_14_modules_attn_norm_parameters_weight_
        )
        l_self_modules_layers_modules_14_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_attn_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_14_modules_attn_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_14_modules_mlp_norm_parameters_weight_ = (
            L_self_modules_layers_modules_14_modules_mlp_norm_parameters_weight_
        )
        l_self_modules_layers_modules_14_modules_mlp_modules_wi_parameters_weight_ = (
            L_self_modules_layers_modules_14_modules_mlp_modules_Wi_parameters_weight_
        )
        l_self_modules_layers_modules_14_modules_mlp_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_14_modules_mlp_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_15_modules_attn_norm_parameters_weight_ = (
            L_self_modules_layers_modules_15_modules_attn_norm_parameters_weight_
        )
        l_self_modules_layers_modules_15_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_attn_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_15_modules_attn_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_15_modules_mlp_norm_parameters_weight_ = (
            L_self_modules_layers_modules_15_modules_mlp_norm_parameters_weight_
        )
        l_self_modules_layers_modules_15_modules_mlp_modules_wi_parameters_weight_ = (
            L_self_modules_layers_modules_15_modules_mlp_modules_Wi_parameters_weight_
        )
        l_self_modules_layers_modules_15_modules_mlp_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_15_modules_mlp_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_16_modules_attn_norm_parameters_weight_ = (
            L_self_modules_layers_modules_16_modules_attn_norm_parameters_weight_
        )
        l_self_modules_layers_modules_16_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_attn_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_16_modules_attn_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_16_modules_mlp_norm_parameters_weight_ = (
            L_self_modules_layers_modules_16_modules_mlp_norm_parameters_weight_
        )
        l_self_modules_layers_modules_16_modules_mlp_modules_wi_parameters_weight_ = (
            L_self_modules_layers_modules_16_modules_mlp_modules_Wi_parameters_weight_
        )
        l_self_modules_layers_modules_16_modules_mlp_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_16_modules_mlp_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_17_modules_attn_norm_parameters_weight_ = (
            L_self_modules_layers_modules_17_modules_attn_norm_parameters_weight_
        )
        l_self_modules_layers_modules_17_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_attn_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_17_modules_attn_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_17_modules_mlp_norm_parameters_weight_ = (
            L_self_modules_layers_modules_17_modules_mlp_norm_parameters_weight_
        )
        l_self_modules_layers_modules_17_modules_mlp_modules_wi_parameters_weight_ = (
            L_self_modules_layers_modules_17_modules_mlp_modules_Wi_parameters_weight_
        )
        l_self_modules_layers_modules_17_modules_mlp_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_17_modules_mlp_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_18_modules_attn_norm_parameters_weight_ = (
            L_self_modules_layers_modules_18_modules_attn_norm_parameters_weight_
        )
        l_self_modules_layers_modules_18_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_18_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_18_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_18_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_18_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_18_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_18_modules_attn_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_18_modules_attn_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_18_modules_mlp_norm_parameters_weight_ = (
            L_self_modules_layers_modules_18_modules_mlp_norm_parameters_weight_
        )
        l_self_modules_layers_modules_18_modules_mlp_modules_wi_parameters_weight_ = (
            L_self_modules_layers_modules_18_modules_mlp_modules_Wi_parameters_weight_
        )
        l_self_modules_layers_modules_18_modules_mlp_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_18_modules_mlp_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_19_modules_attn_norm_parameters_weight_ = (
            L_self_modules_layers_modules_19_modules_attn_norm_parameters_weight_
        )
        l_self_modules_layers_modules_19_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_19_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_19_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_19_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_19_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_19_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_19_modules_attn_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_19_modules_attn_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_19_modules_mlp_norm_parameters_weight_ = (
            L_self_modules_layers_modules_19_modules_mlp_norm_parameters_weight_
        )
        l_self_modules_layers_modules_19_modules_mlp_modules_wi_parameters_weight_ = (
            L_self_modules_layers_modules_19_modules_mlp_modules_Wi_parameters_weight_
        )
        l_self_modules_layers_modules_19_modules_mlp_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_19_modules_mlp_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_20_modules_attn_norm_parameters_weight_ = (
            L_self_modules_layers_modules_20_modules_attn_norm_parameters_weight_
        )
        l_self_modules_layers_modules_20_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_20_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_20_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_20_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_20_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_20_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_20_modules_attn_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_20_modules_attn_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_20_modules_mlp_norm_parameters_weight_ = (
            L_self_modules_layers_modules_20_modules_mlp_norm_parameters_weight_
        )
        l_self_modules_layers_modules_20_modules_mlp_modules_wi_parameters_weight_ = (
            L_self_modules_layers_modules_20_modules_mlp_modules_Wi_parameters_weight_
        )
        l_self_modules_layers_modules_20_modules_mlp_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_20_modules_mlp_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_21_modules_attn_norm_parameters_weight_ = (
            L_self_modules_layers_modules_21_modules_attn_norm_parameters_weight_
        )
        l_self_modules_layers_modules_21_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_21_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_21_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_21_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_21_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_21_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_21_modules_attn_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_21_modules_attn_modules_Wo_parameters_weight_
        )
        l_self_modules_layers_modules_21_modules_mlp_norm_parameters_weight_ = (
            L_self_modules_layers_modules_21_modules_mlp_norm_parameters_weight_
        )
        l_self_modules_layers_modules_21_modules_mlp_modules_wi_parameters_weight_ = (
            L_self_modules_layers_modules_21_modules_mlp_modules_Wi_parameters_weight_
        )
        l_self_modules_layers_modules_21_modules_mlp_modules_wo_parameters_weight_ = (
            L_self_modules_layers_modules_21_modules_mlp_modules_Wo_parameters_weight_
        )
        l_self_modules_final_norm_parameters_weight_ = (
            L_self_modules_final_norm_parameters_weight_
        )
        cache_position = torch.arange(0, 10, device=device(type="cuda", index=0))
        unsqueeze = cache_position.unsqueeze(0)
        position_ids = unsqueeze.expand(1, -1)
        unsqueeze = None
        embedding = torch.nn.functional.embedding(
            l_input_ids_,
            l_self_modules_embeddings_modules_tok_embeddings_parameters_weight_,
            50283,
            None,
            2.0,
            False,
            False,
        )
        l_input_ids_ = (
            l_self_modules_embeddings_modules_tok_embeddings_parameters_weight_
        ) = None
        layer_norm = torch.nn.functional.layer_norm(
            embedding,
            (768,),
            l_self_modules_embeddings_modules_norm_parameters_weight_,
            None,
            1e-05,
        )
        embedding = l_self_modules_embeddings_modules_norm_parameters_weight_ = None
        hidden_states = torch.nn.functional.dropout(layer_norm, 0.0, False, False)
        layer_norm = None
        attention_mask = l_attention_mask_.to(
            device=device(type="cuda", index=0), dtype=torch.bool
        )
        kv_arange = torch.arange(10, device=device(type="cuda", index=0))
        kv_arange += 0
        kv_arange_1 = kv_arange
        kv_arange = None
        batch_arange = torch.arange(1, device=device(type="cuda", index=0))
        head_arange = torch.arange(1, device=device(type="cuda", index=0))
        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions = None
        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(
            1, "error"
        )
        _vmap_increment_nesting = None
        child = torch._C._functorch._add_batch_dim(batch_arange, 0, 1)
        batch_arange = None
        lazy_load_decompositions_1 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_1 = None
        _vmap_increment_nesting_1 = torch._C._functorch._vmap_increment_nesting(
            1, "error"
        )
        _vmap_increment_nesting_1 = None
        child_1 = torch._C._functorch._add_batch_dim(head_arange, 0, 2)
        head_arange = child_1 = None
        lazy_load_decompositions_2 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_2 = None
        _vmap_increment_nesting_2 = torch._C._functorch._vmap_increment_nesting(
            10, "error"
        )
        _vmap_increment_nesting_2 = None
        child_2 = torch._C._functorch._add_batch_dim(cache_position, 0, 3)
        lazy_load_decompositions_3 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_3 = None
        _vmap_increment_nesting_3 = torch._C._functorch._vmap_increment_nesting(
            10, "error"
        )
        _vmap_increment_nesting_3 = None
        child_3 = torch._C._functorch._add_batch_dim(kv_arange_1, 0, 4)
        kv_arange_1 = None
        result = child_2.new_ones((), dtype=torch.bool)
        le = child_3.le(child_2)
        child_2 = None
        result_1 = result.__and__(le)
        result = le = None
        index = torch.ops.aten.index(attention_mask, [child, child_3])
        attention_mask = child = child_3 = None
        result_2 = result_1.__and__(index)
        result_1 = index = None
        batched_outputs = torch._C._functorch._remove_batch_dim(result_2, 4, 10, 0)
        result_2 = None
        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting = None
        batched_outputs_1 = torch._C._functorch._remove_batch_dim(
            batched_outputs, 3, 10, 0
        )
        batched_outputs = None
        _vmap_decrement_nesting_1 = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting_1 = None
        batched_outputs_2 = torch._C._functorch._remove_batch_dim(
            batched_outputs_1, 2, 1, 0
        )
        batched_outputs_1 = None
        _vmap_decrement_nesting_2 = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting_2 = None
        causal_mask = torch._C._functorch._remove_batch_dim(batched_outputs_2, 1, 1, 0)
        batched_outputs_2 = None
        _vmap_decrement_nesting_3 = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting_3 = None
        tensor = torch.tensor(
            0.0, device=device(type="cuda", index=0), dtype=torch.float32
        )
        mask = torch.where(causal_mask, tensor, -3.4028234663852886e38)
        causal_mask = tensor = None
        attention_mask_1 = l_attention_mask_.to(
            device=device(type="cuda", index=0), dtype=torch.bool
        )
        l_attention_mask_ = None
        kv_arange_2 = torch.arange(10, device=device(type="cuda", index=0))
        kv_arange_2 += 0
        kv_arange_3 = kv_arange_2
        kv_arange_2 = None
        batch_arange_1 = torch.arange(1, device=device(type="cuda", index=0))
        head_arange_1 = torch.arange(1, device=device(type="cuda", index=0))
        lazy_load_decompositions_4 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_4 = None
        _vmap_increment_nesting_4 = torch._C._functorch._vmap_increment_nesting(
            1, "error"
        )
        _vmap_increment_nesting_4 = None
        child_4 = torch._C._functorch._add_batch_dim(batch_arange_1, 0, 1)
        batch_arange_1 = None
        lazy_load_decompositions_5 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_5 = None
        _vmap_increment_nesting_5 = torch._C._functorch._vmap_increment_nesting(
            1, "error"
        )
        _vmap_increment_nesting_5 = None
        child_5 = torch._C._functorch._add_batch_dim(head_arange_1, 0, 2)
        head_arange_1 = child_5 = None
        lazy_load_decompositions_6 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_6 = None
        _vmap_increment_nesting_6 = torch._C._functorch._vmap_increment_nesting(
            10, "error"
        )
        _vmap_increment_nesting_6 = None
        child_6 = torch._C._functorch._add_batch_dim(cache_position, 0, 3)
        cache_position = None
        lazy_load_decompositions_7 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_7 = None
        _vmap_increment_nesting_7 = torch._C._functorch._vmap_increment_nesting(
            10, "error"
        )
        _vmap_increment_nesting_7 = None
        child_7 = torch._C._functorch._add_batch_dim(kv_arange_3, 0, 4)
        kv_arange_3 = None
        result_3 = child_6.new_ones((), dtype=torch.bool)
        result_4 = child_6.new_ones((), dtype=torch.bool)
        sub = child_6.sub(64)
        gt = child_7.gt(sub)
        sub = None
        result_5 = result_4.__and__(gt)
        result_4 = gt = None
        le_1 = child_7.le(child_6)
        child_6 = None
        result_6 = result_5.__and__(le_1)
        result_5 = le_1 = None
        result_7 = result_3.__and__(result_6)
        result_3 = result_6 = None
        index_1 = torch.ops.aten.index(attention_mask_1, [child_4, child_7])
        attention_mask_1 = child_4 = child_7 = None
        result_8 = result_7.__and__(index_1)
        result_7 = index_1 = None
        batched_outputs_3 = torch._C._functorch._remove_batch_dim(result_8, 4, 10, 0)
        result_8 = None
        _vmap_decrement_nesting_4 = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting_4 = None
        batched_outputs_4 = torch._C._functorch._remove_batch_dim(
            batched_outputs_3, 3, 10, 0
        )
        batched_outputs_3 = None
        _vmap_decrement_nesting_5 = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting_5 = None
        batched_outputs_5 = torch._C._functorch._remove_batch_dim(
            batched_outputs_4, 2, 1, 0
        )
        batched_outputs_4 = None
        _vmap_decrement_nesting_6 = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting_6 = None
        causal_mask_1 = torch._C._functorch._remove_batch_dim(
            batched_outputs_5, 1, 1, 0
        )
        batched_outputs_5 = None
        _vmap_decrement_nesting_7 = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting_7 = None
        tensor_1 = torch.tensor(
            0.0, device=device(type="cuda", index=0), dtype=torch.float32
        )
        mask_1 = torch.where(causal_mask_1, tensor_1, -3.4028234663852886e38)
        causal_mask_1 = tensor_1 = None
        getitem = l_self_modules_global_rotary_emb_buffers_inv_freq_[
            (None, slice(None, None, None), None)
        ]
        l_self_modules_global_rotary_emb_buffers_inv_freq_ = None
        float_1 = getitem.float()
        getitem = None
        expand_1 = float_1.expand(1, -1, 1)
        float_1 = None
        inv_freq_expanded = expand_1.to(device(type="cuda", index=0))
        expand_1 = None
        getitem_1 = position_ids[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        position_ids_expanded = getitem_1.float()
        getitem_1 = None
        _enter_autocast = torch.amp.autocast_mode._enter_autocast(
            "cuda", None, False, None
        )
        float_3 = inv_freq_expanded.float()
        inv_freq_expanded = None
        float_4 = position_ids_expanded.float()
        position_ids_expanded = None
        matmul = float_3 @ float_4
        float_3 = float_4 = None
        freqs = matmul.transpose(1, 2)
        matmul = None
        emb = torch.cat((freqs, freqs), dim=-1)
        freqs = None
        cos = emb.cos()
        cos_1 = cos * 1.0
        cos = None
        sin = emb.sin()
        emb = None
        sin_1 = sin * 1.0
        sin = None
        _exit_autocast = torch.amp.autocast_mode._exit_autocast(_enter_autocast)
        _enter_autocast = _exit_autocast = None
        cos_4 = cos_1.to(dtype=torch.float32)
        cos_1 = None
        sin_4 = sin_1.to(dtype=torch.float32)
        sin_1 = None
        getitem_2 = l_self_modules_local_rotary_emb_buffers_inv_freq_[
            (None, slice(None, None, None), None)
        ]
        l_self_modules_local_rotary_emb_buffers_inv_freq_ = None
        float_5 = getitem_2.float()
        getitem_2 = None
        expand_2 = float_5.expand(1, -1, 1)
        float_5 = None
        inv_freq_expanded_1 = expand_2.to(device(type="cuda", index=0))
        expand_2 = None
        getitem_3 = position_ids[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        position_ids = None
        position_ids_expanded_1 = getitem_3.float()
        getitem_3 = None
        _enter_autocast_1 = torch.amp.autocast_mode._enter_autocast(
            "cuda", None, False, None
        )
        float_7 = inv_freq_expanded_1.float()
        inv_freq_expanded_1 = None
        float_8 = position_ids_expanded_1.float()
        position_ids_expanded_1 = None
        matmul_1 = float_7 @ float_8
        float_7 = float_8 = None
        freqs_1 = matmul_1.transpose(1, 2)
        matmul_1 = None
        emb_1 = torch.cat((freqs_1, freqs_1), dim=-1)
        freqs_1 = None
        cos_2 = emb_1.cos()
        cos_3 = cos_2 * 1.0
        cos_2 = None
        sin_2 = emb_1.sin()
        emb_1 = None
        sin_3 = sin_2 * 1.0
        sin_2 = None
        _exit_autocast_1 = torch.amp.autocast_mode._exit_autocast(_enter_autocast_1)
        _enter_autocast_1 = _exit_autocast_1 = None
        cos_6 = cos_3.to(dtype=torch.float32)
        cos_3 = None
        sin_6 = sin_3.to(dtype=torch.float32)
        sin_3 = None
        linear = torch._C._nn.linear(
            hidden_states,
            l_self_modules_layers_modules_0_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_0_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view = linear.view((1, 10, -1, 64))
        linear = None
        query_states = view.transpose(1, 2)
        view = None
        linear_1 = torch._C._nn.linear(
            hidden_states,
            l_self_modules_layers_modules_0_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_0_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_1 = linear_1.view((1, 10, -1, 64))
        linear_1 = None
        key_states = view_1.transpose(1, 2)
        view_1 = None
        linear_2 = torch._C._nn.linear(
            hidden_states,
            l_self_modules_layers_modules_0_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_0_modules_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_2 = linear_2.view((1, 10, -1, 64))
        linear_2 = None
        value_states = view_2.transpose(1, 2)
        view_2 = None
        cos_5 = cos_4.unsqueeze(1)
        sin_5 = sin_4.unsqueeze(1)
        mul_4 = query_states * cos_5
        x1 = query_states[(Ellipsis, slice(None, 32, None))]
        x2 = query_states[(Ellipsis, slice(32, None, None))]
        query_states = None
        neg = -x2
        x2 = None
        cat_2 = torch.cat((neg, x1), dim=-1)
        neg = x1 = None
        mul_5 = cat_2 * sin_5
        cat_2 = None
        q_embed = mul_4 + mul_5
        mul_4 = mul_5 = None
        mul_6 = key_states * cos_5
        cos_5 = None
        x1_1 = key_states[(Ellipsis, slice(None, 32, None))]
        x2_1 = key_states[(Ellipsis, slice(32, None, None))]
        key_states = None
        neg_1 = -x2_1
        x2_1 = None
        cat_3 = torch.cat((neg_1, x1_1), dim=-1)
        neg_1 = x1_1 = None
        mul_7 = cat_3 * sin_5
        cat_3 = sin_5 = None
        k_embed = mul_6 + mul_7
        mul_6 = mul_7 = None
        transpose_5 = k_embed.transpose(2, 3)
        matmul_2 = torch.matmul(q_embed, transpose_5)
        q_embed = transpose_5 = None
        attn_weights = matmul_2 * 0.125
        matmul_2 = None
        causal_mask_2 = mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        attn_weights_1 = attn_weights + causal_mask_2
        attn_weights = causal_mask_2 = None
        softmax = torch.nn.functional.softmax(
            attn_weights_1, dim=-1, dtype=torch.float32
        )
        attn_weights_1 = None
        attn_weights_2 = softmax.to(torch.float32)
        softmax = None
        attn_weights_3 = torch.nn.functional.dropout(
            attn_weights_2, p=0.0, training=False
        )
        attn_weights_2 = None
        attn_output = torch.matmul(attn_weights_3, value_states)
        attn_weights_3 = None
        transpose_6 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_6.contiguous()
        transpose_6 = None
        reshape = attn_output_1.reshape(1, 10, -1)
        attn_output_1 = None
        attn_output_2 = reshape.contiguous()
        reshape = None
        linear_3 = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_layers_modules_0_modules_attn_modules_wo_parameters_weight_,
            None,
        )
        attn_output_2 = (
            l_self_modules_layers_modules_0_modules_attn_modules_wo_parameters_weight_
        ) = None
        attn_output_3 = torch.nn.functional.dropout(linear_3, 0.0, False, False)
        linear_3 = None
        hidden_states_1 = hidden_states + attn_output_3
        hidden_states = attn_output_3 = None
        hidden_states_2 = torch.nn.functional.layer_norm(
            hidden_states_1,
            (768,),
            l_self_modules_layers_modules_0_modules_mlp_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_0_modules_mlp_norm_parameters_weight_ = None
        linear_4 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_layers_modules_0_modules_mlp_modules_wi_parameters_weight_,
            None,
        )
        hidden_states_2 = (
            l_self_modules_layers_modules_0_modules_mlp_modules_wi_parameters_weight_
        ) = None
        chunk = linear_4.chunk(2, dim=-1)
        linear_4 = None
        input_1 = chunk[0]
        gate = chunk[1]
        chunk = None
        gelu = torch._C._nn.gelu(input_1)
        input_1 = None
        mul_9 = gelu * gate
        gelu = gate = None
        dropout_3 = torch.nn.functional.dropout(mul_9, 0.0, False, False)
        mul_9 = None
        mlp_output = torch._C._nn.linear(
            dropout_3,
            l_self_modules_layers_modules_0_modules_mlp_modules_wo_parameters_weight_,
            None,
        )
        dropout_3 = (
            l_self_modules_layers_modules_0_modules_mlp_modules_wo_parameters_weight_
        ) = None
        hidden_states_3 = hidden_states_1 + mlp_output
        hidden_states_1 = mlp_output = None
        hidden_states_4 = torch.nn.functional.layer_norm(
            hidden_states_3,
            (768,),
            l_self_modules_layers_modules_1_modules_attn_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_1_modules_attn_norm_parameters_weight_ = None
        linear_6 = torch._C._nn.linear(
            hidden_states_4,
            l_self_modules_layers_modules_1_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_1_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_3 = linear_6.view((1, 10, -1, 64))
        linear_6 = None
        query_states_1 = view_3.transpose(1, 2)
        view_3 = None
        linear_7 = torch._C._nn.linear(
            hidden_states_4,
            l_self_modules_layers_modules_1_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_1_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_4 = linear_7.view((1, 10, -1, 64))
        linear_7 = None
        key_states_1 = view_4.transpose(1, 2)
        view_4 = None
        linear_8 = torch._C._nn.linear(
            hidden_states_4,
            l_self_modules_layers_modules_1_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_4 = l_self_modules_layers_modules_1_modules_attn_modules_v_proj_parameters_weight_ = (None)
        view_5 = linear_8.view((1, 10, -1, 64))
        linear_8 = None
        value_states_1 = view_5.transpose(1, 2)
        view_5 = None
        cos_7 = cos_6.unsqueeze(1)
        sin_7 = sin_6.unsqueeze(1)
        mul_10 = query_states_1 * cos_7
        x1_2 = query_states_1[(Ellipsis, slice(None, 32, None))]
        x2_2 = query_states_1[(Ellipsis, slice(32, None, None))]
        query_states_1 = None
        neg_2 = -x2_2
        x2_2 = None
        cat_4 = torch.cat((neg_2, x1_2), dim=-1)
        neg_2 = x1_2 = None
        mul_11 = cat_4 * sin_7
        cat_4 = None
        q_embed_1 = mul_10 + mul_11
        mul_10 = mul_11 = None
        mul_12 = key_states_1 * cos_7
        cos_7 = None
        x1_3 = key_states_1[(Ellipsis, slice(None, 32, None))]
        x2_3 = key_states_1[(Ellipsis, slice(32, None, None))]
        key_states_1 = None
        neg_3 = -x2_3
        x2_3 = None
        cat_5 = torch.cat((neg_3, x1_3), dim=-1)
        neg_3 = x1_3 = None
        mul_13 = cat_5 * sin_7
        cat_5 = sin_7 = None
        k_embed_1 = mul_12 + mul_13
        mul_12 = mul_13 = None
        transpose_10 = k_embed_1.transpose(2, 3)
        matmul_4 = torch.matmul(q_embed_1, transpose_10)
        q_embed_1 = transpose_10 = None
        attn_weights_4 = matmul_4 * 0.125
        matmul_4 = None
        causal_mask_3 = mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        attn_weights_5 = attn_weights_4 + causal_mask_3
        attn_weights_4 = causal_mask_3 = None
        softmax_1 = torch.nn.functional.softmax(
            attn_weights_5, dim=-1, dtype=torch.float32
        )
        attn_weights_5 = None
        attn_weights_6 = softmax_1.to(torch.float32)
        softmax_1 = None
        attn_weights_7 = torch.nn.functional.dropout(
            attn_weights_6, p=0.0, training=False
        )
        attn_weights_6 = None
        attn_output_4 = torch.matmul(attn_weights_7, value_states_1)
        attn_weights_7 = None
        transpose_11 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_11.contiguous()
        transpose_11 = None
        reshape_1 = attn_output_5.reshape(1, 10, -1)
        attn_output_5 = None
        attn_output_6 = reshape_1.contiguous()
        reshape_1 = None
        linear_9 = torch._C._nn.linear(
            attn_output_6,
            l_self_modules_layers_modules_1_modules_attn_modules_wo_parameters_weight_,
            None,
        )
        attn_output_6 = (
            l_self_modules_layers_modules_1_modules_attn_modules_wo_parameters_weight_
        ) = None
        attn_output_7 = torch.nn.functional.dropout(linear_9, 0.0, False, False)
        linear_9 = None
        hidden_states_5 = hidden_states_3 + attn_output_7
        hidden_states_3 = attn_output_7 = None
        hidden_states_6 = torch.nn.functional.layer_norm(
            hidden_states_5,
            (768,),
            l_self_modules_layers_modules_1_modules_mlp_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_1_modules_mlp_norm_parameters_weight_ = None
        linear_10 = torch._C._nn.linear(
            hidden_states_6,
            l_self_modules_layers_modules_1_modules_mlp_modules_wi_parameters_weight_,
            None,
        )
        hidden_states_6 = (
            l_self_modules_layers_modules_1_modules_mlp_modules_wi_parameters_weight_
        ) = None
        chunk_1 = linear_10.chunk(2, dim=-1)
        linear_10 = None
        input_2 = chunk_1[0]
        gate_1 = chunk_1[1]
        chunk_1 = None
        gelu_1 = torch._C._nn.gelu(input_2)
        input_2 = None
        mul_15 = gelu_1 * gate_1
        gelu_1 = gate_1 = None
        dropout_6 = torch.nn.functional.dropout(mul_15, 0.0, False, False)
        mul_15 = None
        mlp_output_1 = torch._C._nn.linear(
            dropout_6,
            l_self_modules_layers_modules_1_modules_mlp_modules_wo_parameters_weight_,
            None,
        )
        dropout_6 = (
            l_self_modules_layers_modules_1_modules_mlp_modules_wo_parameters_weight_
        ) = None
        hidden_states_7 = hidden_states_5 + mlp_output_1
        hidden_states_5 = mlp_output_1 = None
        hidden_states_8 = torch.nn.functional.layer_norm(
            hidden_states_7,
            (768,),
            l_self_modules_layers_modules_2_modules_attn_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_2_modules_attn_norm_parameters_weight_ = None
        linear_12 = torch._C._nn.linear(
            hidden_states_8,
            l_self_modules_layers_modules_2_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_2_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_6 = linear_12.view((1, 10, -1, 64))
        linear_12 = None
        query_states_2 = view_6.transpose(1, 2)
        view_6 = None
        linear_13 = torch._C._nn.linear(
            hidden_states_8,
            l_self_modules_layers_modules_2_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_2_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_7 = linear_13.view((1, 10, -1, 64))
        linear_13 = None
        key_states_2 = view_7.transpose(1, 2)
        view_7 = None
        linear_14 = torch._C._nn.linear(
            hidden_states_8,
            l_self_modules_layers_modules_2_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_8 = l_self_modules_layers_modules_2_modules_attn_modules_v_proj_parameters_weight_ = (None)
        view_8 = linear_14.view((1, 10, -1, 64))
        linear_14 = None
        value_states_2 = view_8.transpose(1, 2)
        view_8 = None
        cos_8 = cos_6.unsqueeze(1)
        sin_8 = sin_6.unsqueeze(1)
        mul_16 = query_states_2 * cos_8
        x1_4 = query_states_2[(Ellipsis, slice(None, 32, None))]
        x2_4 = query_states_2[(Ellipsis, slice(32, None, None))]
        query_states_2 = None
        neg_4 = -x2_4
        x2_4 = None
        cat_6 = torch.cat((neg_4, x1_4), dim=-1)
        neg_4 = x1_4 = None
        mul_17 = cat_6 * sin_8
        cat_6 = None
        q_embed_2 = mul_16 + mul_17
        mul_16 = mul_17 = None
        mul_18 = key_states_2 * cos_8
        cos_8 = None
        x1_5 = key_states_2[(Ellipsis, slice(None, 32, None))]
        x2_5 = key_states_2[(Ellipsis, slice(32, None, None))]
        key_states_2 = None
        neg_5 = -x2_5
        x2_5 = None
        cat_7 = torch.cat((neg_5, x1_5), dim=-1)
        neg_5 = x1_5 = None
        mul_19 = cat_7 * sin_8
        cat_7 = sin_8 = None
        k_embed_2 = mul_18 + mul_19
        mul_18 = mul_19 = None
        transpose_15 = k_embed_2.transpose(2, 3)
        matmul_6 = torch.matmul(q_embed_2, transpose_15)
        q_embed_2 = transpose_15 = None
        attn_weights_8 = matmul_6 * 0.125
        matmul_6 = None
        causal_mask_4 = mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        attn_weights_9 = attn_weights_8 + causal_mask_4
        attn_weights_8 = causal_mask_4 = None
        softmax_2 = torch.nn.functional.softmax(
            attn_weights_9, dim=-1, dtype=torch.float32
        )
        attn_weights_9 = None
        attn_weights_10 = softmax_2.to(torch.float32)
        softmax_2 = None
        attn_weights_11 = torch.nn.functional.dropout(
            attn_weights_10, p=0.0, training=False
        )
        attn_weights_10 = None
        attn_output_8 = torch.matmul(attn_weights_11, value_states_2)
        attn_weights_11 = None
        transpose_16 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_16.contiguous()
        transpose_16 = None
        reshape_2 = attn_output_9.reshape(1, 10, -1)
        attn_output_9 = None
        attn_output_10 = reshape_2.contiguous()
        reshape_2 = None
        linear_15 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_layers_modules_2_modules_attn_modules_wo_parameters_weight_,
            None,
        )
        attn_output_10 = (
            l_self_modules_layers_modules_2_modules_attn_modules_wo_parameters_weight_
        ) = None
        attn_output_11 = torch.nn.functional.dropout(linear_15, 0.0, False, False)
        linear_15 = None
        hidden_states_9 = hidden_states_7 + attn_output_11
        hidden_states_7 = attn_output_11 = None
        hidden_states_10 = torch.nn.functional.layer_norm(
            hidden_states_9,
            (768,),
            l_self_modules_layers_modules_2_modules_mlp_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_2_modules_mlp_norm_parameters_weight_ = None
        linear_16 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_layers_modules_2_modules_mlp_modules_wi_parameters_weight_,
            None,
        )
        hidden_states_10 = (
            l_self_modules_layers_modules_2_modules_mlp_modules_wi_parameters_weight_
        ) = None
        chunk_2 = linear_16.chunk(2, dim=-1)
        linear_16 = None
        input_3 = chunk_2[0]
        gate_2 = chunk_2[1]
        chunk_2 = None
        gelu_2 = torch._C._nn.gelu(input_3)
        input_3 = None
        mul_21 = gelu_2 * gate_2
        gelu_2 = gate_2 = None
        dropout_9 = torch.nn.functional.dropout(mul_21, 0.0, False, False)
        mul_21 = None
        mlp_output_2 = torch._C._nn.linear(
            dropout_9,
            l_self_modules_layers_modules_2_modules_mlp_modules_wo_parameters_weight_,
            None,
        )
        dropout_9 = (
            l_self_modules_layers_modules_2_modules_mlp_modules_wo_parameters_weight_
        ) = None
        hidden_states_11 = hidden_states_9 + mlp_output_2
        hidden_states_9 = mlp_output_2 = None
        hidden_states_12 = torch.nn.functional.layer_norm(
            hidden_states_11,
            (768,),
            l_self_modules_layers_modules_3_modules_attn_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_3_modules_attn_norm_parameters_weight_ = None
        linear_18 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_layers_modules_3_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_3_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_9 = linear_18.view((1, 10, -1, 64))
        linear_18 = None
        query_states_3 = view_9.transpose(1, 2)
        view_9 = None
        linear_19 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_layers_modules_3_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_3_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_10 = linear_19.view((1, 10, -1, 64))
        linear_19 = None
        key_states_3 = view_10.transpose(1, 2)
        view_10 = None
        linear_20 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_layers_modules_3_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_12 = l_self_modules_layers_modules_3_modules_attn_modules_v_proj_parameters_weight_ = (None)
        view_11 = linear_20.view((1, 10, -1, 64))
        linear_20 = None
        value_states_3 = view_11.transpose(1, 2)
        view_11 = None
        cos_9 = cos_4.unsqueeze(1)
        sin_9 = sin_4.unsqueeze(1)
        mul_22 = query_states_3 * cos_9
        x1_6 = query_states_3[(Ellipsis, slice(None, 32, None))]
        x2_6 = query_states_3[(Ellipsis, slice(32, None, None))]
        query_states_3 = None
        neg_6 = -x2_6
        x2_6 = None
        cat_8 = torch.cat((neg_6, x1_6), dim=-1)
        neg_6 = x1_6 = None
        mul_23 = cat_8 * sin_9
        cat_8 = None
        q_embed_3 = mul_22 + mul_23
        mul_22 = mul_23 = None
        mul_24 = key_states_3 * cos_9
        cos_9 = None
        x1_7 = key_states_3[(Ellipsis, slice(None, 32, None))]
        x2_7 = key_states_3[(Ellipsis, slice(32, None, None))]
        key_states_3 = None
        neg_7 = -x2_7
        x2_7 = None
        cat_9 = torch.cat((neg_7, x1_7), dim=-1)
        neg_7 = x1_7 = None
        mul_25 = cat_9 * sin_9
        cat_9 = sin_9 = None
        k_embed_3 = mul_24 + mul_25
        mul_24 = mul_25 = None
        transpose_20 = k_embed_3.transpose(2, 3)
        matmul_8 = torch.matmul(q_embed_3, transpose_20)
        q_embed_3 = transpose_20 = None
        attn_weights_12 = matmul_8 * 0.125
        matmul_8 = None
        causal_mask_5 = mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        attn_weights_13 = attn_weights_12 + causal_mask_5
        attn_weights_12 = causal_mask_5 = None
        softmax_3 = torch.nn.functional.softmax(
            attn_weights_13, dim=-1, dtype=torch.float32
        )
        attn_weights_13 = None
        attn_weights_14 = softmax_3.to(torch.float32)
        softmax_3 = None
        attn_weights_15 = torch.nn.functional.dropout(
            attn_weights_14, p=0.0, training=False
        )
        attn_weights_14 = None
        attn_output_12 = torch.matmul(attn_weights_15, value_states_3)
        attn_weights_15 = None
        transpose_21 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_21.contiguous()
        transpose_21 = None
        reshape_3 = attn_output_13.reshape(1, 10, -1)
        attn_output_13 = None
        attn_output_14 = reshape_3.contiguous()
        reshape_3 = None
        linear_21 = torch._C._nn.linear(
            attn_output_14,
            l_self_modules_layers_modules_3_modules_attn_modules_wo_parameters_weight_,
            None,
        )
        attn_output_14 = (
            l_self_modules_layers_modules_3_modules_attn_modules_wo_parameters_weight_
        ) = None
        attn_output_15 = torch.nn.functional.dropout(linear_21, 0.0, False, False)
        linear_21 = None
        hidden_states_13 = hidden_states_11 + attn_output_15
        hidden_states_11 = attn_output_15 = None
        hidden_states_14 = torch.nn.functional.layer_norm(
            hidden_states_13,
            (768,),
            l_self_modules_layers_modules_3_modules_mlp_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_3_modules_mlp_norm_parameters_weight_ = None
        linear_22 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_layers_modules_3_modules_mlp_modules_wi_parameters_weight_,
            None,
        )
        hidden_states_14 = (
            l_self_modules_layers_modules_3_modules_mlp_modules_wi_parameters_weight_
        ) = None
        chunk_3 = linear_22.chunk(2, dim=-1)
        linear_22 = None
        input_4 = chunk_3[0]
        gate_3 = chunk_3[1]
        chunk_3 = None
        gelu_3 = torch._C._nn.gelu(input_4)
        input_4 = None
        mul_27 = gelu_3 * gate_3
        gelu_3 = gate_3 = None
        dropout_12 = torch.nn.functional.dropout(mul_27, 0.0, False, False)
        mul_27 = None
        mlp_output_3 = torch._C._nn.linear(
            dropout_12,
            l_self_modules_layers_modules_3_modules_mlp_modules_wo_parameters_weight_,
            None,
        )
        dropout_12 = (
            l_self_modules_layers_modules_3_modules_mlp_modules_wo_parameters_weight_
        ) = None
        hidden_states_15 = hidden_states_13 + mlp_output_3
        hidden_states_13 = mlp_output_3 = None
        hidden_states_16 = torch.nn.functional.layer_norm(
            hidden_states_15,
            (768,),
            l_self_modules_layers_modules_4_modules_attn_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_4_modules_attn_norm_parameters_weight_ = None
        linear_24 = torch._C._nn.linear(
            hidden_states_16,
            l_self_modules_layers_modules_4_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_4_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_12 = linear_24.view((1, 10, -1, 64))
        linear_24 = None
        query_states_4 = view_12.transpose(1, 2)
        view_12 = None
        linear_25 = torch._C._nn.linear(
            hidden_states_16,
            l_self_modules_layers_modules_4_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_4_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_13 = linear_25.view((1, 10, -1, 64))
        linear_25 = None
        key_states_4 = view_13.transpose(1, 2)
        view_13 = None
        linear_26 = torch._C._nn.linear(
            hidden_states_16,
            l_self_modules_layers_modules_4_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_16 = l_self_modules_layers_modules_4_modules_attn_modules_v_proj_parameters_weight_ = (None)
        view_14 = linear_26.view((1, 10, -1, 64))
        linear_26 = None
        value_states_4 = view_14.transpose(1, 2)
        view_14 = None
        cos_10 = cos_6.unsqueeze(1)
        sin_10 = sin_6.unsqueeze(1)
        mul_28 = query_states_4 * cos_10
        x1_8 = query_states_4[(Ellipsis, slice(None, 32, None))]
        x2_8 = query_states_4[(Ellipsis, slice(32, None, None))]
        query_states_4 = None
        neg_8 = -x2_8
        x2_8 = None
        cat_10 = torch.cat((neg_8, x1_8), dim=-1)
        neg_8 = x1_8 = None
        mul_29 = cat_10 * sin_10
        cat_10 = None
        q_embed_4 = mul_28 + mul_29
        mul_28 = mul_29 = None
        mul_30 = key_states_4 * cos_10
        cos_10 = None
        x1_9 = key_states_4[(Ellipsis, slice(None, 32, None))]
        x2_9 = key_states_4[(Ellipsis, slice(32, None, None))]
        key_states_4 = None
        neg_9 = -x2_9
        x2_9 = None
        cat_11 = torch.cat((neg_9, x1_9), dim=-1)
        neg_9 = x1_9 = None
        mul_31 = cat_11 * sin_10
        cat_11 = sin_10 = None
        k_embed_4 = mul_30 + mul_31
        mul_30 = mul_31 = None
        transpose_25 = k_embed_4.transpose(2, 3)
        matmul_10 = torch.matmul(q_embed_4, transpose_25)
        q_embed_4 = transpose_25 = None
        attn_weights_16 = matmul_10 * 0.125
        matmul_10 = None
        causal_mask_6 = mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        attn_weights_17 = attn_weights_16 + causal_mask_6
        attn_weights_16 = causal_mask_6 = None
        softmax_4 = torch.nn.functional.softmax(
            attn_weights_17, dim=-1, dtype=torch.float32
        )
        attn_weights_17 = None
        attn_weights_18 = softmax_4.to(torch.float32)
        softmax_4 = None
        attn_weights_19 = torch.nn.functional.dropout(
            attn_weights_18, p=0.0, training=False
        )
        attn_weights_18 = None
        attn_output_16 = torch.matmul(attn_weights_19, value_states_4)
        attn_weights_19 = None
        transpose_26 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_17 = transpose_26.contiguous()
        transpose_26 = None
        reshape_4 = attn_output_17.reshape(1, 10, -1)
        attn_output_17 = None
        attn_output_18 = reshape_4.contiguous()
        reshape_4 = None
        linear_27 = torch._C._nn.linear(
            attn_output_18,
            l_self_modules_layers_modules_4_modules_attn_modules_wo_parameters_weight_,
            None,
        )
        attn_output_18 = (
            l_self_modules_layers_modules_4_modules_attn_modules_wo_parameters_weight_
        ) = None
        attn_output_19 = torch.nn.functional.dropout(linear_27, 0.0, False, False)
        linear_27 = None
        hidden_states_17 = hidden_states_15 + attn_output_19
        hidden_states_15 = attn_output_19 = None
        hidden_states_18 = torch.nn.functional.layer_norm(
            hidden_states_17,
            (768,),
            l_self_modules_layers_modules_4_modules_mlp_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_4_modules_mlp_norm_parameters_weight_ = None
        linear_28 = torch._C._nn.linear(
            hidden_states_18,
            l_self_modules_layers_modules_4_modules_mlp_modules_wi_parameters_weight_,
            None,
        )
        hidden_states_18 = (
            l_self_modules_layers_modules_4_modules_mlp_modules_wi_parameters_weight_
        ) = None
        chunk_4 = linear_28.chunk(2, dim=-1)
        linear_28 = None
        input_5 = chunk_4[0]
        gate_4 = chunk_4[1]
        chunk_4 = None
        gelu_4 = torch._C._nn.gelu(input_5)
        input_5 = None
        mul_33 = gelu_4 * gate_4
        gelu_4 = gate_4 = None
        dropout_15 = torch.nn.functional.dropout(mul_33, 0.0, False, False)
        mul_33 = None
        mlp_output_4 = torch._C._nn.linear(
            dropout_15,
            l_self_modules_layers_modules_4_modules_mlp_modules_wo_parameters_weight_,
            None,
        )
        dropout_15 = (
            l_self_modules_layers_modules_4_modules_mlp_modules_wo_parameters_weight_
        ) = None
        hidden_states_19 = hidden_states_17 + mlp_output_4
        hidden_states_17 = mlp_output_4 = None
        hidden_states_20 = torch.nn.functional.layer_norm(
            hidden_states_19,
            (768,),
            l_self_modules_layers_modules_5_modules_attn_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_5_modules_attn_norm_parameters_weight_ = None
        linear_30 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_layers_modules_5_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_5_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_15 = linear_30.view((1, 10, -1, 64))
        linear_30 = None
        query_states_5 = view_15.transpose(1, 2)
        view_15 = None
        linear_31 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_layers_modules_5_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_5_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_16 = linear_31.view((1, 10, -1, 64))
        linear_31 = None
        key_states_5 = view_16.transpose(1, 2)
        view_16 = None
        linear_32 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_layers_modules_5_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_20 = l_self_modules_layers_modules_5_modules_attn_modules_v_proj_parameters_weight_ = (None)
        view_17 = linear_32.view((1, 10, -1, 64))
        linear_32 = None
        value_states_5 = view_17.transpose(1, 2)
        view_17 = None
        cos_11 = cos_6.unsqueeze(1)
        sin_11 = sin_6.unsqueeze(1)
        mul_34 = query_states_5 * cos_11
        x1_10 = query_states_5[(Ellipsis, slice(None, 32, None))]
        x2_10 = query_states_5[(Ellipsis, slice(32, None, None))]
        query_states_5 = None
        neg_10 = -x2_10
        x2_10 = None
        cat_12 = torch.cat((neg_10, x1_10), dim=-1)
        neg_10 = x1_10 = None
        mul_35 = cat_12 * sin_11
        cat_12 = None
        q_embed_5 = mul_34 + mul_35
        mul_34 = mul_35 = None
        mul_36 = key_states_5 * cos_11
        cos_11 = None
        x1_11 = key_states_5[(Ellipsis, slice(None, 32, None))]
        x2_11 = key_states_5[(Ellipsis, slice(32, None, None))]
        key_states_5 = None
        neg_11 = -x2_11
        x2_11 = None
        cat_13 = torch.cat((neg_11, x1_11), dim=-1)
        neg_11 = x1_11 = None
        mul_37 = cat_13 * sin_11
        cat_13 = sin_11 = None
        k_embed_5 = mul_36 + mul_37
        mul_36 = mul_37 = None
        transpose_30 = k_embed_5.transpose(2, 3)
        matmul_12 = torch.matmul(q_embed_5, transpose_30)
        q_embed_5 = transpose_30 = None
        attn_weights_20 = matmul_12 * 0.125
        matmul_12 = None
        causal_mask_7 = mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        attn_weights_21 = attn_weights_20 + causal_mask_7
        attn_weights_20 = causal_mask_7 = None
        softmax_5 = torch.nn.functional.softmax(
            attn_weights_21, dim=-1, dtype=torch.float32
        )
        attn_weights_21 = None
        attn_weights_22 = softmax_5.to(torch.float32)
        softmax_5 = None
        attn_weights_23 = torch.nn.functional.dropout(
            attn_weights_22, p=0.0, training=False
        )
        attn_weights_22 = None
        attn_output_20 = torch.matmul(attn_weights_23, value_states_5)
        attn_weights_23 = None
        transpose_31 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_31.contiguous()
        transpose_31 = None
        reshape_5 = attn_output_21.reshape(1, 10, -1)
        attn_output_21 = None
        attn_output_22 = reshape_5.contiguous()
        reshape_5 = None
        linear_33 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_layers_modules_5_modules_attn_modules_wo_parameters_weight_,
            None,
        )
        attn_output_22 = (
            l_self_modules_layers_modules_5_modules_attn_modules_wo_parameters_weight_
        ) = None
        attn_output_23 = torch.nn.functional.dropout(linear_33, 0.0, False, False)
        linear_33 = None
        hidden_states_21 = hidden_states_19 + attn_output_23
        hidden_states_19 = attn_output_23 = None
        hidden_states_22 = torch.nn.functional.layer_norm(
            hidden_states_21,
            (768,),
            l_self_modules_layers_modules_5_modules_mlp_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_5_modules_mlp_norm_parameters_weight_ = None
        linear_34 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_layers_modules_5_modules_mlp_modules_wi_parameters_weight_,
            None,
        )
        hidden_states_22 = (
            l_self_modules_layers_modules_5_modules_mlp_modules_wi_parameters_weight_
        ) = None
        chunk_5 = linear_34.chunk(2, dim=-1)
        linear_34 = None
        input_6 = chunk_5[0]
        gate_5 = chunk_5[1]
        chunk_5 = None
        gelu_5 = torch._C._nn.gelu(input_6)
        input_6 = None
        mul_39 = gelu_5 * gate_5
        gelu_5 = gate_5 = None
        dropout_18 = torch.nn.functional.dropout(mul_39, 0.0, False, False)
        mul_39 = None
        mlp_output_5 = torch._C._nn.linear(
            dropout_18,
            l_self_modules_layers_modules_5_modules_mlp_modules_wo_parameters_weight_,
            None,
        )
        dropout_18 = (
            l_self_modules_layers_modules_5_modules_mlp_modules_wo_parameters_weight_
        ) = None
        hidden_states_23 = hidden_states_21 + mlp_output_5
        hidden_states_21 = mlp_output_5 = None
        hidden_states_24 = torch.nn.functional.layer_norm(
            hidden_states_23,
            (768,),
            l_self_modules_layers_modules_6_modules_attn_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_6_modules_attn_norm_parameters_weight_ = None
        linear_36 = torch._C._nn.linear(
            hidden_states_24,
            l_self_modules_layers_modules_6_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_6_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_18 = linear_36.view((1, 10, -1, 64))
        linear_36 = None
        query_states_6 = view_18.transpose(1, 2)
        view_18 = None
        linear_37 = torch._C._nn.linear(
            hidden_states_24,
            l_self_modules_layers_modules_6_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_6_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_19 = linear_37.view((1, 10, -1, 64))
        linear_37 = None
        key_states_6 = view_19.transpose(1, 2)
        view_19 = None
        linear_38 = torch._C._nn.linear(
            hidden_states_24,
            l_self_modules_layers_modules_6_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_24 = l_self_modules_layers_modules_6_modules_attn_modules_v_proj_parameters_weight_ = (None)
        view_20 = linear_38.view((1, 10, -1, 64))
        linear_38 = None
        value_states_6 = view_20.transpose(1, 2)
        view_20 = None
        cos_12 = cos_4.unsqueeze(1)
        sin_12 = sin_4.unsqueeze(1)
        mul_40 = query_states_6 * cos_12
        x1_12 = query_states_6[(Ellipsis, slice(None, 32, None))]
        x2_12 = query_states_6[(Ellipsis, slice(32, None, None))]
        query_states_6 = None
        neg_12 = -x2_12
        x2_12 = None
        cat_14 = torch.cat((neg_12, x1_12), dim=-1)
        neg_12 = x1_12 = None
        mul_41 = cat_14 * sin_12
        cat_14 = None
        q_embed_6 = mul_40 + mul_41
        mul_40 = mul_41 = None
        mul_42 = key_states_6 * cos_12
        cos_12 = None
        x1_13 = key_states_6[(Ellipsis, slice(None, 32, None))]
        x2_13 = key_states_6[(Ellipsis, slice(32, None, None))]
        key_states_6 = None
        neg_13 = -x2_13
        x2_13 = None
        cat_15 = torch.cat((neg_13, x1_13), dim=-1)
        neg_13 = x1_13 = None
        mul_43 = cat_15 * sin_12
        cat_15 = sin_12 = None
        k_embed_6 = mul_42 + mul_43
        mul_42 = mul_43 = None
        transpose_35 = k_embed_6.transpose(2, 3)
        matmul_14 = torch.matmul(q_embed_6, transpose_35)
        q_embed_6 = transpose_35 = None
        attn_weights_24 = matmul_14 * 0.125
        matmul_14 = None
        causal_mask_8 = mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        attn_weights_25 = attn_weights_24 + causal_mask_8
        attn_weights_24 = causal_mask_8 = None
        softmax_6 = torch.nn.functional.softmax(
            attn_weights_25, dim=-1, dtype=torch.float32
        )
        attn_weights_25 = None
        attn_weights_26 = softmax_6.to(torch.float32)
        softmax_6 = None
        attn_weights_27 = torch.nn.functional.dropout(
            attn_weights_26, p=0.0, training=False
        )
        attn_weights_26 = None
        attn_output_24 = torch.matmul(attn_weights_27, value_states_6)
        attn_weights_27 = None
        transpose_36 = attn_output_24.transpose(1, 2)
        attn_output_24 = None
        attn_output_25 = transpose_36.contiguous()
        transpose_36 = None
        reshape_6 = attn_output_25.reshape(1, 10, -1)
        attn_output_25 = None
        attn_output_26 = reshape_6.contiguous()
        reshape_6 = None
        linear_39 = torch._C._nn.linear(
            attn_output_26,
            l_self_modules_layers_modules_6_modules_attn_modules_wo_parameters_weight_,
            None,
        )
        attn_output_26 = (
            l_self_modules_layers_modules_6_modules_attn_modules_wo_parameters_weight_
        ) = None
        attn_output_27 = torch.nn.functional.dropout(linear_39, 0.0, False, False)
        linear_39 = None
        hidden_states_25 = hidden_states_23 + attn_output_27
        hidden_states_23 = attn_output_27 = None
        hidden_states_26 = torch.nn.functional.layer_norm(
            hidden_states_25,
            (768,),
            l_self_modules_layers_modules_6_modules_mlp_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_6_modules_mlp_norm_parameters_weight_ = None
        linear_40 = torch._C._nn.linear(
            hidden_states_26,
            l_self_modules_layers_modules_6_modules_mlp_modules_wi_parameters_weight_,
            None,
        )
        hidden_states_26 = (
            l_self_modules_layers_modules_6_modules_mlp_modules_wi_parameters_weight_
        ) = None
        chunk_6 = linear_40.chunk(2, dim=-1)
        linear_40 = None
        input_7 = chunk_6[0]
        gate_6 = chunk_6[1]
        chunk_6 = None
        gelu_6 = torch._C._nn.gelu(input_7)
        input_7 = None
        mul_45 = gelu_6 * gate_6
        gelu_6 = gate_6 = None
        dropout_21 = torch.nn.functional.dropout(mul_45, 0.0, False, False)
        mul_45 = None
        mlp_output_6 = torch._C._nn.linear(
            dropout_21,
            l_self_modules_layers_modules_6_modules_mlp_modules_wo_parameters_weight_,
            None,
        )
        dropout_21 = (
            l_self_modules_layers_modules_6_modules_mlp_modules_wo_parameters_weight_
        ) = None
        hidden_states_27 = hidden_states_25 + mlp_output_6
        hidden_states_25 = mlp_output_6 = None
        hidden_states_28 = torch.nn.functional.layer_norm(
            hidden_states_27,
            (768,),
            l_self_modules_layers_modules_7_modules_attn_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_7_modules_attn_norm_parameters_weight_ = None
        linear_42 = torch._C._nn.linear(
            hidden_states_28,
            l_self_modules_layers_modules_7_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_7_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_21 = linear_42.view((1, 10, -1, 64))
        linear_42 = None
        query_states_7 = view_21.transpose(1, 2)
        view_21 = None
        linear_43 = torch._C._nn.linear(
            hidden_states_28,
            l_self_modules_layers_modules_7_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_7_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_22 = linear_43.view((1, 10, -1, 64))
        linear_43 = None
        key_states_7 = view_22.transpose(1, 2)
        view_22 = None
        linear_44 = torch._C._nn.linear(
            hidden_states_28,
            l_self_modules_layers_modules_7_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_28 = l_self_modules_layers_modules_7_modules_attn_modules_v_proj_parameters_weight_ = (None)
        view_23 = linear_44.view((1, 10, -1, 64))
        linear_44 = None
        value_states_7 = view_23.transpose(1, 2)
        view_23 = None
        cos_13 = cos_6.unsqueeze(1)
        sin_13 = sin_6.unsqueeze(1)
        mul_46 = query_states_7 * cos_13
        x1_14 = query_states_7[(Ellipsis, slice(None, 32, None))]
        x2_14 = query_states_7[(Ellipsis, slice(32, None, None))]
        query_states_7 = None
        neg_14 = -x2_14
        x2_14 = None
        cat_16 = torch.cat((neg_14, x1_14), dim=-1)
        neg_14 = x1_14 = None
        mul_47 = cat_16 * sin_13
        cat_16 = None
        q_embed_7 = mul_46 + mul_47
        mul_46 = mul_47 = None
        mul_48 = key_states_7 * cos_13
        cos_13 = None
        x1_15 = key_states_7[(Ellipsis, slice(None, 32, None))]
        x2_15 = key_states_7[(Ellipsis, slice(32, None, None))]
        key_states_7 = None
        neg_15 = -x2_15
        x2_15 = None
        cat_17 = torch.cat((neg_15, x1_15), dim=-1)
        neg_15 = x1_15 = None
        mul_49 = cat_17 * sin_13
        cat_17 = sin_13 = None
        k_embed_7 = mul_48 + mul_49
        mul_48 = mul_49 = None
        transpose_40 = k_embed_7.transpose(2, 3)
        matmul_16 = torch.matmul(q_embed_7, transpose_40)
        q_embed_7 = transpose_40 = None
        attn_weights_28 = matmul_16 * 0.125
        matmul_16 = None
        causal_mask_9 = mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        attn_weights_29 = attn_weights_28 + causal_mask_9
        attn_weights_28 = causal_mask_9 = None
        softmax_7 = torch.nn.functional.softmax(
            attn_weights_29, dim=-1, dtype=torch.float32
        )
        attn_weights_29 = None
        attn_weights_30 = softmax_7.to(torch.float32)
        softmax_7 = None
        attn_weights_31 = torch.nn.functional.dropout(
            attn_weights_30, p=0.0, training=False
        )
        attn_weights_30 = None
        attn_output_28 = torch.matmul(attn_weights_31, value_states_7)
        attn_weights_31 = None
        transpose_41 = attn_output_28.transpose(1, 2)
        attn_output_28 = None
        attn_output_29 = transpose_41.contiguous()
        transpose_41 = None
        reshape_7 = attn_output_29.reshape(1, 10, -1)
        attn_output_29 = None
        attn_output_30 = reshape_7.contiguous()
        reshape_7 = None
        linear_45 = torch._C._nn.linear(
            attn_output_30,
            l_self_modules_layers_modules_7_modules_attn_modules_wo_parameters_weight_,
            None,
        )
        attn_output_30 = (
            l_self_modules_layers_modules_7_modules_attn_modules_wo_parameters_weight_
        ) = None
        attn_output_31 = torch.nn.functional.dropout(linear_45, 0.0, False, False)
        linear_45 = None
        hidden_states_29 = hidden_states_27 + attn_output_31
        hidden_states_27 = attn_output_31 = None
        hidden_states_30 = torch.nn.functional.layer_norm(
            hidden_states_29,
            (768,),
            l_self_modules_layers_modules_7_modules_mlp_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_7_modules_mlp_norm_parameters_weight_ = None
        linear_46 = torch._C._nn.linear(
            hidden_states_30,
            l_self_modules_layers_modules_7_modules_mlp_modules_wi_parameters_weight_,
            None,
        )
        hidden_states_30 = (
            l_self_modules_layers_modules_7_modules_mlp_modules_wi_parameters_weight_
        ) = None
        chunk_7 = linear_46.chunk(2, dim=-1)
        linear_46 = None
        input_8 = chunk_7[0]
        gate_7 = chunk_7[1]
        chunk_7 = None
        gelu_7 = torch._C._nn.gelu(input_8)
        input_8 = None
        mul_51 = gelu_7 * gate_7
        gelu_7 = gate_7 = None
        dropout_24 = torch.nn.functional.dropout(mul_51, 0.0, False, False)
        mul_51 = None
        mlp_output_7 = torch._C._nn.linear(
            dropout_24,
            l_self_modules_layers_modules_7_modules_mlp_modules_wo_parameters_weight_,
            None,
        )
        dropout_24 = (
            l_self_modules_layers_modules_7_modules_mlp_modules_wo_parameters_weight_
        ) = None
        hidden_states_31 = hidden_states_29 + mlp_output_7
        hidden_states_29 = mlp_output_7 = None
        hidden_states_32 = torch.nn.functional.layer_norm(
            hidden_states_31,
            (768,),
            l_self_modules_layers_modules_8_modules_attn_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_8_modules_attn_norm_parameters_weight_ = None
        linear_48 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_layers_modules_8_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_8_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_24 = linear_48.view((1, 10, -1, 64))
        linear_48 = None
        query_states_8 = view_24.transpose(1, 2)
        view_24 = None
        linear_49 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_layers_modules_8_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_8_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_25 = linear_49.view((1, 10, -1, 64))
        linear_49 = None
        key_states_8 = view_25.transpose(1, 2)
        view_25 = None
        linear_50 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_layers_modules_8_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_32 = l_self_modules_layers_modules_8_modules_attn_modules_v_proj_parameters_weight_ = (None)
        view_26 = linear_50.view((1, 10, -1, 64))
        linear_50 = None
        value_states_8 = view_26.transpose(1, 2)
        view_26 = None
        cos_14 = cos_6.unsqueeze(1)
        sin_14 = sin_6.unsqueeze(1)
        mul_52 = query_states_8 * cos_14
        x1_16 = query_states_8[(Ellipsis, slice(None, 32, None))]
        x2_16 = query_states_8[(Ellipsis, slice(32, None, None))]
        query_states_8 = None
        neg_16 = -x2_16
        x2_16 = None
        cat_18 = torch.cat((neg_16, x1_16), dim=-1)
        neg_16 = x1_16 = None
        mul_53 = cat_18 * sin_14
        cat_18 = None
        q_embed_8 = mul_52 + mul_53
        mul_52 = mul_53 = None
        mul_54 = key_states_8 * cos_14
        cos_14 = None
        x1_17 = key_states_8[(Ellipsis, slice(None, 32, None))]
        x2_17 = key_states_8[(Ellipsis, slice(32, None, None))]
        key_states_8 = None
        neg_17 = -x2_17
        x2_17 = None
        cat_19 = torch.cat((neg_17, x1_17), dim=-1)
        neg_17 = x1_17 = None
        mul_55 = cat_19 * sin_14
        cat_19 = sin_14 = None
        k_embed_8 = mul_54 + mul_55
        mul_54 = mul_55 = None
        transpose_45 = k_embed_8.transpose(2, 3)
        matmul_18 = torch.matmul(q_embed_8, transpose_45)
        q_embed_8 = transpose_45 = None
        attn_weights_32 = matmul_18 * 0.125
        matmul_18 = None
        causal_mask_10 = mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        attn_weights_33 = attn_weights_32 + causal_mask_10
        attn_weights_32 = causal_mask_10 = None
        softmax_8 = torch.nn.functional.softmax(
            attn_weights_33, dim=-1, dtype=torch.float32
        )
        attn_weights_33 = None
        attn_weights_34 = softmax_8.to(torch.float32)
        softmax_8 = None
        attn_weights_35 = torch.nn.functional.dropout(
            attn_weights_34, p=0.0, training=False
        )
        attn_weights_34 = None
        attn_output_32 = torch.matmul(attn_weights_35, value_states_8)
        attn_weights_35 = None
        transpose_46 = attn_output_32.transpose(1, 2)
        attn_output_32 = None
        attn_output_33 = transpose_46.contiguous()
        transpose_46 = None
        reshape_8 = attn_output_33.reshape(1, 10, -1)
        attn_output_33 = None
        attn_output_34 = reshape_8.contiguous()
        reshape_8 = None
        linear_51 = torch._C._nn.linear(
            attn_output_34,
            l_self_modules_layers_modules_8_modules_attn_modules_wo_parameters_weight_,
            None,
        )
        attn_output_34 = (
            l_self_modules_layers_modules_8_modules_attn_modules_wo_parameters_weight_
        ) = None
        attn_output_35 = torch.nn.functional.dropout(linear_51, 0.0, False, False)
        linear_51 = None
        hidden_states_33 = hidden_states_31 + attn_output_35
        hidden_states_31 = attn_output_35 = None
        hidden_states_34 = torch.nn.functional.layer_norm(
            hidden_states_33,
            (768,),
            l_self_modules_layers_modules_8_modules_mlp_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_8_modules_mlp_norm_parameters_weight_ = None
        linear_52 = torch._C._nn.linear(
            hidden_states_34,
            l_self_modules_layers_modules_8_modules_mlp_modules_wi_parameters_weight_,
            None,
        )
        hidden_states_34 = (
            l_self_modules_layers_modules_8_modules_mlp_modules_wi_parameters_weight_
        ) = None
        chunk_8 = linear_52.chunk(2, dim=-1)
        linear_52 = None
        input_9 = chunk_8[0]
        gate_8 = chunk_8[1]
        chunk_8 = None
        gelu_8 = torch._C._nn.gelu(input_9)
        input_9 = None
        mul_57 = gelu_8 * gate_8
        gelu_8 = gate_8 = None
        dropout_27 = torch.nn.functional.dropout(mul_57, 0.0, False, False)
        mul_57 = None
        mlp_output_8 = torch._C._nn.linear(
            dropout_27,
            l_self_modules_layers_modules_8_modules_mlp_modules_wo_parameters_weight_,
            None,
        )
        dropout_27 = (
            l_self_modules_layers_modules_8_modules_mlp_modules_wo_parameters_weight_
        ) = None
        hidden_states_35 = hidden_states_33 + mlp_output_8
        hidden_states_33 = mlp_output_8 = None
        hidden_states_36 = torch.nn.functional.layer_norm(
            hidden_states_35,
            (768,),
            l_self_modules_layers_modules_9_modules_attn_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_9_modules_attn_norm_parameters_weight_ = None
        linear_54 = torch._C._nn.linear(
            hidden_states_36,
            l_self_modules_layers_modules_9_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_9_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_27 = linear_54.view((1, 10, -1, 64))
        linear_54 = None
        query_states_9 = view_27.transpose(1, 2)
        view_27 = None
        linear_55 = torch._C._nn.linear(
            hidden_states_36,
            l_self_modules_layers_modules_9_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_9_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_28 = linear_55.view((1, 10, -1, 64))
        linear_55 = None
        key_states_9 = view_28.transpose(1, 2)
        view_28 = None
        linear_56 = torch._C._nn.linear(
            hidden_states_36,
            l_self_modules_layers_modules_9_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_36 = l_self_modules_layers_modules_9_modules_attn_modules_v_proj_parameters_weight_ = (None)
        view_29 = linear_56.view((1, 10, -1, 64))
        linear_56 = None
        value_states_9 = view_29.transpose(1, 2)
        view_29 = None
        cos_15 = cos_4.unsqueeze(1)
        sin_15 = sin_4.unsqueeze(1)
        mul_58 = query_states_9 * cos_15
        x1_18 = query_states_9[(Ellipsis, slice(None, 32, None))]
        x2_18 = query_states_9[(Ellipsis, slice(32, None, None))]
        query_states_9 = None
        neg_18 = -x2_18
        x2_18 = None
        cat_20 = torch.cat((neg_18, x1_18), dim=-1)
        neg_18 = x1_18 = None
        mul_59 = cat_20 * sin_15
        cat_20 = None
        q_embed_9 = mul_58 + mul_59
        mul_58 = mul_59 = None
        mul_60 = key_states_9 * cos_15
        cos_15 = None
        x1_19 = key_states_9[(Ellipsis, slice(None, 32, None))]
        x2_19 = key_states_9[(Ellipsis, slice(32, None, None))]
        key_states_9 = None
        neg_19 = -x2_19
        x2_19 = None
        cat_21 = torch.cat((neg_19, x1_19), dim=-1)
        neg_19 = x1_19 = None
        mul_61 = cat_21 * sin_15
        cat_21 = sin_15 = None
        k_embed_9 = mul_60 + mul_61
        mul_60 = mul_61 = None
        transpose_50 = k_embed_9.transpose(2, 3)
        matmul_20 = torch.matmul(q_embed_9, transpose_50)
        q_embed_9 = transpose_50 = None
        attn_weights_36 = matmul_20 * 0.125
        matmul_20 = None
        causal_mask_11 = mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        attn_weights_37 = attn_weights_36 + causal_mask_11
        attn_weights_36 = causal_mask_11 = None
        softmax_9 = torch.nn.functional.softmax(
            attn_weights_37, dim=-1, dtype=torch.float32
        )
        attn_weights_37 = None
        attn_weights_38 = softmax_9.to(torch.float32)
        softmax_9 = None
        attn_weights_39 = torch.nn.functional.dropout(
            attn_weights_38, p=0.0, training=False
        )
        attn_weights_38 = None
        attn_output_36 = torch.matmul(attn_weights_39, value_states_9)
        attn_weights_39 = None
        transpose_51 = attn_output_36.transpose(1, 2)
        attn_output_36 = None
        attn_output_37 = transpose_51.contiguous()
        transpose_51 = None
        reshape_9 = attn_output_37.reshape(1, 10, -1)
        attn_output_37 = None
        attn_output_38 = reshape_9.contiguous()
        reshape_9 = None
        linear_57 = torch._C._nn.linear(
            attn_output_38,
            l_self_modules_layers_modules_9_modules_attn_modules_wo_parameters_weight_,
            None,
        )
        attn_output_38 = (
            l_self_modules_layers_modules_9_modules_attn_modules_wo_parameters_weight_
        ) = None
        attn_output_39 = torch.nn.functional.dropout(linear_57, 0.0, False, False)
        linear_57 = None
        hidden_states_37 = hidden_states_35 + attn_output_39
        hidden_states_35 = attn_output_39 = None
        hidden_states_38 = torch.nn.functional.layer_norm(
            hidden_states_37,
            (768,),
            l_self_modules_layers_modules_9_modules_mlp_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_9_modules_mlp_norm_parameters_weight_ = None
        linear_58 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_layers_modules_9_modules_mlp_modules_wi_parameters_weight_,
            None,
        )
        hidden_states_38 = (
            l_self_modules_layers_modules_9_modules_mlp_modules_wi_parameters_weight_
        ) = None
        chunk_9 = linear_58.chunk(2, dim=-1)
        linear_58 = None
        input_10 = chunk_9[0]
        gate_9 = chunk_9[1]
        chunk_9 = None
        gelu_9 = torch._C._nn.gelu(input_10)
        input_10 = None
        mul_63 = gelu_9 * gate_9
        gelu_9 = gate_9 = None
        dropout_30 = torch.nn.functional.dropout(mul_63, 0.0, False, False)
        mul_63 = None
        mlp_output_9 = torch._C._nn.linear(
            dropout_30,
            l_self_modules_layers_modules_9_modules_mlp_modules_wo_parameters_weight_,
            None,
        )
        dropout_30 = (
            l_self_modules_layers_modules_9_modules_mlp_modules_wo_parameters_weight_
        ) = None
        hidden_states_39 = hidden_states_37 + mlp_output_9
        hidden_states_37 = mlp_output_9 = None
        hidden_states_40 = torch.nn.functional.layer_norm(
            hidden_states_39,
            (768,),
            l_self_modules_layers_modules_10_modules_attn_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_10_modules_attn_norm_parameters_weight_ = None
        linear_60 = torch._C._nn.linear(
            hidden_states_40,
            l_self_modules_layers_modules_10_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_10_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_30 = linear_60.view((1, 10, -1, 64))
        linear_60 = None
        query_states_10 = view_30.transpose(1, 2)
        view_30 = None
        linear_61 = torch._C._nn.linear(
            hidden_states_40,
            l_self_modules_layers_modules_10_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_10_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_31 = linear_61.view((1, 10, -1, 64))
        linear_61 = None
        key_states_10 = view_31.transpose(1, 2)
        view_31 = None
        linear_62 = torch._C._nn.linear(
            hidden_states_40,
            l_self_modules_layers_modules_10_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_40 = l_self_modules_layers_modules_10_modules_attn_modules_v_proj_parameters_weight_ = (None)
        view_32 = linear_62.view((1, 10, -1, 64))
        linear_62 = None
        value_states_10 = view_32.transpose(1, 2)
        view_32 = None
        cos_16 = cos_6.unsqueeze(1)
        sin_16 = sin_6.unsqueeze(1)
        mul_64 = query_states_10 * cos_16
        x1_20 = query_states_10[(Ellipsis, slice(None, 32, None))]
        x2_20 = query_states_10[(Ellipsis, slice(32, None, None))]
        query_states_10 = None
        neg_20 = -x2_20
        x2_20 = None
        cat_22 = torch.cat((neg_20, x1_20), dim=-1)
        neg_20 = x1_20 = None
        mul_65 = cat_22 * sin_16
        cat_22 = None
        q_embed_10 = mul_64 + mul_65
        mul_64 = mul_65 = None
        mul_66 = key_states_10 * cos_16
        cos_16 = None
        x1_21 = key_states_10[(Ellipsis, slice(None, 32, None))]
        x2_21 = key_states_10[(Ellipsis, slice(32, None, None))]
        key_states_10 = None
        neg_21 = -x2_21
        x2_21 = None
        cat_23 = torch.cat((neg_21, x1_21), dim=-1)
        neg_21 = x1_21 = None
        mul_67 = cat_23 * sin_16
        cat_23 = sin_16 = None
        k_embed_10 = mul_66 + mul_67
        mul_66 = mul_67 = None
        transpose_55 = k_embed_10.transpose(2, 3)
        matmul_22 = torch.matmul(q_embed_10, transpose_55)
        q_embed_10 = transpose_55 = None
        attn_weights_40 = matmul_22 * 0.125
        matmul_22 = None
        causal_mask_12 = mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        attn_weights_41 = attn_weights_40 + causal_mask_12
        attn_weights_40 = causal_mask_12 = None
        softmax_10 = torch.nn.functional.softmax(
            attn_weights_41, dim=-1, dtype=torch.float32
        )
        attn_weights_41 = None
        attn_weights_42 = softmax_10.to(torch.float32)
        softmax_10 = None
        attn_weights_43 = torch.nn.functional.dropout(
            attn_weights_42, p=0.0, training=False
        )
        attn_weights_42 = None
        attn_output_40 = torch.matmul(attn_weights_43, value_states_10)
        attn_weights_43 = None
        transpose_56 = attn_output_40.transpose(1, 2)
        attn_output_40 = None
        attn_output_41 = transpose_56.contiguous()
        transpose_56 = None
        reshape_10 = attn_output_41.reshape(1, 10, -1)
        attn_output_41 = None
        attn_output_42 = reshape_10.contiguous()
        reshape_10 = None
        linear_63 = torch._C._nn.linear(
            attn_output_42,
            l_self_modules_layers_modules_10_modules_attn_modules_wo_parameters_weight_,
            None,
        )
        attn_output_42 = (
            l_self_modules_layers_modules_10_modules_attn_modules_wo_parameters_weight_
        ) = None
        attn_output_43 = torch.nn.functional.dropout(linear_63, 0.0, False, False)
        linear_63 = None
        hidden_states_41 = hidden_states_39 + attn_output_43
        hidden_states_39 = attn_output_43 = None
        hidden_states_42 = torch.nn.functional.layer_norm(
            hidden_states_41,
            (768,),
            l_self_modules_layers_modules_10_modules_mlp_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_10_modules_mlp_norm_parameters_weight_ = None
        linear_64 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_layers_modules_10_modules_mlp_modules_wi_parameters_weight_,
            None,
        )
        hidden_states_42 = (
            l_self_modules_layers_modules_10_modules_mlp_modules_wi_parameters_weight_
        ) = None
        chunk_10 = linear_64.chunk(2, dim=-1)
        linear_64 = None
        input_11 = chunk_10[0]
        gate_10 = chunk_10[1]
        chunk_10 = None
        gelu_10 = torch._C._nn.gelu(input_11)
        input_11 = None
        mul_69 = gelu_10 * gate_10
        gelu_10 = gate_10 = None
        dropout_33 = torch.nn.functional.dropout(mul_69, 0.0, False, False)
        mul_69 = None
        mlp_output_10 = torch._C._nn.linear(
            dropout_33,
            l_self_modules_layers_modules_10_modules_mlp_modules_wo_parameters_weight_,
            None,
        )
        dropout_33 = (
            l_self_modules_layers_modules_10_modules_mlp_modules_wo_parameters_weight_
        ) = None
        hidden_states_43 = hidden_states_41 + mlp_output_10
        hidden_states_41 = mlp_output_10 = None
        hidden_states_44 = torch.nn.functional.layer_norm(
            hidden_states_43,
            (768,),
            l_self_modules_layers_modules_11_modules_attn_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_11_modules_attn_norm_parameters_weight_ = None
        linear_66 = torch._C._nn.linear(
            hidden_states_44,
            l_self_modules_layers_modules_11_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_11_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_33 = linear_66.view((1, 10, -1, 64))
        linear_66 = None
        query_states_11 = view_33.transpose(1, 2)
        view_33 = None
        linear_67 = torch._C._nn.linear(
            hidden_states_44,
            l_self_modules_layers_modules_11_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_11_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_34 = linear_67.view((1, 10, -1, 64))
        linear_67 = None
        key_states_11 = view_34.transpose(1, 2)
        view_34 = None
        linear_68 = torch._C._nn.linear(
            hidden_states_44,
            l_self_modules_layers_modules_11_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_44 = l_self_modules_layers_modules_11_modules_attn_modules_v_proj_parameters_weight_ = (None)
        view_35 = linear_68.view((1, 10, -1, 64))
        linear_68 = None
        value_states_11 = view_35.transpose(1, 2)
        view_35 = None
        cos_17 = cos_6.unsqueeze(1)
        sin_17 = sin_6.unsqueeze(1)
        mul_70 = query_states_11 * cos_17
        x1_22 = query_states_11[(Ellipsis, slice(None, 32, None))]
        x2_22 = query_states_11[(Ellipsis, slice(32, None, None))]
        query_states_11 = None
        neg_22 = -x2_22
        x2_22 = None
        cat_24 = torch.cat((neg_22, x1_22), dim=-1)
        neg_22 = x1_22 = None
        mul_71 = cat_24 * sin_17
        cat_24 = None
        q_embed_11 = mul_70 + mul_71
        mul_70 = mul_71 = None
        mul_72 = key_states_11 * cos_17
        cos_17 = None
        x1_23 = key_states_11[(Ellipsis, slice(None, 32, None))]
        x2_23 = key_states_11[(Ellipsis, slice(32, None, None))]
        key_states_11 = None
        neg_23 = -x2_23
        x2_23 = None
        cat_25 = torch.cat((neg_23, x1_23), dim=-1)
        neg_23 = x1_23 = None
        mul_73 = cat_25 * sin_17
        cat_25 = sin_17 = None
        k_embed_11 = mul_72 + mul_73
        mul_72 = mul_73 = None
        transpose_60 = k_embed_11.transpose(2, 3)
        matmul_24 = torch.matmul(q_embed_11, transpose_60)
        q_embed_11 = transpose_60 = None
        attn_weights_44 = matmul_24 * 0.125
        matmul_24 = None
        causal_mask_13 = mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        attn_weights_45 = attn_weights_44 + causal_mask_13
        attn_weights_44 = causal_mask_13 = None
        softmax_11 = torch.nn.functional.softmax(
            attn_weights_45, dim=-1, dtype=torch.float32
        )
        attn_weights_45 = None
        attn_weights_46 = softmax_11.to(torch.float32)
        softmax_11 = None
        attn_weights_47 = torch.nn.functional.dropout(
            attn_weights_46, p=0.0, training=False
        )
        attn_weights_46 = None
        attn_output_44 = torch.matmul(attn_weights_47, value_states_11)
        attn_weights_47 = None
        transpose_61 = attn_output_44.transpose(1, 2)
        attn_output_44 = None
        attn_output_45 = transpose_61.contiguous()
        transpose_61 = None
        reshape_11 = attn_output_45.reshape(1, 10, -1)
        attn_output_45 = None
        attn_output_46 = reshape_11.contiguous()
        reshape_11 = None
        linear_69 = torch._C._nn.linear(
            attn_output_46,
            l_self_modules_layers_modules_11_modules_attn_modules_wo_parameters_weight_,
            None,
        )
        attn_output_46 = (
            l_self_modules_layers_modules_11_modules_attn_modules_wo_parameters_weight_
        ) = None
        attn_output_47 = torch.nn.functional.dropout(linear_69, 0.0, False, False)
        linear_69 = None
        hidden_states_45 = hidden_states_43 + attn_output_47
        hidden_states_43 = attn_output_47 = None
        hidden_states_46 = torch.nn.functional.layer_norm(
            hidden_states_45,
            (768,),
            l_self_modules_layers_modules_11_modules_mlp_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_11_modules_mlp_norm_parameters_weight_ = None
        linear_70 = torch._C._nn.linear(
            hidden_states_46,
            l_self_modules_layers_modules_11_modules_mlp_modules_wi_parameters_weight_,
            None,
        )
        hidden_states_46 = (
            l_self_modules_layers_modules_11_modules_mlp_modules_wi_parameters_weight_
        ) = None
        chunk_11 = linear_70.chunk(2, dim=-1)
        linear_70 = None
        input_12 = chunk_11[0]
        gate_11 = chunk_11[1]
        chunk_11 = None
        gelu_11 = torch._C._nn.gelu(input_12)
        input_12 = None
        mul_75 = gelu_11 * gate_11
        gelu_11 = gate_11 = None
        dropout_36 = torch.nn.functional.dropout(mul_75, 0.0, False, False)
        mul_75 = None
        mlp_output_11 = torch._C._nn.linear(
            dropout_36,
            l_self_modules_layers_modules_11_modules_mlp_modules_wo_parameters_weight_,
            None,
        )
        dropout_36 = (
            l_self_modules_layers_modules_11_modules_mlp_modules_wo_parameters_weight_
        ) = None
        hidden_states_47 = hidden_states_45 + mlp_output_11
        hidden_states_45 = mlp_output_11 = None
        hidden_states_48 = torch.nn.functional.layer_norm(
            hidden_states_47,
            (768,),
            l_self_modules_layers_modules_12_modules_attn_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_12_modules_attn_norm_parameters_weight_ = None
        linear_72 = torch._C._nn.linear(
            hidden_states_48,
            l_self_modules_layers_modules_12_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_12_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_36 = linear_72.view((1, 10, -1, 64))
        linear_72 = None
        query_states_12 = view_36.transpose(1, 2)
        view_36 = None
        linear_73 = torch._C._nn.linear(
            hidden_states_48,
            l_self_modules_layers_modules_12_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_12_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_37 = linear_73.view((1, 10, -1, 64))
        linear_73 = None
        key_states_12 = view_37.transpose(1, 2)
        view_37 = None
        linear_74 = torch._C._nn.linear(
            hidden_states_48,
            l_self_modules_layers_modules_12_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_48 = l_self_modules_layers_modules_12_modules_attn_modules_v_proj_parameters_weight_ = (None)
        view_38 = linear_74.view((1, 10, -1, 64))
        linear_74 = None
        value_states_12 = view_38.transpose(1, 2)
        view_38 = None
        cos_18 = cos_4.unsqueeze(1)
        sin_18 = sin_4.unsqueeze(1)
        mul_76 = query_states_12 * cos_18
        x1_24 = query_states_12[(Ellipsis, slice(None, 32, None))]
        x2_24 = query_states_12[(Ellipsis, slice(32, None, None))]
        query_states_12 = None
        neg_24 = -x2_24
        x2_24 = None
        cat_26 = torch.cat((neg_24, x1_24), dim=-1)
        neg_24 = x1_24 = None
        mul_77 = cat_26 * sin_18
        cat_26 = None
        q_embed_12 = mul_76 + mul_77
        mul_76 = mul_77 = None
        mul_78 = key_states_12 * cos_18
        cos_18 = None
        x1_25 = key_states_12[(Ellipsis, slice(None, 32, None))]
        x2_25 = key_states_12[(Ellipsis, slice(32, None, None))]
        key_states_12 = None
        neg_25 = -x2_25
        x2_25 = None
        cat_27 = torch.cat((neg_25, x1_25), dim=-1)
        neg_25 = x1_25 = None
        mul_79 = cat_27 * sin_18
        cat_27 = sin_18 = None
        k_embed_12 = mul_78 + mul_79
        mul_78 = mul_79 = None
        transpose_65 = k_embed_12.transpose(2, 3)
        matmul_26 = torch.matmul(q_embed_12, transpose_65)
        q_embed_12 = transpose_65 = None
        attn_weights_48 = matmul_26 * 0.125
        matmul_26 = None
        causal_mask_14 = mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        attn_weights_49 = attn_weights_48 + causal_mask_14
        attn_weights_48 = causal_mask_14 = None
        softmax_12 = torch.nn.functional.softmax(
            attn_weights_49, dim=-1, dtype=torch.float32
        )
        attn_weights_49 = None
        attn_weights_50 = softmax_12.to(torch.float32)
        softmax_12 = None
        attn_weights_51 = torch.nn.functional.dropout(
            attn_weights_50, p=0.0, training=False
        )
        attn_weights_50 = None
        attn_output_48 = torch.matmul(attn_weights_51, value_states_12)
        attn_weights_51 = None
        transpose_66 = attn_output_48.transpose(1, 2)
        attn_output_48 = None
        attn_output_49 = transpose_66.contiguous()
        transpose_66 = None
        reshape_12 = attn_output_49.reshape(1, 10, -1)
        attn_output_49 = None
        attn_output_50 = reshape_12.contiguous()
        reshape_12 = None
        linear_75 = torch._C._nn.linear(
            attn_output_50,
            l_self_modules_layers_modules_12_modules_attn_modules_wo_parameters_weight_,
            None,
        )
        attn_output_50 = (
            l_self_modules_layers_modules_12_modules_attn_modules_wo_parameters_weight_
        ) = None
        attn_output_51 = torch.nn.functional.dropout(linear_75, 0.0, False, False)
        linear_75 = None
        hidden_states_49 = hidden_states_47 + attn_output_51
        hidden_states_47 = attn_output_51 = None
        hidden_states_50 = torch.nn.functional.layer_norm(
            hidden_states_49,
            (768,),
            l_self_modules_layers_modules_12_modules_mlp_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_12_modules_mlp_norm_parameters_weight_ = None
        linear_76 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_layers_modules_12_modules_mlp_modules_wi_parameters_weight_,
            None,
        )
        hidden_states_50 = (
            l_self_modules_layers_modules_12_modules_mlp_modules_wi_parameters_weight_
        ) = None
        chunk_12 = linear_76.chunk(2, dim=-1)
        linear_76 = None
        input_13 = chunk_12[0]
        gate_12 = chunk_12[1]
        chunk_12 = None
        gelu_12 = torch._C._nn.gelu(input_13)
        input_13 = None
        mul_81 = gelu_12 * gate_12
        gelu_12 = gate_12 = None
        dropout_39 = torch.nn.functional.dropout(mul_81, 0.0, False, False)
        mul_81 = None
        mlp_output_12 = torch._C._nn.linear(
            dropout_39,
            l_self_modules_layers_modules_12_modules_mlp_modules_wo_parameters_weight_,
            None,
        )
        dropout_39 = (
            l_self_modules_layers_modules_12_modules_mlp_modules_wo_parameters_weight_
        ) = None
        hidden_states_51 = hidden_states_49 + mlp_output_12
        hidden_states_49 = mlp_output_12 = None
        hidden_states_52 = torch.nn.functional.layer_norm(
            hidden_states_51,
            (768,),
            l_self_modules_layers_modules_13_modules_attn_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_13_modules_attn_norm_parameters_weight_ = None
        linear_78 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_layers_modules_13_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_13_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_39 = linear_78.view((1, 10, -1, 64))
        linear_78 = None
        query_states_13 = view_39.transpose(1, 2)
        view_39 = None
        linear_79 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_layers_modules_13_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_13_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_40 = linear_79.view((1, 10, -1, 64))
        linear_79 = None
        key_states_13 = view_40.transpose(1, 2)
        view_40 = None
        linear_80 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_layers_modules_13_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_52 = l_self_modules_layers_modules_13_modules_attn_modules_v_proj_parameters_weight_ = (None)
        view_41 = linear_80.view((1, 10, -1, 64))
        linear_80 = None
        value_states_13 = view_41.transpose(1, 2)
        view_41 = None
        cos_19 = cos_6.unsqueeze(1)
        sin_19 = sin_6.unsqueeze(1)
        mul_82 = query_states_13 * cos_19
        x1_26 = query_states_13[(Ellipsis, slice(None, 32, None))]
        x2_26 = query_states_13[(Ellipsis, slice(32, None, None))]
        query_states_13 = None
        neg_26 = -x2_26
        x2_26 = None
        cat_28 = torch.cat((neg_26, x1_26), dim=-1)
        neg_26 = x1_26 = None
        mul_83 = cat_28 * sin_19
        cat_28 = None
        q_embed_13 = mul_82 + mul_83
        mul_82 = mul_83 = None
        mul_84 = key_states_13 * cos_19
        cos_19 = None
        x1_27 = key_states_13[(Ellipsis, slice(None, 32, None))]
        x2_27 = key_states_13[(Ellipsis, slice(32, None, None))]
        key_states_13 = None
        neg_27 = -x2_27
        x2_27 = None
        cat_29 = torch.cat((neg_27, x1_27), dim=-1)
        neg_27 = x1_27 = None
        mul_85 = cat_29 * sin_19
        cat_29 = sin_19 = None
        k_embed_13 = mul_84 + mul_85
        mul_84 = mul_85 = None
        transpose_70 = k_embed_13.transpose(2, 3)
        matmul_28 = torch.matmul(q_embed_13, transpose_70)
        q_embed_13 = transpose_70 = None
        attn_weights_52 = matmul_28 * 0.125
        matmul_28 = None
        causal_mask_15 = mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        attn_weights_53 = attn_weights_52 + causal_mask_15
        attn_weights_52 = causal_mask_15 = None
        softmax_13 = torch.nn.functional.softmax(
            attn_weights_53, dim=-1, dtype=torch.float32
        )
        attn_weights_53 = None
        attn_weights_54 = softmax_13.to(torch.float32)
        softmax_13 = None
        attn_weights_55 = torch.nn.functional.dropout(
            attn_weights_54, p=0.0, training=False
        )
        attn_weights_54 = None
        attn_output_52 = torch.matmul(attn_weights_55, value_states_13)
        attn_weights_55 = None
        transpose_71 = attn_output_52.transpose(1, 2)
        attn_output_52 = None
        attn_output_53 = transpose_71.contiguous()
        transpose_71 = None
        reshape_13 = attn_output_53.reshape(1, 10, -1)
        attn_output_53 = None
        attn_output_54 = reshape_13.contiguous()
        reshape_13 = None
        linear_81 = torch._C._nn.linear(
            attn_output_54,
            l_self_modules_layers_modules_13_modules_attn_modules_wo_parameters_weight_,
            None,
        )
        attn_output_54 = (
            l_self_modules_layers_modules_13_modules_attn_modules_wo_parameters_weight_
        ) = None
        attn_output_55 = torch.nn.functional.dropout(linear_81, 0.0, False, False)
        linear_81 = None
        hidden_states_53 = hidden_states_51 + attn_output_55
        hidden_states_51 = attn_output_55 = None
        hidden_states_54 = torch.nn.functional.layer_norm(
            hidden_states_53,
            (768,),
            l_self_modules_layers_modules_13_modules_mlp_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_13_modules_mlp_norm_parameters_weight_ = None
        linear_82 = torch._C._nn.linear(
            hidden_states_54,
            l_self_modules_layers_modules_13_modules_mlp_modules_wi_parameters_weight_,
            None,
        )
        hidden_states_54 = (
            l_self_modules_layers_modules_13_modules_mlp_modules_wi_parameters_weight_
        ) = None
        chunk_13 = linear_82.chunk(2, dim=-1)
        linear_82 = None
        input_14 = chunk_13[0]
        gate_13 = chunk_13[1]
        chunk_13 = None
        gelu_13 = torch._C._nn.gelu(input_14)
        input_14 = None
        mul_87 = gelu_13 * gate_13
        gelu_13 = gate_13 = None
        dropout_42 = torch.nn.functional.dropout(mul_87, 0.0, False, False)
        mul_87 = None
        mlp_output_13 = torch._C._nn.linear(
            dropout_42,
            l_self_modules_layers_modules_13_modules_mlp_modules_wo_parameters_weight_,
            None,
        )
        dropout_42 = (
            l_self_modules_layers_modules_13_modules_mlp_modules_wo_parameters_weight_
        ) = None
        hidden_states_55 = hidden_states_53 + mlp_output_13
        hidden_states_53 = mlp_output_13 = None
        hidden_states_56 = torch.nn.functional.layer_norm(
            hidden_states_55,
            (768,),
            l_self_modules_layers_modules_14_modules_attn_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_14_modules_attn_norm_parameters_weight_ = None
        linear_84 = torch._C._nn.linear(
            hidden_states_56,
            l_self_modules_layers_modules_14_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_14_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_42 = linear_84.view((1, 10, -1, 64))
        linear_84 = None
        query_states_14 = view_42.transpose(1, 2)
        view_42 = None
        linear_85 = torch._C._nn.linear(
            hidden_states_56,
            l_self_modules_layers_modules_14_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_14_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_43 = linear_85.view((1, 10, -1, 64))
        linear_85 = None
        key_states_14 = view_43.transpose(1, 2)
        view_43 = None
        linear_86 = torch._C._nn.linear(
            hidden_states_56,
            l_self_modules_layers_modules_14_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_56 = l_self_modules_layers_modules_14_modules_attn_modules_v_proj_parameters_weight_ = (None)
        view_44 = linear_86.view((1, 10, -1, 64))
        linear_86 = None
        value_states_14 = view_44.transpose(1, 2)
        view_44 = None
        cos_20 = cos_6.unsqueeze(1)
        sin_20 = sin_6.unsqueeze(1)
        mul_88 = query_states_14 * cos_20
        x1_28 = query_states_14[(Ellipsis, slice(None, 32, None))]
        x2_28 = query_states_14[(Ellipsis, slice(32, None, None))]
        query_states_14 = None
        neg_28 = -x2_28
        x2_28 = None
        cat_30 = torch.cat((neg_28, x1_28), dim=-1)
        neg_28 = x1_28 = None
        mul_89 = cat_30 * sin_20
        cat_30 = None
        q_embed_14 = mul_88 + mul_89
        mul_88 = mul_89 = None
        mul_90 = key_states_14 * cos_20
        cos_20 = None
        x1_29 = key_states_14[(Ellipsis, slice(None, 32, None))]
        x2_29 = key_states_14[(Ellipsis, slice(32, None, None))]
        key_states_14 = None
        neg_29 = -x2_29
        x2_29 = None
        cat_31 = torch.cat((neg_29, x1_29), dim=-1)
        neg_29 = x1_29 = None
        mul_91 = cat_31 * sin_20
        cat_31 = sin_20 = None
        k_embed_14 = mul_90 + mul_91
        mul_90 = mul_91 = None
        transpose_75 = k_embed_14.transpose(2, 3)
        matmul_30 = torch.matmul(q_embed_14, transpose_75)
        q_embed_14 = transpose_75 = None
        attn_weights_56 = matmul_30 * 0.125
        matmul_30 = None
        causal_mask_16 = mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        attn_weights_57 = attn_weights_56 + causal_mask_16
        attn_weights_56 = causal_mask_16 = None
        softmax_14 = torch.nn.functional.softmax(
            attn_weights_57, dim=-1, dtype=torch.float32
        )
        attn_weights_57 = None
        attn_weights_58 = softmax_14.to(torch.float32)
        softmax_14 = None
        attn_weights_59 = torch.nn.functional.dropout(
            attn_weights_58, p=0.0, training=False
        )
        attn_weights_58 = None
        attn_output_56 = torch.matmul(attn_weights_59, value_states_14)
        attn_weights_59 = None
        transpose_76 = attn_output_56.transpose(1, 2)
        attn_output_56 = None
        attn_output_57 = transpose_76.contiguous()
        transpose_76 = None
        reshape_14 = attn_output_57.reshape(1, 10, -1)
        attn_output_57 = None
        attn_output_58 = reshape_14.contiguous()
        reshape_14 = None
        linear_87 = torch._C._nn.linear(
            attn_output_58,
            l_self_modules_layers_modules_14_modules_attn_modules_wo_parameters_weight_,
            None,
        )
        attn_output_58 = (
            l_self_modules_layers_modules_14_modules_attn_modules_wo_parameters_weight_
        ) = None
        attn_output_59 = torch.nn.functional.dropout(linear_87, 0.0, False, False)
        linear_87 = None
        hidden_states_57 = hidden_states_55 + attn_output_59
        hidden_states_55 = attn_output_59 = None
        hidden_states_58 = torch.nn.functional.layer_norm(
            hidden_states_57,
            (768,),
            l_self_modules_layers_modules_14_modules_mlp_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_14_modules_mlp_norm_parameters_weight_ = None
        linear_88 = torch._C._nn.linear(
            hidden_states_58,
            l_self_modules_layers_modules_14_modules_mlp_modules_wi_parameters_weight_,
            None,
        )
        hidden_states_58 = (
            l_self_modules_layers_modules_14_modules_mlp_modules_wi_parameters_weight_
        ) = None
        chunk_14 = linear_88.chunk(2, dim=-1)
        linear_88 = None
        input_15 = chunk_14[0]
        gate_14 = chunk_14[1]
        chunk_14 = None
        gelu_14 = torch._C._nn.gelu(input_15)
        input_15 = None
        mul_93 = gelu_14 * gate_14
        gelu_14 = gate_14 = None
        dropout_45 = torch.nn.functional.dropout(mul_93, 0.0, False, False)
        mul_93 = None
        mlp_output_14 = torch._C._nn.linear(
            dropout_45,
            l_self_modules_layers_modules_14_modules_mlp_modules_wo_parameters_weight_,
            None,
        )
        dropout_45 = (
            l_self_modules_layers_modules_14_modules_mlp_modules_wo_parameters_weight_
        ) = None
        hidden_states_59 = hidden_states_57 + mlp_output_14
        hidden_states_57 = mlp_output_14 = None
        hidden_states_60 = torch.nn.functional.layer_norm(
            hidden_states_59,
            (768,),
            l_self_modules_layers_modules_15_modules_attn_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_15_modules_attn_norm_parameters_weight_ = None
        linear_90 = torch._C._nn.linear(
            hidden_states_60,
            l_self_modules_layers_modules_15_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_15_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_45 = linear_90.view((1, 10, -1, 64))
        linear_90 = None
        query_states_15 = view_45.transpose(1, 2)
        view_45 = None
        linear_91 = torch._C._nn.linear(
            hidden_states_60,
            l_self_modules_layers_modules_15_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_15_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_46 = linear_91.view((1, 10, -1, 64))
        linear_91 = None
        key_states_15 = view_46.transpose(1, 2)
        view_46 = None
        linear_92 = torch._C._nn.linear(
            hidden_states_60,
            l_self_modules_layers_modules_15_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_60 = l_self_modules_layers_modules_15_modules_attn_modules_v_proj_parameters_weight_ = (None)
        view_47 = linear_92.view((1, 10, -1, 64))
        linear_92 = None
        value_states_15 = view_47.transpose(1, 2)
        view_47 = None
        cos_21 = cos_4.unsqueeze(1)
        sin_21 = sin_4.unsqueeze(1)
        mul_94 = query_states_15 * cos_21
        x1_30 = query_states_15[(Ellipsis, slice(None, 32, None))]
        x2_30 = query_states_15[(Ellipsis, slice(32, None, None))]
        query_states_15 = None
        neg_30 = -x2_30
        x2_30 = None
        cat_32 = torch.cat((neg_30, x1_30), dim=-1)
        neg_30 = x1_30 = None
        mul_95 = cat_32 * sin_21
        cat_32 = None
        q_embed_15 = mul_94 + mul_95
        mul_94 = mul_95 = None
        mul_96 = key_states_15 * cos_21
        cos_21 = None
        x1_31 = key_states_15[(Ellipsis, slice(None, 32, None))]
        x2_31 = key_states_15[(Ellipsis, slice(32, None, None))]
        key_states_15 = None
        neg_31 = -x2_31
        x2_31 = None
        cat_33 = torch.cat((neg_31, x1_31), dim=-1)
        neg_31 = x1_31 = None
        mul_97 = cat_33 * sin_21
        cat_33 = sin_21 = None
        k_embed_15 = mul_96 + mul_97
        mul_96 = mul_97 = None
        transpose_80 = k_embed_15.transpose(2, 3)
        matmul_32 = torch.matmul(q_embed_15, transpose_80)
        q_embed_15 = transpose_80 = None
        attn_weights_60 = matmul_32 * 0.125
        matmul_32 = None
        causal_mask_17 = mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        attn_weights_61 = attn_weights_60 + causal_mask_17
        attn_weights_60 = causal_mask_17 = None
        softmax_15 = torch.nn.functional.softmax(
            attn_weights_61, dim=-1, dtype=torch.float32
        )
        attn_weights_61 = None
        attn_weights_62 = softmax_15.to(torch.float32)
        softmax_15 = None
        attn_weights_63 = torch.nn.functional.dropout(
            attn_weights_62, p=0.0, training=False
        )
        attn_weights_62 = None
        attn_output_60 = torch.matmul(attn_weights_63, value_states_15)
        attn_weights_63 = None
        transpose_81 = attn_output_60.transpose(1, 2)
        attn_output_60 = None
        attn_output_61 = transpose_81.contiguous()
        transpose_81 = None
        reshape_15 = attn_output_61.reshape(1, 10, -1)
        attn_output_61 = None
        attn_output_62 = reshape_15.contiguous()
        reshape_15 = None
        linear_93 = torch._C._nn.linear(
            attn_output_62,
            l_self_modules_layers_modules_15_modules_attn_modules_wo_parameters_weight_,
            None,
        )
        attn_output_62 = (
            l_self_modules_layers_modules_15_modules_attn_modules_wo_parameters_weight_
        ) = None
        attn_output_63 = torch.nn.functional.dropout(linear_93, 0.0, False, False)
        linear_93 = None
        hidden_states_61 = hidden_states_59 + attn_output_63
        hidden_states_59 = attn_output_63 = None
        hidden_states_62 = torch.nn.functional.layer_norm(
            hidden_states_61,
            (768,),
            l_self_modules_layers_modules_15_modules_mlp_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_15_modules_mlp_norm_parameters_weight_ = None
        linear_94 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_layers_modules_15_modules_mlp_modules_wi_parameters_weight_,
            None,
        )
        hidden_states_62 = (
            l_self_modules_layers_modules_15_modules_mlp_modules_wi_parameters_weight_
        ) = None
        chunk_15 = linear_94.chunk(2, dim=-1)
        linear_94 = None
        input_16 = chunk_15[0]
        gate_15 = chunk_15[1]
        chunk_15 = None
        gelu_15 = torch._C._nn.gelu(input_16)
        input_16 = None
        mul_99 = gelu_15 * gate_15
        gelu_15 = gate_15 = None
        dropout_48 = torch.nn.functional.dropout(mul_99, 0.0, False, False)
        mul_99 = None
        mlp_output_15 = torch._C._nn.linear(
            dropout_48,
            l_self_modules_layers_modules_15_modules_mlp_modules_wo_parameters_weight_,
            None,
        )
        dropout_48 = (
            l_self_modules_layers_modules_15_modules_mlp_modules_wo_parameters_weight_
        ) = None
        hidden_states_63 = hidden_states_61 + mlp_output_15
        hidden_states_61 = mlp_output_15 = None
        hidden_states_64 = torch.nn.functional.layer_norm(
            hidden_states_63,
            (768,),
            l_self_modules_layers_modules_16_modules_attn_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_16_modules_attn_norm_parameters_weight_ = None
        linear_96 = torch._C._nn.linear(
            hidden_states_64,
            l_self_modules_layers_modules_16_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_16_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_48 = linear_96.view((1, 10, -1, 64))
        linear_96 = None
        query_states_16 = view_48.transpose(1, 2)
        view_48 = None
        linear_97 = torch._C._nn.linear(
            hidden_states_64,
            l_self_modules_layers_modules_16_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_16_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_49 = linear_97.view((1, 10, -1, 64))
        linear_97 = None
        key_states_16 = view_49.transpose(1, 2)
        view_49 = None
        linear_98 = torch._C._nn.linear(
            hidden_states_64,
            l_self_modules_layers_modules_16_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_64 = l_self_modules_layers_modules_16_modules_attn_modules_v_proj_parameters_weight_ = (None)
        view_50 = linear_98.view((1, 10, -1, 64))
        linear_98 = None
        value_states_16 = view_50.transpose(1, 2)
        view_50 = None
        cos_22 = cos_6.unsqueeze(1)
        sin_22 = sin_6.unsqueeze(1)
        mul_100 = query_states_16 * cos_22
        x1_32 = query_states_16[(Ellipsis, slice(None, 32, None))]
        x2_32 = query_states_16[(Ellipsis, slice(32, None, None))]
        query_states_16 = None
        neg_32 = -x2_32
        x2_32 = None
        cat_34 = torch.cat((neg_32, x1_32), dim=-1)
        neg_32 = x1_32 = None
        mul_101 = cat_34 * sin_22
        cat_34 = None
        q_embed_16 = mul_100 + mul_101
        mul_100 = mul_101 = None
        mul_102 = key_states_16 * cos_22
        cos_22 = None
        x1_33 = key_states_16[(Ellipsis, slice(None, 32, None))]
        x2_33 = key_states_16[(Ellipsis, slice(32, None, None))]
        key_states_16 = None
        neg_33 = -x2_33
        x2_33 = None
        cat_35 = torch.cat((neg_33, x1_33), dim=-1)
        neg_33 = x1_33 = None
        mul_103 = cat_35 * sin_22
        cat_35 = sin_22 = None
        k_embed_16 = mul_102 + mul_103
        mul_102 = mul_103 = None
        transpose_85 = k_embed_16.transpose(2, 3)
        matmul_34 = torch.matmul(q_embed_16, transpose_85)
        q_embed_16 = transpose_85 = None
        attn_weights_64 = matmul_34 * 0.125
        matmul_34 = None
        causal_mask_18 = mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        attn_weights_65 = attn_weights_64 + causal_mask_18
        attn_weights_64 = causal_mask_18 = None
        softmax_16 = torch.nn.functional.softmax(
            attn_weights_65, dim=-1, dtype=torch.float32
        )
        attn_weights_65 = None
        attn_weights_66 = softmax_16.to(torch.float32)
        softmax_16 = None
        attn_weights_67 = torch.nn.functional.dropout(
            attn_weights_66, p=0.0, training=False
        )
        attn_weights_66 = None
        attn_output_64 = torch.matmul(attn_weights_67, value_states_16)
        attn_weights_67 = None
        transpose_86 = attn_output_64.transpose(1, 2)
        attn_output_64 = None
        attn_output_65 = transpose_86.contiguous()
        transpose_86 = None
        reshape_16 = attn_output_65.reshape(1, 10, -1)
        attn_output_65 = None
        attn_output_66 = reshape_16.contiguous()
        reshape_16 = None
        linear_99 = torch._C._nn.linear(
            attn_output_66,
            l_self_modules_layers_modules_16_modules_attn_modules_wo_parameters_weight_,
            None,
        )
        attn_output_66 = (
            l_self_modules_layers_modules_16_modules_attn_modules_wo_parameters_weight_
        ) = None
        attn_output_67 = torch.nn.functional.dropout(linear_99, 0.0, False, False)
        linear_99 = None
        hidden_states_65 = hidden_states_63 + attn_output_67
        hidden_states_63 = attn_output_67 = None
        hidden_states_66 = torch.nn.functional.layer_norm(
            hidden_states_65,
            (768,),
            l_self_modules_layers_modules_16_modules_mlp_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_16_modules_mlp_norm_parameters_weight_ = None
        linear_100 = torch._C._nn.linear(
            hidden_states_66,
            l_self_modules_layers_modules_16_modules_mlp_modules_wi_parameters_weight_,
            None,
        )
        hidden_states_66 = (
            l_self_modules_layers_modules_16_modules_mlp_modules_wi_parameters_weight_
        ) = None
        chunk_16 = linear_100.chunk(2, dim=-1)
        linear_100 = None
        input_17 = chunk_16[0]
        gate_16 = chunk_16[1]
        chunk_16 = None
        gelu_16 = torch._C._nn.gelu(input_17)
        input_17 = None
        mul_105 = gelu_16 * gate_16
        gelu_16 = gate_16 = None
        dropout_51 = torch.nn.functional.dropout(mul_105, 0.0, False, False)
        mul_105 = None
        mlp_output_16 = torch._C._nn.linear(
            dropout_51,
            l_self_modules_layers_modules_16_modules_mlp_modules_wo_parameters_weight_,
            None,
        )
        dropout_51 = (
            l_self_modules_layers_modules_16_modules_mlp_modules_wo_parameters_weight_
        ) = None
        hidden_states_67 = hidden_states_65 + mlp_output_16
        hidden_states_65 = mlp_output_16 = None
        hidden_states_68 = torch.nn.functional.layer_norm(
            hidden_states_67,
            (768,),
            l_self_modules_layers_modules_17_modules_attn_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_17_modules_attn_norm_parameters_weight_ = None
        linear_102 = torch._C._nn.linear(
            hidden_states_68,
            l_self_modules_layers_modules_17_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_17_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_51 = linear_102.view((1, 10, -1, 64))
        linear_102 = None
        query_states_17 = view_51.transpose(1, 2)
        view_51 = None
        linear_103 = torch._C._nn.linear(
            hidden_states_68,
            l_self_modules_layers_modules_17_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_17_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_52 = linear_103.view((1, 10, -1, 64))
        linear_103 = None
        key_states_17 = view_52.transpose(1, 2)
        view_52 = None
        linear_104 = torch._C._nn.linear(
            hidden_states_68,
            l_self_modules_layers_modules_17_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_68 = l_self_modules_layers_modules_17_modules_attn_modules_v_proj_parameters_weight_ = (None)
        view_53 = linear_104.view((1, 10, -1, 64))
        linear_104 = None
        value_states_17 = view_53.transpose(1, 2)
        view_53 = None
        cos_23 = cos_6.unsqueeze(1)
        sin_23 = sin_6.unsqueeze(1)
        mul_106 = query_states_17 * cos_23
        x1_34 = query_states_17[(Ellipsis, slice(None, 32, None))]
        x2_34 = query_states_17[(Ellipsis, slice(32, None, None))]
        query_states_17 = None
        neg_34 = -x2_34
        x2_34 = None
        cat_36 = torch.cat((neg_34, x1_34), dim=-1)
        neg_34 = x1_34 = None
        mul_107 = cat_36 * sin_23
        cat_36 = None
        q_embed_17 = mul_106 + mul_107
        mul_106 = mul_107 = None
        mul_108 = key_states_17 * cos_23
        cos_23 = None
        x1_35 = key_states_17[(Ellipsis, slice(None, 32, None))]
        x2_35 = key_states_17[(Ellipsis, slice(32, None, None))]
        key_states_17 = None
        neg_35 = -x2_35
        x2_35 = None
        cat_37 = torch.cat((neg_35, x1_35), dim=-1)
        neg_35 = x1_35 = None
        mul_109 = cat_37 * sin_23
        cat_37 = sin_23 = None
        k_embed_17 = mul_108 + mul_109
        mul_108 = mul_109 = None
        transpose_90 = k_embed_17.transpose(2, 3)
        matmul_36 = torch.matmul(q_embed_17, transpose_90)
        q_embed_17 = transpose_90 = None
        attn_weights_68 = matmul_36 * 0.125
        matmul_36 = None
        causal_mask_19 = mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        attn_weights_69 = attn_weights_68 + causal_mask_19
        attn_weights_68 = causal_mask_19 = None
        softmax_17 = torch.nn.functional.softmax(
            attn_weights_69, dim=-1, dtype=torch.float32
        )
        attn_weights_69 = None
        attn_weights_70 = softmax_17.to(torch.float32)
        softmax_17 = None
        attn_weights_71 = torch.nn.functional.dropout(
            attn_weights_70, p=0.0, training=False
        )
        attn_weights_70 = None
        attn_output_68 = torch.matmul(attn_weights_71, value_states_17)
        attn_weights_71 = None
        transpose_91 = attn_output_68.transpose(1, 2)
        attn_output_68 = None
        attn_output_69 = transpose_91.contiguous()
        transpose_91 = None
        reshape_17 = attn_output_69.reshape(1, 10, -1)
        attn_output_69 = None
        attn_output_70 = reshape_17.contiguous()
        reshape_17 = None
        linear_105 = torch._C._nn.linear(
            attn_output_70,
            l_self_modules_layers_modules_17_modules_attn_modules_wo_parameters_weight_,
            None,
        )
        attn_output_70 = (
            l_self_modules_layers_modules_17_modules_attn_modules_wo_parameters_weight_
        ) = None
        attn_output_71 = torch.nn.functional.dropout(linear_105, 0.0, False, False)
        linear_105 = None
        hidden_states_69 = hidden_states_67 + attn_output_71
        hidden_states_67 = attn_output_71 = None
        hidden_states_70 = torch.nn.functional.layer_norm(
            hidden_states_69,
            (768,),
            l_self_modules_layers_modules_17_modules_mlp_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_17_modules_mlp_norm_parameters_weight_ = None
        linear_106 = torch._C._nn.linear(
            hidden_states_70,
            l_self_modules_layers_modules_17_modules_mlp_modules_wi_parameters_weight_,
            None,
        )
        hidden_states_70 = (
            l_self_modules_layers_modules_17_modules_mlp_modules_wi_parameters_weight_
        ) = None
        chunk_17 = linear_106.chunk(2, dim=-1)
        linear_106 = None
        input_18 = chunk_17[0]
        gate_17 = chunk_17[1]
        chunk_17 = None
        gelu_17 = torch._C._nn.gelu(input_18)
        input_18 = None
        mul_111 = gelu_17 * gate_17
        gelu_17 = gate_17 = None
        dropout_54 = torch.nn.functional.dropout(mul_111, 0.0, False, False)
        mul_111 = None
        mlp_output_17 = torch._C._nn.linear(
            dropout_54,
            l_self_modules_layers_modules_17_modules_mlp_modules_wo_parameters_weight_,
            None,
        )
        dropout_54 = (
            l_self_modules_layers_modules_17_modules_mlp_modules_wo_parameters_weight_
        ) = None
        hidden_states_71 = hidden_states_69 + mlp_output_17
        hidden_states_69 = mlp_output_17 = None
        hidden_states_72 = torch.nn.functional.layer_norm(
            hidden_states_71,
            (768,),
            l_self_modules_layers_modules_18_modules_attn_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_18_modules_attn_norm_parameters_weight_ = None
        linear_108 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_layers_modules_18_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_18_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_54 = linear_108.view((1, 10, -1, 64))
        linear_108 = None
        query_states_18 = view_54.transpose(1, 2)
        view_54 = None
        linear_109 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_layers_modules_18_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_18_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_55 = linear_109.view((1, 10, -1, 64))
        linear_109 = None
        key_states_18 = view_55.transpose(1, 2)
        view_55 = None
        linear_110 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_layers_modules_18_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_72 = l_self_modules_layers_modules_18_modules_attn_modules_v_proj_parameters_weight_ = (None)
        view_56 = linear_110.view((1, 10, -1, 64))
        linear_110 = None
        value_states_18 = view_56.transpose(1, 2)
        view_56 = None
        cos_24 = cos_4.unsqueeze(1)
        sin_24 = sin_4.unsqueeze(1)
        mul_112 = query_states_18 * cos_24
        x1_36 = query_states_18[(Ellipsis, slice(None, 32, None))]
        x2_36 = query_states_18[(Ellipsis, slice(32, None, None))]
        query_states_18 = None
        neg_36 = -x2_36
        x2_36 = None
        cat_38 = torch.cat((neg_36, x1_36), dim=-1)
        neg_36 = x1_36 = None
        mul_113 = cat_38 * sin_24
        cat_38 = None
        q_embed_18 = mul_112 + mul_113
        mul_112 = mul_113 = None
        mul_114 = key_states_18 * cos_24
        cos_24 = None
        x1_37 = key_states_18[(Ellipsis, slice(None, 32, None))]
        x2_37 = key_states_18[(Ellipsis, slice(32, None, None))]
        key_states_18 = None
        neg_37 = -x2_37
        x2_37 = None
        cat_39 = torch.cat((neg_37, x1_37), dim=-1)
        neg_37 = x1_37 = None
        mul_115 = cat_39 * sin_24
        cat_39 = sin_24 = None
        k_embed_18 = mul_114 + mul_115
        mul_114 = mul_115 = None
        transpose_95 = k_embed_18.transpose(2, 3)
        matmul_38 = torch.matmul(q_embed_18, transpose_95)
        q_embed_18 = transpose_95 = None
        attn_weights_72 = matmul_38 * 0.125
        matmul_38 = None
        causal_mask_20 = mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        attn_weights_73 = attn_weights_72 + causal_mask_20
        attn_weights_72 = causal_mask_20 = None
        softmax_18 = torch.nn.functional.softmax(
            attn_weights_73, dim=-1, dtype=torch.float32
        )
        attn_weights_73 = None
        attn_weights_74 = softmax_18.to(torch.float32)
        softmax_18 = None
        attn_weights_75 = torch.nn.functional.dropout(
            attn_weights_74, p=0.0, training=False
        )
        attn_weights_74 = None
        attn_output_72 = torch.matmul(attn_weights_75, value_states_18)
        attn_weights_75 = None
        transpose_96 = attn_output_72.transpose(1, 2)
        attn_output_72 = None
        attn_output_73 = transpose_96.contiguous()
        transpose_96 = None
        reshape_18 = attn_output_73.reshape(1, 10, -1)
        attn_output_73 = None
        attn_output_74 = reshape_18.contiguous()
        reshape_18 = None
        linear_111 = torch._C._nn.linear(
            attn_output_74,
            l_self_modules_layers_modules_18_modules_attn_modules_wo_parameters_weight_,
            None,
        )
        attn_output_74 = (
            l_self_modules_layers_modules_18_modules_attn_modules_wo_parameters_weight_
        ) = None
        attn_output_75 = torch.nn.functional.dropout(linear_111, 0.0, False, False)
        linear_111 = None
        hidden_states_73 = hidden_states_71 + attn_output_75
        hidden_states_71 = attn_output_75 = None
        hidden_states_74 = torch.nn.functional.layer_norm(
            hidden_states_73,
            (768,),
            l_self_modules_layers_modules_18_modules_mlp_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_18_modules_mlp_norm_parameters_weight_ = None
        linear_112 = torch._C._nn.linear(
            hidden_states_74,
            l_self_modules_layers_modules_18_modules_mlp_modules_wi_parameters_weight_,
            None,
        )
        hidden_states_74 = (
            l_self_modules_layers_modules_18_modules_mlp_modules_wi_parameters_weight_
        ) = None
        chunk_18 = linear_112.chunk(2, dim=-1)
        linear_112 = None
        input_19 = chunk_18[0]
        gate_18 = chunk_18[1]
        chunk_18 = None
        gelu_18 = torch._C._nn.gelu(input_19)
        input_19 = None
        mul_117 = gelu_18 * gate_18
        gelu_18 = gate_18 = None
        dropout_57 = torch.nn.functional.dropout(mul_117, 0.0, False, False)
        mul_117 = None
        mlp_output_18 = torch._C._nn.linear(
            dropout_57,
            l_self_modules_layers_modules_18_modules_mlp_modules_wo_parameters_weight_,
            None,
        )
        dropout_57 = (
            l_self_modules_layers_modules_18_modules_mlp_modules_wo_parameters_weight_
        ) = None
        hidden_states_75 = hidden_states_73 + mlp_output_18
        hidden_states_73 = mlp_output_18 = None
        hidden_states_76 = torch.nn.functional.layer_norm(
            hidden_states_75,
            (768,),
            l_self_modules_layers_modules_19_modules_attn_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_19_modules_attn_norm_parameters_weight_ = None
        linear_114 = torch._C._nn.linear(
            hidden_states_76,
            l_self_modules_layers_modules_19_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_19_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_57 = linear_114.view((1, 10, -1, 64))
        linear_114 = None
        query_states_19 = view_57.transpose(1, 2)
        view_57 = None
        linear_115 = torch._C._nn.linear(
            hidden_states_76,
            l_self_modules_layers_modules_19_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_19_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_58 = linear_115.view((1, 10, -1, 64))
        linear_115 = None
        key_states_19 = view_58.transpose(1, 2)
        view_58 = None
        linear_116 = torch._C._nn.linear(
            hidden_states_76,
            l_self_modules_layers_modules_19_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_76 = l_self_modules_layers_modules_19_modules_attn_modules_v_proj_parameters_weight_ = (None)
        view_59 = linear_116.view((1, 10, -1, 64))
        linear_116 = None
        value_states_19 = view_59.transpose(1, 2)
        view_59 = None
        cos_25 = cos_6.unsqueeze(1)
        sin_25 = sin_6.unsqueeze(1)
        mul_118 = query_states_19 * cos_25
        x1_38 = query_states_19[(Ellipsis, slice(None, 32, None))]
        x2_38 = query_states_19[(Ellipsis, slice(32, None, None))]
        query_states_19 = None
        neg_38 = -x2_38
        x2_38 = None
        cat_40 = torch.cat((neg_38, x1_38), dim=-1)
        neg_38 = x1_38 = None
        mul_119 = cat_40 * sin_25
        cat_40 = None
        q_embed_19 = mul_118 + mul_119
        mul_118 = mul_119 = None
        mul_120 = key_states_19 * cos_25
        cos_25 = None
        x1_39 = key_states_19[(Ellipsis, slice(None, 32, None))]
        x2_39 = key_states_19[(Ellipsis, slice(32, None, None))]
        key_states_19 = None
        neg_39 = -x2_39
        x2_39 = None
        cat_41 = torch.cat((neg_39, x1_39), dim=-1)
        neg_39 = x1_39 = None
        mul_121 = cat_41 * sin_25
        cat_41 = sin_25 = None
        k_embed_19 = mul_120 + mul_121
        mul_120 = mul_121 = None
        transpose_100 = k_embed_19.transpose(2, 3)
        matmul_40 = torch.matmul(q_embed_19, transpose_100)
        q_embed_19 = transpose_100 = None
        attn_weights_76 = matmul_40 * 0.125
        matmul_40 = None
        causal_mask_21 = mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        attn_weights_77 = attn_weights_76 + causal_mask_21
        attn_weights_76 = causal_mask_21 = None
        softmax_19 = torch.nn.functional.softmax(
            attn_weights_77, dim=-1, dtype=torch.float32
        )
        attn_weights_77 = None
        attn_weights_78 = softmax_19.to(torch.float32)
        softmax_19 = None
        attn_weights_79 = torch.nn.functional.dropout(
            attn_weights_78, p=0.0, training=False
        )
        attn_weights_78 = None
        attn_output_76 = torch.matmul(attn_weights_79, value_states_19)
        attn_weights_79 = None
        transpose_101 = attn_output_76.transpose(1, 2)
        attn_output_76 = None
        attn_output_77 = transpose_101.contiguous()
        transpose_101 = None
        reshape_19 = attn_output_77.reshape(1, 10, -1)
        attn_output_77 = None
        attn_output_78 = reshape_19.contiguous()
        reshape_19 = None
        linear_117 = torch._C._nn.linear(
            attn_output_78,
            l_self_modules_layers_modules_19_modules_attn_modules_wo_parameters_weight_,
            None,
        )
        attn_output_78 = (
            l_self_modules_layers_modules_19_modules_attn_modules_wo_parameters_weight_
        ) = None
        attn_output_79 = torch.nn.functional.dropout(linear_117, 0.0, False, False)
        linear_117 = None
        hidden_states_77 = hidden_states_75 + attn_output_79
        hidden_states_75 = attn_output_79 = None
        hidden_states_78 = torch.nn.functional.layer_norm(
            hidden_states_77,
            (768,),
            l_self_modules_layers_modules_19_modules_mlp_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_19_modules_mlp_norm_parameters_weight_ = None
        linear_118 = torch._C._nn.linear(
            hidden_states_78,
            l_self_modules_layers_modules_19_modules_mlp_modules_wi_parameters_weight_,
            None,
        )
        hidden_states_78 = (
            l_self_modules_layers_modules_19_modules_mlp_modules_wi_parameters_weight_
        ) = None
        chunk_19 = linear_118.chunk(2, dim=-1)
        linear_118 = None
        input_20 = chunk_19[0]
        gate_19 = chunk_19[1]
        chunk_19 = None
        gelu_19 = torch._C._nn.gelu(input_20)
        input_20 = None
        mul_123 = gelu_19 * gate_19
        gelu_19 = gate_19 = None
        dropout_60 = torch.nn.functional.dropout(mul_123, 0.0, False, False)
        mul_123 = None
        mlp_output_19 = torch._C._nn.linear(
            dropout_60,
            l_self_modules_layers_modules_19_modules_mlp_modules_wo_parameters_weight_,
            None,
        )
        dropout_60 = (
            l_self_modules_layers_modules_19_modules_mlp_modules_wo_parameters_weight_
        ) = None
        hidden_states_79 = hidden_states_77 + mlp_output_19
        hidden_states_77 = mlp_output_19 = None
        hidden_states_80 = torch.nn.functional.layer_norm(
            hidden_states_79,
            (768,),
            l_self_modules_layers_modules_20_modules_attn_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_20_modules_attn_norm_parameters_weight_ = None
        linear_120 = torch._C._nn.linear(
            hidden_states_80,
            l_self_modules_layers_modules_20_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_20_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_60 = linear_120.view((1, 10, -1, 64))
        linear_120 = None
        query_states_20 = view_60.transpose(1, 2)
        view_60 = None
        linear_121 = torch._C._nn.linear(
            hidden_states_80,
            l_self_modules_layers_modules_20_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_20_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_61 = linear_121.view((1, 10, -1, 64))
        linear_121 = None
        key_states_20 = view_61.transpose(1, 2)
        view_61 = None
        linear_122 = torch._C._nn.linear(
            hidden_states_80,
            l_self_modules_layers_modules_20_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_80 = l_self_modules_layers_modules_20_modules_attn_modules_v_proj_parameters_weight_ = (None)
        view_62 = linear_122.view((1, 10, -1, 64))
        linear_122 = None
        value_states_20 = view_62.transpose(1, 2)
        view_62 = None
        cos_26 = cos_6.unsqueeze(1)
        cos_6 = None
        sin_26 = sin_6.unsqueeze(1)
        sin_6 = None
        mul_124 = query_states_20 * cos_26
        x1_40 = query_states_20[(Ellipsis, slice(None, 32, None))]
        x2_40 = query_states_20[(Ellipsis, slice(32, None, None))]
        query_states_20 = None
        neg_40 = -x2_40
        x2_40 = None
        cat_42 = torch.cat((neg_40, x1_40), dim=-1)
        neg_40 = x1_40 = None
        mul_125 = cat_42 * sin_26
        cat_42 = None
        q_embed_20 = mul_124 + mul_125
        mul_124 = mul_125 = None
        mul_126 = key_states_20 * cos_26
        cos_26 = None
        x1_41 = key_states_20[(Ellipsis, slice(None, 32, None))]
        x2_41 = key_states_20[(Ellipsis, slice(32, None, None))]
        key_states_20 = None
        neg_41 = -x2_41
        x2_41 = None
        cat_43 = torch.cat((neg_41, x1_41), dim=-1)
        neg_41 = x1_41 = None
        mul_127 = cat_43 * sin_26
        cat_43 = sin_26 = None
        k_embed_20 = mul_126 + mul_127
        mul_126 = mul_127 = None
        transpose_105 = k_embed_20.transpose(2, 3)
        matmul_42 = torch.matmul(q_embed_20, transpose_105)
        q_embed_20 = transpose_105 = None
        attn_weights_80 = matmul_42 * 0.125
        matmul_42 = None
        causal_mask_22 = mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        mask_1 = None
        attn_weights_81 = attn_weights_80 + causal_mask_22
        attn_weights_80 = causal_mask_22 = None
        softmax_20 = torch.nn.functional.softmax(
            attn_weights_81, dim=-1, dtype=torch.float32
        )
        attn_weights_81 = None
        attn_weights_82 = softmax_20.to(torch.float32)
        softmax_20 = None
        attn_weights_83 = torch.nn.functional.dropout(
            attn_weights_82, p=0.0, training=False
        )
        attn_weights_82 = None
        attn_output_80 = torch.matmul(attn_weights_83, value_states_20)
        attn_weights_83 = None
        transpose_106 = attn_output_80.transpose(1, 2)
        attn_output_80 = None
        attn_output_81 = transpose_106.contiguous()
        transpose_106 = None
        reshape_20 = attn_output_81.reshape(1, 10, -1)
        attn_output_81 = None
        attn_output_82 = reshape_20.contiguous()
        reshape_20 = None
        linear_123 = torch._C._nn.linear(
            attn_output_82,
            l_self_modules_layers_modules_20_modules_attn_modules_wo_parameters_weight_,
            None,
        )
        attn_output_82 = (
            l_self_modules_layers_modules_20_modules_attn_modules_wo_parameters_weight_
        ) = None
        attn_output_83 = torch.nn.functional.dropout(linear_123, 0.0, False, False)
        linear_123 = None
        hidden_states_81 = hidden_states_79 + attn_output_83
        hidden_states_79 = attn_output_83 = None
        hidden_states_82 = torch.nn.functional.layer_norm(
            hidden_states_81,
            (768,),
            l_self_modules_layers_modules_20_modules_mlp_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_20_modules_mlp_norm_parameters_weight_ = None
        linear_124 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_layers_modules_20_modules_mlp_modules_wi_parameters_weight_,
            None,
        )
        hidden_states_82 = (
            l_self_modules_layers_modules_20_modules_mlp_modules_wi_parameters_weight_
        ) = None
        chunk_20 = linear_124.chunk(2, dim=-1)
        linear_124 = None
        input_21 = chunk_20[0]
        gate_20 = chunk_20[1]
        chunk_20 = None
        gelu_20 = torch._C._nn.gelu(input_21)
        input_21 = None
        mul_129 = gelu_20 * gate_20
        gelu_20 = gate_20 = None
        dropout_63 = torch.nn.functional.dropout(mul_129, 0.0, False, False)
        mul_129 = None
        mlp_output_20 = torch._C._nn.linear(
            dropout_63,
            l_self_modules_layers_modules_20_modules_mlp_modules_wo_parameters_weight_,
            None,
        )
        dropout_63 = (
            l_self_modules_layers_modules_20_modules_mlp_modules_wo_parameters_weight_
        ) = None
        hidden_states_83 = hidden_states_81 + mlp_output_20
        hidden_states_81 = mlp_output_20 = None
        hidden_states_84 = torch.nn.functional.layer_norm(
            hidden_states_83,
            (768,),
            l_self_modules_layers_modules_21_modules_attn_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_21_modules_attn_norm_parameters_weight_ = None
        linear_126 = torch._C._nn.linear(
            hidden_states_84,
            l_self_modules_layers_modules_21_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_21_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_63 = linear_126.view((1, 10, -1, 64))
        linear_126 = None
        query_states_21 = view_63.transpose(1, 2)
        view_63 = None
        linear_127 = torch._C._nn.linear(
            hidden_states_84,
            l_self_modules_layers_modules_21_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_21_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_64 = linear_127.view((1, 10, -1, 64))
        linear_127 = None
        key_states_21 = view_64.transpose(1, 2)
        view_64 = None
        linear_128 = torch._C._nn.linear(
            hidden_states_84,
            l_self_modules_layers_modules_21_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_84 = l_self_modules_layers_modules_21_modules_attn_modules_v_proj_parameters_weight_ = (None)
        view_65 = linear_128.view((1, 10, -1, 64))
        linear_128 = None
        value_states_21 = view_65.transpose(1, 2)
        view_65 = None
        cos_27 = cos_4.unsqueeze(1)
        cos_4 = None
        sin_27 = sin_4.unsqueeze(1)
        sin_4 = None
        mul_130 = query_states_21 * cos_27
        x1_42 = query_states_21[(Ellipsis, slice(None, 32, None))]
        x2_42 = query_states_21[(Ellipsis, slice(32, None, None))]
        query_states_21 = None
        neg_42 = -x2_42
        x2_42 = None
        cat_44 = torch.cat((neg_42, x1_42), dim=-1)
        neg_42 = x1_42 = None
        mul_131 = cat_44 * sin_27
        cat_44 = None
        q_embed_21 = mul_130 + mul_131
        mul_130 = mul_131 = None
        mul_132 = key_states_21 * cos_27
        cos_27 = None
        x1_43 = key_states_21[(Ellipsis, slice(None, 32, None))]
        x2_43 = key_states_21[(Ellipsis, slice(32, None, None))]
        key_states_21 = None
        neg_43 = -x2_43
        x2_43 = None
        cat_45 = torch.cat((neg_43, x1_43), dim=-1)
        neg_43 = x1_43 = None
        mul_133 = cat_45 * sin_27
        cat_45 = sin_27 = None
        k_embed_21 = mul_132 + mul_133
        mul_132 = mul_133 = None
        transpose_110 = k_embed_21.transpose(2, 3)
        matmul_44 = torch.matmul(q_embed_21, transpose_110)
        q_embed_21 = transpose_110 = None
        attn_weights_84 = matmul_44 * 0.125
        matmul_44 = None
        causal_mask_23 = mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        mask = None
        attn_weights_85 = attn_weights_84 + causal_mask_23
        attn_weights_84 = causal_mask_23 = None
        softmax_21 = torch.nn.functional.softmax(
            attn_weights_85, dim=-1, dtype=torch.float32
        )
        attn_weights_85 = None
        attn_weights_86 = softmax_21.to(torch.float32)
        softmax_21 = None
        attn_weights_87 = torch.nn.functional.dropout(
            attn_weights_86, p=0.0, training=False
        )
        attn_weights_86 = None
        attn_output_84 = torch.matmul(attn_weights_87, value_states_21)
        attn_weights_87 = None
        transpose_111 = attn_output_84.transpose(1, 2)
        attn_output_84 = None
        attn_output_85 = transpose_111.contiguous()
        transpose_111 = None
        reshape_21 = attn_output_85.reshape(1, 10, -1)
        attn_output_85 = None
        attn_output_86 = reshape_21.contiguous()
        reshape_21 = None
        linear_129 = torch._C._nn.linear(
            attn_output_86,
            l_self_modules_layers_modules_21_modules_attn_modules_wo_parameters_weight_,
            None,
        )
        attn_output_86 = (
            l_self_modules_layers_modules_21_modules_attn_modules_wo_parameters_weight_
        ) = None
        attn_output_87 = torch.nn.functional.dropout(linear_129, 0.0, False, False)
        linear_129 = None
        hidden_states_85 = hidden_states_83 + attn_output_87
        hidden_states_83 = attn_output_87 = None
        hidden_states_86 = torch.nn.functional.layer_norm(
            hidden_states_85,
            (768,),
            l_self_modules_layers_modules_21_modules_mlp_norm_parameters_weight_,
            None,
            1e-05,
        )
        l_self_modules_layers_modules_21_modules_mlp_norm_parameters_weight_ = None
        linear_130 = torch._C._nn.linear(
            hidden_states_86,
            l_self_modules_layers_modules_21_modules_mlp_modules_wi_parameters_weight_,
            None,
        )
        hidden_states_86 = (
            l_self_modules_layers_modules_21_modules_mlp_modules_wi_parameters_weight_
        ) = None
        chunk_21 = linear_130.chunk(2, dim=-1)
        linear_130 = None
        input_22 = chunk_21[0]
        gate_21 = chunk_21[1]
        chunk_21 = None
        gelu_21 = torch._C._nn.gelu(input_22)
        input_22 = None
        mul_135 = gelu_21 * gate_21
        gelu_21 = gate_21 = None
        dropout_66 = torch.nn.functional.dropout(mul_135, 0.0, False, False)
        mul_135 = None
        mlp_output_21 = torch._C._nn.linear(
            dropout_66,
            l_self_modules_layers_modules_21_modules_mlp_modules_wo_parameters_weight_,
            None,
        )
        dropout_66 = (
            l_self_modules_layers_modules_21_modules_mlp_modules_wo_parameters_weight_
        ) = None
        hidden_states_87 = hidden_states_85 + mlp_output_21
        hidden_states_85 = mlp_output_21 = None
        hidden_states_88 = torch.nn.functional.layer_norm(
            hidden_states_87,
            (768,),
            l_self_modules_final_norm_parameters_weight_,
            None,
            1e-05,
        )
        hidden_states_87 = l_self_modules_final_norm_parameters_weight_ = None
        return (
            value_states,
            k_embed,
            value_states_1,
            k_embed_1,
            value_states_2,
            k_embed_2,
            value_states_3,
            k_embed_3,
            value_states_4,
            k_embed_4,
            value_states_5,
            k_embed_5,
            value_states_6,
            k_embed_6,
            value_states_7,
            k_embed_7,
            value_states_8,
            k_embed_8,
            value_states_9,
            k_embed_9,
            value_states_10,
            k_embed_10,
            value_states_11,
            k_embed_11,
            value_states_12,
            k_embed_12,
            value_states_13,
            k_embed_13,
            value_states_14,
            k_embed_14,
            value_states_15,
            k_embed_15,
            value_states_16,
            k_embed_16,
            value_states_17,
            k_embed_17,
            value_states_18,
            k_embed_18,
            value_states_19,
            k_embed_19,
            value_states_20,
            k_embed_20,
            value_states_21,
            k_embed_21,
            hidden_states_88,
        )
