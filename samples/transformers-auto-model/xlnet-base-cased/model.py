import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_token_type_ids_: torch.Tensor,
        L_attention_mask_: torch.Tensor,
        L_self_modules_word_embedding_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_rel_attn_parameters_q_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_rel_attn_parameters_k_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_rel_attn_parameters_v_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_rel_attn_parameters_r_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_rel_attn_parameters_r_w_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_rel_attn_parameters_r_r_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_rel_attn_parameters_r_s_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_rel_attn_parameters_seg_embed_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_rel_attn_parameters_o_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_rel_attn_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_rel_attn_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_ff_modules_layer_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_ff_modules_layer_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_ff_modules_layer_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_ff_modules_layer_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_ff_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_ff_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_rel_attn_parameters_q_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_rel_attn_parameters_k_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_rel_attn_parameters_v_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_rel_attn_parameters_r_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_rel_attn_parameters_r_w_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_rel_attn_parameters_r_r_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_rel_attn_parameters_r_s_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_rel_attn_parameters_seg_embed_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_rel_attn_parameters_o_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_rel_attn_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_rel_attn_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_ff_modules_layer_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_ff_modules_layer_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_ff_modules_layer_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_ff_modules_layer_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_ff_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_ff_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_rel_attn_parameters_q_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_rel_attn_parameters_k_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_rel_attn_parameters_v_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_rel_attn_parameters_r_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_rel_attn_parameters_r_w_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_rel_attn_parameters_r_r_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_rel_attn_parameters_r_s_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_rel_attn_parameters_seg_embed_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_rel_attn_parameters_o_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_rel_attn_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_rel_attn_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_ff_modules_layer_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_ff_modules_layer_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_ff_modules_layer_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_ff_modules_layer_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_ff_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_ff_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_rel_attn_parameters_q_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_rel_attn_parameters_k_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_rel_attn_parameters_v_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_rel_attn_parameters_r_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_rel_attn_parameters_r_w_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_rel_attn_parameters_r_r_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_rel_attn_parameters_r_s_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_rel_attn_parameters_seg_embed_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_rel_attn_parameters_o_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_rel_attn_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_rel_attn_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_ff_modules_layer_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_ff_modules_layer_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_ff_modules_layer_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_ff_modules_layer_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_ff_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_ff_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_rel_attn_parameters_q_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_rel_attn_parameters_k_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_rel_attn_parameters_v_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_rel_attn_parameters_r_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_rel_attn_parameters_r_w_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_rel_attn_parameters_r_r_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_rel_attn_parameters_r_s_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_rel_attn_parameters_seg_embed_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_rel_attn_parameters_o_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_rel_attn_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_rel_attn_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_ff_modules_layer_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_ff_modules_layer_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_ff_modules_layer_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_ff_modules_layer_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_ff_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_ff_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_rel_attn_parameters_q_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_rel_attn_parameters_k_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_rel_attn_parameters_v_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_rel_attn_parameters_r_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_rel_attn_parameters_r_w_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_rel_attn_parameters_r_r_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_rel_attn_parameters_r_s_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_rel_attn_parameters_seg_embed_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_rel_attn_parameters_o_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_rel_attn_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_rel_attn_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_ff_modules_layer_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_ff_modules_layer_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_ff_modules_layer_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_ff_modules_layer_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_ff_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_ff_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_rel_attn_parameters_q_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_rel_attn_parameters_k_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_rel_attn_parameters_v_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_rel_attn_parameters_r_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_rel_attn_parameters_r_w_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_rel_attn_parameters_r_r_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_rel_attn_parameters_r_s_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_rel_attn_parameters_seg_embed_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_rel_attn_parameters_o_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_rel_attn_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_rel_attn_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_ff_modules_layer_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_ff_modules_layer_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_ff_modules_layer_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_ff_modules_layer_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_ff_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_ff_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_rel_attn_parameters_q_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_rel_attn_parameters_k_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_rel_attn_parameters_v_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_rel_attn_parameters_r_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_rel_attn_parameters_r_w_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_rel_attn_parameters_r_r_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_rel_attn_parameters_r_s_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_rel_attn_parameters_seg_embed_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_rel_attn_parameters_o_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_rel_attn_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_rel_attn_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_ff_modules_layer_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_ff_modules_layer_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_ff_modules_layer_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_ff_modules_layer_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_ff_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_ff_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_rel_attn_parameters_q_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_rel_attn_parameters_k_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_rel_attn_parameters_v_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_rel_attn_parameters_r_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_rel_attn_parameters_r_w_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_rel_attn_parameters_r_r_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_rel_attn_parameters_r_s_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_rel_attn_parameters_seg_embed_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_rel_attn_parameters_o_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_rel_attn_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_rel_attn_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_ff_modules_layer_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_ff_modules_layer_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_ff_modules_layer_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_ff_modules_layer_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_ff_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_ff_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_rel_attn_parameters_q_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_rel_attn_parameters_k_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_rel_attn_parameters_v_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_rel_attn_parameters_r_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_rel_attn_parameters_r_w_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_rel_attn_parameters_r_r_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_rel_attn_parameters_r_s_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_rel_attn_parameters_seg_embed_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_rel_attn_parameters_o_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_rel_attn_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_rel_attn_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_ff_modules_layer_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_ff_modules_layer_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_ff_modules_layer_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_ff_modules_layer_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_ff_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_ff_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_rel_attn_parameters_q_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_rel_attn_parameters_k_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_rel_attn_parameters_v_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_rel_attn_parameters_r_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_rel_attn_parameters_r_w_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_rel_attn_parameters_r_r_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_rel_attn_parameters_r_s_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_rel_attn_parameters_seg_embed_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_rel_attn_parameters_o_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_rel_attn_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_rel_attn_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_ff_modules_layer_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_ff_modules_layer_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_ff_modules_layer_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_ff_modules_layer_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_ff_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_ff_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_rel_attn_parameters_q_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_rel_attn_parameters_k_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_rel_attn_parameters_v_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_rel_attn_parameters_r_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_rel_attn_parameters_r_w_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_rel_attn_parameters_r_r_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_rel_attn_parameters_r_s_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_rel_attn_parameters_seg_embed_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_rel_attn_parameters_o_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_rel_attn_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_rel_attn_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_ff_modules_layer_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_ff_modules_layer_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_ff_modules_layer_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_ff_modules_layer_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_ff_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_ff_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_input_ids_ = L_input_ids_
        l_token_type_ids_ = L_token_type_ids_
        l_attention_mask_ = L_attention_mask_
        l_self_modules_word_embedding_parameters_weight_ = (
            L_self_modules_word_embedding_parameters_weight_
        )
        l_self_modules_layer_modules_0_modules_rel_attn_parameters_q_ = (
            L_self_modules_layer_modules_0_modules_rel_attn_parameters_q_
        )
        l_self_modules_layer_modules_0_modules_rel_attn_parameters_k_ = (
            L_self_modules_layer_modules_0_modules_rel_attn_parameters_k_
        )
        l_self_modules_layer_modules_0_modules_rel_attn_parameters_v_ = (
            L_self_modules_layer_modules_0_modules_rel_attn_parameters_v_
        )
        l_self_modules_layer_modules_0_modules_rel_attn_parameters_r_ = (
            L_self_modules_layer_modules_0_modules_rel_attn_parameters_r_
        )
        l_self_modules_layer_modules_0_modules_rel_attn_parameters_r_w_bias_ = (
            L_self_modules_layer_modules_0_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_0_modules_rel_attn_parameters_r_r_bias_ = (
            L_self_modules_layer_modules_0_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_0_modules_rel_attn_parameters_r_s_bias_ = (
            L_self_modules_layer_modules_0_modules_rel_attn_parameters_r_s_bias_
        )
        l_self_modules_layer_modules_0_modules_rel_attn_parameters_seg_embed_ = (
            L_self_modules_layer_modules_0_modules_rel_attn_parameters_seg_embed_
        )
        l_self_modules_layer_modules_0_modules_rel_attn_parameters_o_ = (
            L_self_modules_layer_modules_0_modules_rel_attn_parameters_o_
        )
        l_self_modules_layer_modules_0_modules_rel_attn_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_0_modules_rel_attn_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_0_modules_rel_attn_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_0_modules_rel_attn_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_0_modules_ff_modules_layer_1_parameters_weight_ = (
            L_self_modules_layer_modules_0_modules_ff_modules_layer_1_parameters_weight_
        )
        l_self_modules_layer_modules_0_modules_ff_modules_layer_1_parameters_bias_ = (
            L_self_modules_layer_modules_0_modules_ff_modules_layer_1_parameters_bias_
        )
        l_self_modules_layer_modules_0_modules_ff_modules_layer_2_parameters_weight_ = (
            L_self_modules_layer_modules_0_modules_ff_modules_layer_2_parameters_weight_
        )
        l_self_modules_layer_modules_0_modules_ff_modules_layer_2_parameters_bias_ = (
            L_self_modules_layer_modules_0_modules_ff_modules_layer_2_parameters_bias_
        )
        l_self_modules_layer_modules_0_modules_ff_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_0_modules_ff_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_0_modules_ff_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_0_modules_ff_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_1_modules_rel_attn_parameters_q_ = (
            L_self_modules_layer_modules_1_modules_rel_attn_parameters_q_
        )
        l_self_modules_layer_modules_1_modules_rel_attn_parameters_k_ = (
            L_self_modules_layer_modules_1_modules_rel_attn_parameters_k_
        )
        l_self_modules_layer_modules_1_modules_rel_attn_parameters_v_ = (
            L_self_modules_layer_modules_1_modules_rel_attn_parameters_v_
        )
        l_self_modules_layer_modules_1_modules_rel_attn_parameters_r_ = (
            L_self_modules_layer_modules_1_modules_rel_attn_parameters_r_
        )
        l_self_modules_layer_modules_1_modules_rel_attn_parameters_r_w_bias_ = (
            L_self_modules_layer_modules_1_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_1_modules_rel_attn_parameters_r_r_bias_ = (
            L_self_modules_layer_modules_1_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_1_modules_rel_attn_parameters_r_s_bias_ = (
            L_self_modules_layer_modules_1_modules_rel_attn_parameters_r_s_bias_
        )
        l_self_modules_layer_modules_1_modules_rel_attn_parameters_seg_embed_ = (
            L_self_modules_layer_modules_1_modules_rel_attn_parameters_seg_embed_
        )
        l_self_modules_layer_modules_1_modules_rel_attn_parameters_o_ = (
            L_self_modules_layer_modules_1_modules_rel_attn_parameters_o_
        )
        l_self_modules_layer_modules_1_modules_rel_attn_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_1_modules_rel_attn_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_1_modules_rel_attn_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_1_modules_rel_attn_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_1_modules_ff_modules_layer_1_parameters_weight_ = (
            L_self_modules_layer_modules_1_modules_ff_modules_layer_1_parameters_weight_
        )
        l_self_modules_layer_modules_1_modules_ff_modules_layer_1_parameters_bias_ = (
            L_self_modules_layer_modules_1_modules_ff_modules_layer_1_parameters_bias_
        )
        l_self_modules_layer_modules_1_modules_ff_modules_layer_2_parameters_weight_ = (
            L_self_modules_layer_modules_1_modules_ff_modules_layer_2_parameters_weight_
        )
        l_self_modules_layer_modules_1_modules_ff_modules_layer_2_parameters_bias_ = (
            L_self_modules_layer_modules_1_modules_ff_modules_layer_2_parameters_bias_
        )
        l_self_modules_layer_modules_1_modules_ff_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_1_modules_ff_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_1_modules_ff_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_1_modules_ff_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_2_modules_rel_attn_parameters_q_ = (
            L_self_modules_layer_modules_2_modules_rel_attn_parameters_q_
        )
        l_self_modules_layer_modules_2_modules_rel_attn_parameters_k_ = (
            L_self_modules_layer_modules_2_modules_rel_attn_parameters_k_
        )
        l_self_modules_layer_modules_2_modules_rel_attn_parameters_v_ = (
            L_self_modules_layer_modules_2_modules_rel_attn_parameters_v_
        )
        l_self_modules_layer_modules_2_modules_rel_attn_parameters_r_ = (
            L_self_modules_layer_modules_2_modules_rel_attn_parameters_r_
        )
        l_self_modules_layer_modules_2_modules_rel_attn_parameters_r_w_bias_ = (
            L_self_modules_layer_modules_2_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_2_modules_rel_attn_parameters_r_r_bias_ = (
            L_self_modules_layer_modules_2_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_2_modules_rel_attn_parameters_r_s_bias_ = (
            L_self_modules_layer_modules_2_modules_rel_attn_parameters_r_s_bias_
        )
        l_self_modules_layer_modules_2_modules_rel_attn_parameters_seg_embed_ = (
            L_self_modules_layer_modules_2_modules_rel_attn_parameters_seg_embed_
        )
        l_self_modules_layer_modules_2_modules_rel_attn_parameters_o_ = (
            L_self_modules_layer_modules_2_modules_rel_attn_parameters_o_
        )
        l_self_modules_layer_modules_2_modules_rel_attn_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_2_modules_rel_attn_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_2_modules_rel_attn_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_2_modules_rel_attn_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_2_modules_ff_modules_layer_1_parameters_weight_ = (
            L_self_modules_layer_modules_2_modules_ff_modules_layer_1_parameters_weight_
        )
        l_self_modules_layer_modules_2_modules_ff_modules_layer_1_parameters_bias_ = (
            L_self_modules_layer_modules_2_modules_ff_modules_layer_1_parameters_bias_
        )
        l_self_modules_layer_modules_2_modules_ff_modules_layer_2_parameters_weight_ = (
            L_self_modules_layer_modules_2_modules_ff_modules_layer_2_parameters_weight_
        )
        l_self_modules_layer_modules_2_modules_ff_modules_layer_2_parameters_bias_ = (
            L_self_modules_layer_modules_2_modules_ff_modules_layer_2_parameters_bias_
        )
        l_self_modules_layer_modules_2_modules_ff_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_2_modules_ff_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_2_modules_ff_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_2_modules_ff_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_3_modules_rel_attn_parameters_q_ = (
            L_self_modules_layer_modules_3_modules_rel_attn_parameters_q_
        )
        l_self_modules_layer_modules_3_modules_rel_attn_parameters_k_ = (
            L_self_modules_layer_modules_3_modules_rel_attn_parameters_k_
        )
        l_self_modules_layer_modules_3_modules_rel_attn_parameters_v_ = (
            L_self_modules_layer_modules_3_modules_rel_attn_parameters_v_
        )
        l_self_modules_layer_modules_3_modules_rel_attn_parameters_r_ = (
            L_self_modules_layer_modules_3_modules_rel_attn_parameters_r_
        )
        l_self_modules_layer_modules_3_modules_rel_attn_parameters_r_w_bias_ = (
            L_self_modules_layer_modules_3_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_3_modules_rel_attn_parameters_r_r_bias_ = (
            L_self_modules_layer_modules_3_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_3_modules_rel_attn_parameters_r_s_bias_ = (
            L_self_modules_layer_modules_3_modules_rel_attn_parameters_r_s_bias_
        )
        l_self_modules_layer_modules_3_modules_rel_attn_parameters_seg_embed_ = (
            L_self_modules_layer_modules_3_modules_rel_attn_parameters_seg_embed_
        )
        l_self_modules_layer_modules_3_modules_rel_attn_parameters_o_ = (
            L_self_modules_layer_modules_3_modules_rel_attn_parameters_o_
        )
        l_self_modules_layer_modules_3_modules_rel_attn_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_3_modules_rel_attn_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_3_modules_rel_attn_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_3_modules_rel_attn_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_3_modules_ff_modules_layer_1_parameters_weight_ = (
            L_self_modules_layer_modules_3_modules_ff_modules_layer_1_parameters_weight_
        )
        l_self_modules_layer_modules_3_modules_ff_modules_layer_1_parameters_bias_ = (
            L_self_modules_layer_modules_3_modules_ff_modules_layer_1_parameters_bias_
        )
        l_self_modules_layer_modules_3_modules_ff_modules_layer_2_parameters_weight_ = (
            L_self_modules_layer_modules_3_modules_ff_modules_layer_2_parameters_weight_
        )
        l_self_modules_layer_modules_3_modules_ff_modules_layer_2_parameters_bias_ = (
            L_self_modules_layer_modules_3_modules_ff_modules_layer_2_parameters_bias_
        )
        l_self_modules_layer_modules_3_modules_ff_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_3_modules_ff_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_3_modules_ff_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_3_modules_ff_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_4_modules_rel_attn_parameters_q_ = (
            L_self_modules_layer_modules_4_modules_rel_attn_parameters_q_
        )
        l_self_modules_layer_modules_4_modules_rel_attn_parameters_k_ = (
            L_self_modules_layer_modules_4_modules_rel_attn_parameters_k_
        )
        l_self_modules_layer_modules_4_modules_rel_attn_parameters_v_ = (
            L_self_modules_layer_modules_4_modules_rel_attn_parameters_v_
        )
        l_self_modules_layer_modules_4_modules_rel_attn_parameters_r_ = (
            L_self_modules_layer_modules_4_modules_rel_attn_parameters_r_
        )
        l_self_modules_layer_modules_4_modules_rel_attn_parameters_r_w_bias_ = (
            L_self_modules_layer_modules_4_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_4_modules_rel_attn_parameters_r_r_bias_ = (
            L_self_modules_layer_modules_4_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_4_modules_rel_attn_parameters_r_s_bias_ = (
            L_self_modules_layer_modules_4_modules_rel_attn_parameters_r_s_bias_
        )
        l_self_modules_layer_modules_4_modules_rel_attn_parameters_seg_embed_ = (
            L_self_modules_layer_modules_4_modules_rel_attn_parameters_seg_embed_
        )
        l_self_modules_layer_modules_4_modules_rel_attn_parameters_o_ = (
            L_self_modules_layer_modules_4_modules_rel_attn_parameters_o_
        )
        l_self_modules_layer_modules_4_modules_rel_attn_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_4_modules_rel_attn_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_4_modules_rel_attn_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_4_modules_rel_attn_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_4_modules_ff_modules_layer_1_parameters_weight_ = (
            L_self_modules_layer_modules_4_modules_ff_modules_layer_1_parameters_weight_
        )
        l_self_modules_layer_modules_4_modules_ff_modules_layer_1_parameters_bias_ = (
            L_self_modules_layer_modules_4_modules_ff_modules_layer_1_parameters_bias_
        )
        l_self_modules_layer_modules_4_modules_ff_modules_layer_2_parameters_weight_ = (
            L_self_modules_layer_modules_4_modules_ff_modules_layer_2_parameters_weight_
        )
        l_self_modules_layer_modules_4_modules_ff_modules_layer_2_parameters_bias_ = (
            L_self_modules_layer_modules_4_modules_ff_modules_layer_2_parameters_bias_
        )
        l_self_modules_layer_modules_4_modules_ff_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_4_modules_ff_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_4_modules_ff_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_4_modules_ff_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_5_modules_rel_attn_parameters_q_ = (
            L_self_modules_layer_modules_5_modules_rel_attn_parameters_q_
        )
        l_self_modules_layer_modules_5_modules_rel_attn_parameters_k_ = (
            L_self_modules_layer_modules_5_modules_rel_attn_parameters_k_
        )
        l_self_modules_layer_modules_5_modules_rel_attn_parameters_v_ = (
            L_self_modules_layer_modules_5_modules_rel_attn_parameters_v_
        )
        l_self_modules_layer_modules_5_modules_rel_attn_parameters_r_ = (
            L_self_modules_layer_modules_5_modules_rel_attn_parameters_r_
        )
        l_self_modules_layer_modules_5_modules_rel_attn_parameters_r_w_bias_ = (
            L_self_modules_layer_modules_5_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_5_modules_rel_attn_parameters_r_r_bias_ = (
            L_self_modules_layer_modules_5_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_5_modules_rel_attn_parameters_r_s_bias_ = (
            L_self_modules_layer_modules_5_modules_rel_attn_parameters_r_s_bias_
        )
        l_self_modules_layer_modules_5_modules_rel_attn_parameters_seg_embed_ = (
            L_self_modules_layer_modules_5_modules_rel_attn_parameters_seg_embed_
        )
        l_self_modules_layer_modules_5_modules_rel_attn_parameters_o_ = (
            L_self_modules_layer_modules_5_modules_rel_attn_parameters_o_
        )
        l_self_modules_layer_modules_5_modules_rel_attn_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_5_modules_rel_attn_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_5_modules_rel_attn_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_5_modules_rel_attn_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_5_modules_ff_modules_layer_1_parameters_weight_ = (
            L_self_modules_layer_modules_5_modules_ff_modules_layer_1_parameters_weight_
        )
        l_self_modules_layer_modules_5_modules_ff_modules_layer_1_parameters_bias_ = (
            L_self_modules_layer_modules_5_modules_ff_modules_layer_1_parameters_bias_
        )
        l_self_modules_layer_modules_5_modules_ff_modules_layer_2_parameters_weight_ = (
            L_self_modules_layer_modules_5_modules_ff_modules_layer_2_parameters_weight_
        )
        l_self_modules_layer_modules_5_modules_ff_modules_layer_2_parameters_bias_ = (
            L_self_modules_layer_modules_5_modules_ff_modules_layer_2_parameters_bias_
        )
        l_self_modules_layer_modules_5_modules_ff_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_5_modules_ff_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_5_modules_ff_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_5_modules_ff_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_6_modules_rel_attn_parameters_q_ = (
            L_self_modules_layer_modules_6_modules_rel_attn_parameters_q_
        )
        l_self_modules_layer_modules_6_modules_rel_attn_parameters_k_ = (
            L_self_modules_layer_modules_6_modules_rel_attn_parameters_k_
        )
        l_self_modules_layer_modules_6_modules_rel_attn_parameters_v_ = (
            L_self_modules_layer_modules_6_modules_rel_attn_parameters_v_
        )
        l_self_modules_layer_modules_6_modules_rel_attn_parameters_r_ = (
            L_self_modules_layer_modules_6_modules_rel_attn_parameters_r_
        )
        l_self_modules_layer_modules_6_modules_rel_attn_parameters_r_w_bias_ = (
            L_self_modules_layer_modules_6_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_6_modules_rel_attn_parameters_r_r_bias_ = (
            L_self_modules_layer_modules_6_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_6_modules_rel_attn_parameters_r_s_bias_ = (
            L_self_modules_layer_modules_6_modules_rel_attn_parameters_r_s_bias_
        )
        l_self_modules_layer_modules_6_modules_rel_attn_parameters_seg_embed_ = (
            L_self_modules_layer_modules_6_modules_rel_attn_parameters_seg_embed_
        )
        l_self_modules_layer_modules_6_modules_rel_attn_parameters_o_ = (
            L_self_modules_layer_modules_6_modules_rel_attn_parameters_o_
        )
        l_self_modules_layer_modules_6_modules_rel_attn_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_6_modules_rel_attn_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_6_modules_rel_attn_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_6_modules_rel_attn_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_6_modules_ff_modules_layer_1_parameters_weight_ = (
            L_self_modules_layer_modules_6_modules_ff_modules_layer_1_parameters_weight_
        )
        l_self_modules_layer_modules_6_modules_ff_modules_layer_1_parameters_bias_ = (
            L_self_modules_layer_modules_6_modules_ff_modules_layer_1_parameters_bias_
        )
        l_self_modules_layer_modules_6_modules_ff_modules_layer_2_parameters_weight_ = (
            L_self_modules_layer_modules_6_modules_ff_modules_layer_2_parameters_weight_
        )
        l_self_modules_layer_modules_6_modules_ff_modules_layer_2_parameters_bias_ = (
            L_self_modules_layer_modules_6_modules_ff_modules_layer_2_parameters_bias_
        )
        l_self_modules_layer_modules_6_modules_ff_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_6_modules_ff_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_6_modules_ff_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_6_modules_ff_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_7_modules_rel_attn_parameters_q_ = (
            L_self_modules_layer_modules_7_modules_rel_attn_parameters_q_
        )
        l_self_modules_layer_modules_7_modules_rel_attn_parameters_k_ = (
            L_self_modules_layer_modules_7_modules_rel_attn_parameters_k_
        )
        l_self_modules_layer_modules_7_modules_rel_attn_parameters_v_ = (
            L_self_modules_layer_modules_7_modules_rel_attn_parameters_v_
        )
        l_self_modules_layer_modules_7_modules_rel_attn_parameters_r_ = (
            L_self_modules_layer_modules_7_modules_rel_attn_parameters_r_
        )
        l_self_modules_layer_modules_7_modules_rel_attn_parameters_r_w_bias_ = (
            L_self_modules_layer_modules_7_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_7_modules_rel_attn_parameters_r_r_bias_ = (
            L_self_modules_layer_modules_7_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_7_modules_rel_attn_parameters_r_s_bias_ = (
            L_self_modules_layer_modules_7_modules_rel_attn_parameters_r_s_bias_
        )
        l_self_modules_layer_modules_7_modules_rel_attn_parameters_seg_embed_ = (
            L_self_modules_layer_modules_7_modules_rel_attn_parameters_seg_embed_
        )
        l_self_modules_layer_modules_7_modules_rel_attn_parameters_o_ = (
            L_self_modules_layer_modules_7_modules_rel_attn_parameters_o_
        )
        l_self_modules_layer_modules_7_modules_rel_attn_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_7_modules_rel_attn_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_7_modules_rel_attn_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_7_modules_rel_attn_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_7_modules_ff_modules_layer_1_parameters_weight_ = (
            L_self_modules_layer_modules_7_modules_ff_modules_layer_1_parameters_weight_
        )
        l_self_modules_layer_modules_7_modules_ff_modules_layer_1_parameters_bias_ = (
            L_self_modules_layer_modules_7_modules_ff_modules_layer_1_parameters_bias_
        )
        l_self_modules_layer_modules_7_modules_ff_modules_layer_2_parameters_weight_ = (
            L_self_modules_layer_modules_7_modules_ff_modules_layer_2_parameters_weight_
        )
        l_self_modules_layer_modules_7_modules_ff_modules_layer_2_parameters_bias_ = (
            L_self_modules_layer_modules_7_modules_ff_modules_layer_2_parameters_bias_
        )
        l_self_modules_layer_modules_7_modules_ff_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_7_modules_ff_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_7_modules_ff_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_7_modules_ff_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_8_modules_rel_attn_parameters_q_ = (
            L_self_modules_layer_modules_8_modules_rel_attn_parameters_q_
        )
        l_self_modules_layer_modules_8_modules_rel_attn_parameters_k_ = (
            L_self_modules_layer_modules_8_modules_rel_attn_parameters_k_
        )
        l_self_modules_layer_modules_8_modules_rel_attn_parameters_v_ = (
            L_self_modules_layer_modules_8_modules_rel_attn_parameters_v_
        )
        l_self_modules_layer_modules_8_modules_rel_attn_parameters_r_ = (
            L_self_modules_layer_modules_8_modules_rel_attn_parameters_r_
        )
        l_self_modules_layer_modules_8_modules_rel_attn_parameters_r_w_bias_ = (
            L_self_modules_layer_modules_8_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_8_modules_rel_attn_parameters_r_r_bias_ = (
            L_self_modules_layer_modules_8_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_8_modules_rel_attn_parameters_r_s_bias_ = (
            L_self_modules_layer_modules_8_modules_rel_attn_parameters_r_s_bias_
        )
        l_self_modules_layer_modules_8_modules_rel_attn_parameters_seg_embed_ = (
            L_self_modules_layer_modules_8_modules_rel_attn_parameters_seg_embed_
        )
        l_self_modules_layer_modules_8_modules_rel_attn_parameters_o_ = (
            L_self_modules_layer_modules_8_modules_rel_attn_parameters_o_
        )
        l_self_modules_layer_modules_8_modules_rel_attn_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_8_modules_rel_attn_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_8_modules_rel_attn_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_8_modules_rel_attn_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_8_modules_ff_modules_layer_1_parameters_weight_ = (
            L_self_modules_layer_modules_8_modules_ff_modules_layer_1_parameters_weight_
        )
        l_self_modules_layer_modules_8_modules_ff_modules_layer_1_parameters_bias_ = (
            L_self_modules_layer_modules_8_modules_ff_modules_layer_1_parameters_bias_
        )
        l_self_modules_layer_modules_8_modules_ff_modules_layer_2_parameters_weight_ = (
            L_self_modules_layer_modules_8_modules_ff_modules_layer_2_parameters_weight_
        )
        l_self_modules_layer_modules_8_modules_ff_modules_layer_2_parameters_bias_ = (
            L_self_modules_layer_modules_8_modules_ff_modules_layer_2_parameters_bias_
        )
        l_self_modules_layer_modules_8_modules_ff_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_8_modules_ff_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_8_modules_ff_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_8_modules_ff_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_9_modules_rel_attn_parameters_q_ = (
            L_self_modules_layer_modules_9_modules_rel_attn_parameters_q_
        )
        l_self_modules_layer_modules_9_modules_rel_attn_parameters_k_ = (
            L_self_modules_layer_modules_9_modules_rel_attn_parameters_k_
        )
        l_self_modules_layer_modules_9_modules_rel_attn_parameters_v_ = (
            L_self_modules_layer_modules_9_modules_rel_attn_parameters_v_
        )
        l_self_modules_layer_modules_9_modules_rel_attn_parameters_r_ = (
            L_self_modules_layer_modules_9_modules_rel_attn_parameters_r_
        )
        l_self_modules_layer_modules_9_modules_rel_attn_parameters_r_w_bias_ = (
            L_self_modules_layer_modules_9_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_9_modules_rel_attn_parameters_r_r_bias_ = (
            L_self_modules_layer_modules_9_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_9_modules_rel_attn_parameters_r_s_bias_ = (
            L_self_modules_layer_modules_9_modules_rel_attn_parameters_r_s_bias_
        )
        l_self_modules_layer_modules_9_modules_rel_attn_parameters_seg_embed_ = (
            L_self_modules_layer_modules_9_modules_rel_attn_parameters_seg_embed_
        )
        l_self_modules_layer_modules_9_modules_rel_attn_parameters_o_ = (
            L_self_modules_layer_modules_9_modules_rel_attn_parameters_o_
        )
        l_self_modules_layer_modules_9_modules_rel_attn_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_9_modules_rel_attn_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_9_modules_rel_attn_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_9_modules_rel_attn_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_9_modules_ff_modules_layer_1_parameters_weight_ = (
            L_self_modules_layer_modules_9_modules_ff_modules_layer_1_parameters_weight_
        )
        l_self_modules_layer_modules_9_modules_ff_modules_layer_1_parameters_bias_ = (
            L_self_modules_layer_modules_9_modules_ff_modules_layer_1_parameters_bias_
        )
        l_self_modules_layer_modules_9_modules_ff_modules_layer_2_parameters_weight_ = (
            L_self_modules_layer_modules_9_modules_ff_modules_layer_2_parameters_weight_
        )
        l_self_modules_layer_modules_9_modules_ff_modules_layer_2_parameters_bias_ = (
            L_self_modules_layer_modules_9_modules_ff_modules_layer_2_parameters_bias_
        )
        l_self_modules_layer_modules_9_modules_ff_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_9_modules_ff_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_9_modules_ff_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_9_modules_ff_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_10_modules_rel_attn_parameters_q_ = (
            L_self_modules_layer_modules_10_modules_rel_attn_parameters_q_
        )
        l_self_modules_layer_modules_10_modules_rel_attn_parameters_k_ = (
            L_self_modules_layer_modules_10_modules_rel_attn_parameters_k_
        )
        l_self_modules_layer_modules_10_modules_rel_attn_parameters_v_ = (
            L_self_modules_layer_modules_10_modules_rel_attn_parameters_v_
        )
        l_self_modules_layer_modules_10_modules_rel_attn_parameters_r_ = (
            L_self_modules_layer_modules_10_modules_rel_attn_parameters_r_
        )
        l_self_modules_layer_modules_10_modules_rel_attn_parameters_r_w_bias_ = (
            L_self_modules_layer_modules_10_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_10_modules_rel_attn_parameters_r_r_bias_ = (
            L_self_modules_layer_modules_10_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_10_modules_rel_attn_parameters_r_s_bias_ = (
            L_self_modules_layer_modules_10_modules_rel_attn_parameters_r_s_bias_
        )
        l_self_modules_layer_modules_10_modules_rel_attn_parameters_seg_embed_ = (
            L_self_modules_layer_modules_10_modules_rel_attn_parameters_seg_embed_
        )
        l_self_modules_layer_modules_10_modules_rel_attn_parameters_o_ = (
            L_self_modules_layer_modules_10_modules_rel_attn_parameters_o_
        )
        l_self_modules_layer_modules_10_modules_rel_attn_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_10_modules_rel_attn_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_10_modules_rel_attn_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_10_modules_rel_attn_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_10_modules_ff_modules_layer_1_parameters_weight_ = L_self_modules_layer_modules_10_modules_ff_modules_layer_1_parameters_weight_
        l_self_modules_layer_modules_10_modules_ff_modules_layer_1_parameters_bias_ = (
            L_self_modules_layer_modules_10_modules_ff_modules_layer_1_parameters_bias_
        )
        l_self_modules_layer_modules_10_modules_ff_modules_layer_2_parameters_weight_ = L_self_modules_layer_modules_10_modules_ff_modules_layer_2_parameters_weight_
        l_self_modules_layer_modules_10_modules_ff_modules_layer_2_parameters_bias_ = (
            L_self_modules_layer_modules_10_modules_ff_modules_layer_2_parameters_bias_
        )
        l_self_modules_layer_modules_10_modules_ff_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_10_modules_ff_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_10_modules_ff_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_10_modules_ff_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_11_modules_rel_attn_parameters_q_ = (
            L_self_modules_layer_modules_11_modules_rel_attn_parameters_q_
        )
        l_self_modules_layer_modules_11_modules_rel_attn_parameters_k_ = (
            L_self_modules_layer_modules_11_modules_rel_attn_parameters_k_
        )
        l_self_modules_layer_modules_11_modules_rel_attn_parameters_v_ = (
            L_self_modules_layer_modules_11_modules_rel_attn_parameters_v_
        )
        l_self_modules_layer_modules_11_modules_rel_attn_parameters_r_ = (
            L_self_modules_layer_modules_11_modules_rel_attn_parameters_r_
        )
        l_self_modules_layer_modules_11_modules_rel_attn_parameters_r_w_bias_ = (
            L_self_modules_layer_modules_11_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_11_modules_rel_attn_parameters_r_r_bias_ = (
            L_self_modules_layer_modules_11_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_11_modules_rel_attn_parameters_r_s_bias_ = (
            L_self_modules_layer_modules_11_modules_rel_attn_parameters_r_s_bias_
        )
        l_self_modules_layer_modules_11_modules_rel_attn_parameters_seg_embed_ = (
            L_self_modules_layer_modules_11_modules_rel_attn_parameters_seg_embed_
        )
        l_self_modules_layer_modules_11_modules_rel_attn_parameters_o_ = (
            L_self_modules_layer_modules_11_modules_rel_attn_parameters_o_
        )
        l_self_modules_layer_modules_11_modules_rel_attn_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_11_modules_rel_attn_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_11_modules_rel_attn_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_11_modules_rel_attn_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_11_modules_ff_modules_layer_1_parameters_weight_ = L_self_modules_layer_modules_11_modules_ff_modules_layer_1_parameters_weight_
        l_self_modules_layer_modules_11_modules_ff_modules_layer_1_parameters_bias_ = (
            L_self_modules_layer_modules_11_modules_ff_modules_layer_1_parameters_bias_
        )
        l_self_modules_layer_modules_11_modules_ff_modules_layer_2_parameters_weight_ = L_self_modules_layer_modules_11_modules_ff_modules_layer_2_parameters_weight_
        l_self_modules_layer_modules_11_modules_ff_modules_layer_2_parameters_bias_ = (
            L_self_modules_layer_modules_11_modules_ff_modules_layer_2_parameters_bias_
        )
        l_self_modules_layer_modules_11_modules_ff_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_11_modules_ff_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_11_modules_ff_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_11_modules_ff_modules_layer_norm_parameters_bias_
        transpose = l_input_ids_.transpose(0, 1)
        l_input_ids_ = None
        input_ids = transpose.contiguous()
        transpose = None
        transpose_1 = l_token_type_ids_.transpose(0, 1)
        l_token_type_ids_ = None
        token_type_ids = transpose_1.contiguous()
        transpose_1 = None
        transpose_2 = l_attention_mask_.transpose(0, 1)
        l_attention_mask_ = None
        attention_mask = transpose_2.contiguous()
        transpose_2 = None
        input_mask = 1.0 - attention_mask
        attention_mask = None
        data_mask = input_mask[None]
        input_mask = None
        attn_mask = data_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        data_mask = None
        gt = attn_mask > 0
        attn_mask = None
        attn_mask_1 = gt.to(torch.float32)
        gt = None
        eye = torch.eye(13)
        to_1 = eye.to(attn_mask_1)
        eye = None
        non_tgt_mask = -to_1
        to_1 = None
        getitem_2 = non_tgt_mask[
            (slice(None, None, None), slice(None, None, None), None, None)
        ]
        non_tgt_mask = None
        add = attn_mask_1 + getitem_2
        getitem_2 = None
        gt_1 = add > 0
        add = None
        non_tgt_mask_1 = gt_1.to(attn_mask_1)
        gt_1 = attn_mask_1 = None
        word_emb_k = torch.nn.functional.embedding(
            input_ids,
            l_self_modules_word_embedding_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        input_ids = l_self_modules_word_embedding_parameters_weight_ = None
        output_h = torch.nn.functional.dropout(word_emb_k, 0.1, False, False)
        word_emb_k = None
        getitem_3 = token_type_ids[(slice(None, None, None), None)]
        getitem_4 = token_type_ids[(None, slice(None, None, None))]
        token_type_ids = None
        ne = getitem_3 != getitem_4
        getitem_3 = getitem_4 = None
        seg_mat = ne.long()
        ne = None
        one_hot = torch._C._nn.one_hot(seg_mat, num_classes=2)
        seg_mat = None
        seg_mat_1 = one_hot.to(torch.float32)
        one_hot = None
        arange = torch.arange(0, 768, 2.0, dtype=torch.int64)
        freq_seq = arange.float()
        arange = None
        truediv = freq_seq / 768
        freq_seq = None
        pow_1 = torch.pow(10000, truediv)
        truediv = None
        inv_freq = 1 / pow_1
        pow_1 = None
        arange_1 = torch.arange(13, -13, -1.0, dtype=torch.int64)
        fwd_pos_seq = arange_1.float()
        arange_1 = None
        sinusoid_inp = torch.functional.einsum("i,d->id", fwd_pos_seq, inv_freq)
        fwd_pos_seq = inv_freq = None
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        sinusoid_inp = None
        pos_emb = torch.cat([sin, cos], dim=-1)
        sin = cos = None
        pos_emb_1 = pos_emb[(slice(None, None, None), None, slice(None, None, None))]
        pos_emb = None
        pos_emb_2 = pos_emb_1.expand(-1, 1, -1)
        pos_emb_1 = None
        pos_emb_3 = pos_emb_2.to(device(type="cuda", index=0))
        pos_emb_2 = None
        pos_emb_4 = torch.nn.functional.dropout(pos_emb_3, 0.1, False, False)
        pos_emb_3 = None
        new_mem = output_h[slice(0, None, None)]
        detach = new_mem.detach()
        new_mem = None
        q_head_h = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_h,
            l_self_modules_layer_modules_0_modules_rel_attn_parameters_q_,
        )
        l_self_modules_layer_modules_0_modules_rel_attn_parameters_q_ = None
        k_head_h = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_h,
            l_self_modules_layer_modules_0_modules_rel_attn_parameters_k_,
        )
        l_self_modules_layer_modules_0_modules_rel_attn_parameters_k_ = None
        v_head_h = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_h,
            l_self_modules_layer_modules_0_modules_rel_attn_parameters_v_,
        )
        l_self_modules_layer_modules_0_modules_rel_attn_parameters_v_ = None
        type_1 = pos_emb_4.type(torch.float32)
        k_head_r = torch.functional.einsum(
            "ibh,hnd->ibnd",
            type_1,
            l_self_modules_layer_modules_0_modules_rel_attn_parameters_r_,
        )
        type_1 = l_self_modules_layer_modules_0_modules_rel_attn_parameters_r_ = None
        add_1 = (
            q_head_h
            + l_self_modules_layer_modules_0_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_0_modules_rel_attn_parameters_r_w_bias_ = None
        ac = torch.functional.einsum("ibnd,jbnd->bnij", add_1, k_head_h)
        add_1 = k_head_h = None
        add_2 = (
            q_head_h
            + l_self_modules_layer_modules_0_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_0_modules_rel_attn_parameters_r_r_bias_ = None
        bd = torch.functional.einsum("ibnd,jbnd->bnij", add_2, k_head_r)
        add_2 = k_head_r = None
        x = bd.reshape(1, 12, 26, 13)
        bd = None
        x_1 = x[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        x = None
        x_2 = x_1.reshape(1, 12, 13, 25)
        x_1 = None
        arange_2 = torch.arange(
            13, device=device(type="cuda", index=0), dtype=torch.int64
        )
        x_3 = torch.index_select(x_2, 3, arange_2)
        x_2 = arange_2 = None
        add_3 = (
            q_head_h
            + l_self_modules_layer_modules_0_modules_rel_attn_parameters_r_s_bias_
        )
        q_head_h = (
            l_self_modules_layer_modules_0_modules_rel_attn_parameters_r_s_bias_
        ) = None
        ef = torch.functional.einsum(
            "ibnd,snd->ibns",
            add_3,
            l_self_modules_layer_modules_0_modules_rel_attn_parameters_seg_embed_,
        )
        add_3 = (
            l_self_modules_layer_modules_0_modules_rel_attn_parameters_seg_embed_
        ) = None
        ef_1 = torch.functional.einsum("ijbs,ibns->bnij", seg_mat_1, ef)
        ef = None
        add_4 = ac + x_3
        ac = x_3 = None
        add_5 = add_4 + ef_1
        add_4 = ef_1 = None
        attn_score = add_5 * 0.125
        add_5 = None
        einsum_9 = torch.functional.einsum("ijbn->bnij", non_tgt_mask_1)
        mul_1 = 1e30 * einsum_9
        einsum_9 = None
        attn_score_1 = attn_score - mul_1
        attn_score = mul_1 = None
        attn_prob = torch.nn.functional.softmax(attn_score_1, dim=3)
        attn_score_1 = None
        attn_prob_1 = torch.nn.functional.dropout(attn_prob, 0.1, False, False)
        attn_prob = None
        attn_vec = torch.functional.einsum("bnij,jbnd->ibnd", attn_prob_1, v_head_h)
        attn_prob_1 = v_head_h = None
        attn_out = torch.functional.einsum(
            "ibnd,hnd->ibh",
            attn_vec,
            l_self_modules_layer_modules_0_modules_rel_attn_parameters_o_,
        )
        attn_vec = l_self_modules_layer_modules_0_modules_rel_attn_parameters_o_ = None
        attn_out_1 = torch.nn.functional.dropout(attn_out, 0.1, False, False)
        attn_out = None
        attn_out_2 = attn_out_1 + output_h
        attn_out_1 = output_h = None
        output = torch.nn.functional.layer_norm(
            attn_out_2,
            (768,),
            l_self_modules_layer_modules_0_modules_rel_attn_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_0_modules_rel_attn_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        attn_out_2 = l_self_modules_layer_modules_0_modules_rel_attn_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_0_modules_rel_attn_modules_layer_norm_parameters_bias_ = (None)
        output_1 = torch._C._nn.linear(
            output,
            l_self_modules_layer_modules_0_modules_ff_modules_layer_1_parameters_weight_,
            l_self_modules_layer_modules_0_modules_ff_modules_layer_1_parameters_bias_,
        )
        l_self_modules_layer_modules_0_modules_ff_modules_layer_1_parameters_weight_ = (
            l_self_modules_layer_modules_0_modules_ff_modules_layer_1_parameters_bias_
        ) = None
        output_2 = torch._C._nn.gelu(output_1)
        output_1 = None
        output_3 = torch.nn.functional.dropout(output_2, 0.1, False, False)
        output_2 = None
        output_4 = torch._C._nn.linear(
            output_3,
            l_self_modules_layer_modules_0_modules_ff_modules_layer_2_parameters_weight_,
            l_self_modules_layer_modules_0_modules_ff_modules_layer_2_parameters_bias_,
        )
        output_3 = (
            l_self_modules_layer_modules_0_modules_ff_modules_layer_2_parameters_weight_
        ) = (
            l_self_modules_layer_modules_0_modules_ff_modules_layer_2_parameters_bias_
        ) = None
        output_5 = torch.nn.functional.dropout(output_4, 0.1, False, False)
        output_4 = None
        add_7 = output_5 + output
        output_5 = output = None
        output_6 = torch.nn.functional.layer_norm(
            add_7,
            (768,),
            l_self_modules_layer_modules_0_modules_ff_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_0_modules_ff_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_7 = l_self_modules_layer_modules_0_modules_ff_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_0_modules_ff_modules_layer_norm_parameters_bias_ = (None)
        new_mem_1 = output_6[slice(0, None, None)]
        detach_1 = new_mem_1.detach()
        new_mem_1 = None
        q_head_h_1 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_6,
            l_self_modules_layer_modules_1_modules_rel_attn_parameters_q_,
        )
        l_self_modules_layer_modules_1_modules_rel_attn_parameters_q_ = None
        k_head_h_1 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_6,
            l_self_modules_layer_modules_1_modules_rel_attn_parameters_k_,
        )
        l_self_modules_layer_modules_1_modules_rel_attn_parameters_k_ = None
        v_head_h_1 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_6,
            l_self_modules_layer_modules_1_modules_rel_attn_parameters_v_,
        )
        l_self_modules_layer_modules_1_modules_rel_attn_parameters_v_ = None
        type_2 = pos_emb_4.type(torch.float32)
        k_head_r_1 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            type_2,
            l_self_modules_layer_modules_1_modules_rel_attn_parameters_r_,
        )
        type_2 = l_self_modules_layer_modules_1_modules_rel_attn_parameters_r_ = None
        add_8 = (
            q_head_h_1
            + l_self_modules_layer_modules_1_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_1_modules_rel_attn_parameters_r_w_bias_ = None
        ac_1 = torch.functional.einsum("ibnd,jbnd->bnij", add_8, k_head_h_1)
        add_8 = k_head_h_1 = None
        add_9 = (
            q_head_h_1
            + l_self_modules_layer_modules_1_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_1_modules_rel_attn_parameters_r_r_bias_ = None
        bd_1 = torch.functional.einsum("ibnd,jbnd->bnij", add_9, k_head_r_1)
        add_9 = k_head_r_1 = None
        x_4 = bd_1.reshape(1, 12, 26, 13)
        bd_1 = None
        x_5 = x_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        x_4 = None
        x_6 = x_5.reshape(1, 12, 13, 25)
        x_5 = None
        arange_3 = torch.arange(
            13, device=device(type="cuda", index=0), dtype=torch.int64
        )
        x_7 = torch.index_select(x_6, 3, arange_3)
        x_6 = arange_3 = None
        add_10 = (
            q_head_h_1
            + l_self_modules_layer_modules_1_modules_rel_attn_parameters_r_s_bias_
        )
        q_head_h_1 = (
            l_self_modules_layer_modules_1_modules_rel_attn_parameters_r_s_bias_
        ) = None
        ef_2 = torch.functional.einsum(
            "ibnd,snd->ibns",
            add_10,
            l_self_modules_layer_modules_1_modules_rel_attn_parameters_seg_embed_,
        )
        add_10 = (
            l_self_modules_layer_modules_1_modules_rel_attn_parameters_seg_embed_
        ) = None
        ef_3 = torch.functional.einsum("ijbs,ibns->bnij", seg_mat_1, ef_2)
        ef_2 = None
        add_11 = ac_1 + x_7
        ac_1 = x_7 = None
        add_12 = add_11 + ef_3
        add_11 = ef_3 = None
        attn_score_2 = add_12 * 0.125
        add_12 = None
        einsum_20 = torch.functional.einsum("ijbn->bnij", non_tgt_mask_1)
        mul_3 = 1e30 * einsum_20
        einsum_20 = None
        attn_score_3 = attn_score_2 - mul_3
        attn_score_2 = mul_3 = None
        attn_prob_2 = torch.nn.functional.softmax(attn_score_3, dim=3)
        attn_score_3 = None
        attn_prob_3 = torch.nn.functional.dropout(attn_prob_2, 0.1, False, False)
        attn_prob_2 = None
        attn_vec_1 = torch.functional.einsum("bnij,jbnd->ibnd", attn_prob_3, v_head_h_1)
        attn_prob_3 = v_head_h_1 = None
        attn_out_3 = torch.functional.einsum(
            "ibnd,hnd->ibh",
            attn_vec_1,
            l_self_modules_layer_modules_1_modules_rel_attn_parameters_o_,
        )
        attn_vec_1 = (
            l_self_modules_layer_modules_1_modules_rel_attn_parameters_o_
        ) = None
        attn_out_4 = torch.nn.functional.dropout(attn_out_3, 0.1, False, False)
        attn_out_3 = None
        attn_out_5 = attn_out_4 + output_6
        attn_out_4 = output_6 = None
        output_7 = torch.nn.functional.layer_norm(
            attn_out_5,
            (768,),
            l_self_modules_layer_modules_1_modules_rel_attn_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_1_modules_rel_attn_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        attn_out_5 = l_self_modules_layer_modules_1_modules_rel_attn_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_1_modules_rel_attn_modules_layer_norm_parameters_bias_ = (None)
        output_8 = torch._C._nn.linear(
            output_7,
            l_self_modules_layer_modules_1_modules_ff_modules_layer_1_parameters_weight_,
            l_self_modules_layer_modules_1_modules_ff_modules_layer_1_parameters_bias_,
        )
        l_self_modules_layer_modules_1_modules_ff_modules_layer_1_parameters_weight_ = (
            l_self_modules_layer_modules_1_modules_ff_modules_layer_1_parameters_bias_
        ) = None
        output_9 = torch._C._nn.gelu(output_8)
        output_8 = None
        output_10 = torch.nn.functional.dropout(output_9, 0.1, False, False)
        output_9 = None
        output_11 = torch._C._nn.linear(
            output_10,
            l_self_modules_layer_modules_1_modules_ff_modules_layer_2_parameters_weight_,
            l_self_modules_layer_modules_1_modules_ff_modules_layer_2_parameters_bias_,
        )
        output_10 = (
            l_self_modules_layer_modules_1_modules_ff_modules_layer_2_parameters_weight_
        ) = (
            l_self_modules_layer_modules_1_modules_ff_modules_layer_2_parameters_bias_
        ) = None
        output_12 = torch.nn.functional.dropout(output_11, 0.1, False, False)
        output_11 = None
        add_14 = output_12 + output_7
        output_12 = output_7 = None
        output_13 = torch.nn.functional.layer_norm(
            add_14,
            (768,),
            l_self_modules_layer_modules_1_modules_ff_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_1_modules_ff_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_14 = l_self_modules_layer_modules_1_modules_ff_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_1_modules_ff_modules_layer_norm_parameters_bias_ = (None)
        new_mem_2 = output_13[slice(0, None, None)]
        detach_2 = new_mem_2.detach()
        new_mem_2 = None
        q_head_h_2 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_13,
            l_self_modules_layer_modules_2_modules_rel_attn_parameters_q_,
        )
        l_self_modules_layer_modules_2_modules_rel_attn_parameters_q_ = None
        k_head_h_2 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_13,
            l_self_modules_layer_modules_2_modules_rel_attn_parameters_k_,
        )
        l_self_modules_layer_modules_2_modules_rel_attn_parameters_k_ = None
        v_head_h_2 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_13,
            l_self_modules_layer_modules_2_modules_rel_attn_parameters_v_,
        )
        l_self_modules_layer_modules_2_modules_rel_attn_parameters_v_ = None
        type_3 = pos_emb_4.type(torch.float32)
        k_head_r_2 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            type_3,
            l_self_modules_layer_modules_2_modules_rel_attn_parameters_r_,
        )
        type_3 = l_self_modules_layer_modules_2_modules_rel_attn_parameters_r_ = None
        add_15 = (
            q_head_h_2
            + l_self_modules_layer_modules_2_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_2_modules_rel_attn_parameters_r_w_bias_ = None
        ac_2 = torch.functional.einsum("ibnd,jbnd->bnij", add_15, k_head_h_2)
        add_15 = k_head_h_2 = None
        add_16 = (
            q_head_h_2
            + l_self_modules_layer_modules_2_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_2_modules_rel_attn_parameters_r_r_bias_ = None
        bd_2 = torch.functional.einsum("ibnd,jbnd->bnij", add_16, k_head_r_2)
        add_16 = k_head_r_2 = None
        x_8 = bd_2.reshape(1, 12, 26, 13)
        bd_2 = None
        x_9 = x_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        x_8 = None
        x_10 = x_9.reshape(1, 12, 13, 25)
        x_9 = None
        arange_4 = torch.arange(
            13, device=device(type="cuda", index=0), dtype=torch.int64
        )
        x_11 = torch.index_select(x_10, 3, arange_4)
        x_10 = arange_4 = None
        add_17 = (
            q_head_h_2
            + l_self_modules_layer_modules_2_modules_rel_attn_parameters_r_s_bias_
        )
        q_head_h_2 = (
            l_self_modules_layer_modules_2_modules_rel_attn_parameters_r_s_bias_
        ) = None
        ef_4 = torch.functional.einsum(
            "ibnd,snd->ibns",
            add_17,
            l_self_modules_layer_modules_2_modules_rel_attn_parameters_seg_embed_,
        )
        add_17 = (
            l_self_modules_layer_modules_2_modules_rel_attn_parameters_seg_embed_
        ) = None
        ef_5 = torch.functional.einsum("ijbs,ibns->bnij", seg_mat_1, ef_4)
        ef_4 = None
        add_18 = ac_2 + x_11
        ac_2 = x_11 = None
        add_19 = add_18 + ef_5
        add_18 = ef_5 = None
        attn_score_4 = add_19 * 0.125
        add_19 = None
        einsum_31 = torch.functional.einsum("ijbn->bnij", non_tgt_mask_1)
        mul_5 = 1e30 * einsum_31
        einsum_31 = None
        attn_score_5 = attn_score_4 - mul_5
        attn_score_4 = mul_5 = None
        attn_prob_4 = torch.nn.functional.softmax(attn_score_5, dim=3)
        attn_score_5 = None
        attn_prob_5 = torch.nn.functional.dropout(attn_prob_4, 0.1, False, False)
        attn_prob_4 = None
        attn_vec_2 = torch.functional.einsum("bnij,jbnd->ibnd", attn_prob_5, v_head_h_2)
        attn_prob_5 = v_head_h_2 = None
        attn_out_6 = torch.functional.einsum(
            "ibnd,hnd->ibh",
            attn_vec_2,
            l_self_modules_layer_modules_2_modules_rel_attn_parameters_o_,
        )
        attn_vec_2 = (
            l_self_modules_layer_modules_2_modules_rel_attn_parameters_o_
        ) = None
        attn_out_7 = torch.nn.functional.dropout(attn_out_6, 0.1, False, False)
        attn_out_6 = None
        attn_out_8 = attn_out_7 + output_13
        attn_out_7 = output_13 = None
        output_14 = torch.nn.functional.layer_norm(
            attn_out_8,
            (768,),
            l_self_modules_layer_modules_2_modules_rel_attn_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_2_modules_rel_attn_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        attn_out_8 = l_self_modules_layer_modules_2_modules_rel_attn_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_2_modules_rel_attn_modules_layer_norm_parameters_bias_ = (None)
        output_15 = torch._C._nn.linear(
            output_14,
            l_self_modules_layer_modules_2_modules_ff_modules_layer_1_parameters_weight_,
            l_self_modules_layer_modules_2_modules_ff_modules_layer_1_parameters_bias_,
        )
        l_self_modules_layer_modules_2_modules_ff_modules_layer_1_parameters_weight_ = (
            l_self_modules_layer_modules_2_modules_ff_modules_layer_1_parameters_bias_
        ) = None
        output_16 = torch._C._nn.gelu(output_15)
        output_15 = None
        output_17 = torch.nn.functional.dropout(output_16, 0.1, False, False)
        output_16 = None
        output_18 = torch._C._nn.linear(
            output_17,
            l_self_modules_layer_modules_2_modules_ff_modules_layer_2_parameters_weight_,
            l_self_modules_layer_modules_2_modules_ff_modules_layer_2_parameters_bias_,
        )
        output_17 = (
            l_self_modules_layer_modules_2_modules_ff_modules_layer_2_parameters_weight_
        ) = (
            l_self_modules_layer_modules_2_modules_ff_modules_layer_2_parameters_bias_
        ) = None
        output_19 = torch.nn.functional.dropout(output_18, 0.1, False, False)
        output_18 = None
        add_21 = output_19 + output_14
        output_19 = output_14 = None
        output_20 = torch.nn.functional.layer_norm(
            add_21,
            (768,),
            l_self_modules_layer_modules_2_modules_ff_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_2_modules_ff_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_21 = l_self_modules_layer_modules_2_modules_ff_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_2_modules_ff_modules_layer_norm_parameters_bias_ = (None)
        new_mem_3 = output_20[slice(0, None, None)]
        detach_3 = new_mem_3.detach()
        new_mem_3 = None
        q_head_h_3 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_20,
            l_self_modules_layer_modules_3_modules_rel_attn_parameters_q_,
        )
        l_self_modules_layer_modules_3_modules_rel_attn_parameters_q_ = None
        k_head_h_3 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_20,
            l_self_modules_layer_modules_3_modules_rel_attn_parameters_k_,
        )
        l_self_modules_layer_modules_3_modules_rel_attn_parameters_k_ = None
        v_head_h_3 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_20,
            l_self_modules_layer_modules_3_modules_rel_attn_parameters_v_,
        )
        l_self_modules_layer_modules_3_modules_rel_attn_parameters_v_ = None
        type_4 = pos_emb_4.type(torch.float32)
        k_head_r_3 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            type_4,
            l_self_modules_layer_modules_3_modules_rel_attn_parameters_r_,
        )
        type_4 = l_self_modules_layer_modules_3_modules_rel_attn_parameters_r_ = None
        add_22 = (
            q_head_h_3
            + l_self_modules_layer_modules_3_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_3_modules_rel_attn_parameters_r_w_bias_ = None
        ac_3 = torch.functional.einsum("ibnd,jbnd->bnij", add_22, k_head_h_3)
        add_22 = k_head_h_3 = None
        add_23 = (
            q_head_h_3
            + l_self_modules_layer_modules_3_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_3_modules_rel_attn_parameters_r_r_bias_ = None
        bd_3 = torch.functional.einsum("ibnd,jbnd->bnij", add_23, k_head_r_3)
        add_23 = k_head_r_3 = None
        x_12 = bd_3.reshape(1, 12, 26, 13)
        bd_3 = None
        x_13 = x_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        x_12 = None
        x_14 = x_13.reshape(1, 12, 13, 25)
        x_13 = None
        arange_5 = torch.arange(
            13, device=device(type="cuda", index=0), dtype=torch.int64
        )
        x_15 = torch.index_select(x_14, 3, arange_5)
        x_14 = arange_5 = None
        add_24 = (
            q_head_h_3
            + l_self_modules_layer_modules_3_modules_rel_attn_parameters_r_s_bias_
        )
        q_head_h_3 = (
            l_self_modules_layer_modules_3_modules_rel_attn_parameters_r_s_bias_
        ) = None
        ef_6 = torch.functional.einsum(
            "ibnd,snd->ibns",
            add_24,
            l_self_modules_layer_modules_3_modules_rel_attn_parameters_seg_embed_,
        )
        add_24 = (
            l_self_modules_layer_modules_3_modules_rel_attn_parameters_seg_embed_
        ) = None
        ef_7 = torch.functional.einsum("ijbs,ibns->bnij", seg_mat_1, ef_6)
        ef_6 = None
        add_25 = ac_3 + x_15
        ac_3 = x_15 = None
        add_26 = add_25 + ef_7
        add_25 = ef_7 = None
        attn_score_6 = add_26 * 0.125
        add_26 = None
        einsum_42 = torch.functional.einsum("ijbn->bnij", non_tgt_mask_1)
        mul_7 = 1e30 * einsum_42
        einsum_42 = None
        attn_score_7 = attn_score_6 - mul_7
        attn_score_6 = mul_7 = None
        attn_prob_6 = torch.nn.functional.softmax(attn_score_7, dim=3)
        attn_score_7 = None
        attn_prob_7 = torch.nn.functional.dropout(attn_prob_6, 0.1, False, False)
        attn_prob_6 = None
        attn_vec_3 = torch.functional.einsum("bnij,jbnd->ibnd", attn_prob_7, v_head_h_3)
        attn_prob_7 = v_head_h_3 = None
        attn_out_9 = torch.functional.einsum(
            "ibnd,hnd->ibh",
            attn_vec_3,
            l_self_modules_layer_modules_3_modules_rel_attn_parameters_o_,
        )
        attn_vec_3 = (
            l_self_modules_layer_modules_3_modules_rel_attn_parameters_o_
        ) = None
        attn_out_10 = torch.nn.functional.dropout(attn_out_9, 0.1, False, False)
        attn_out_9 = None
        attn_out_11 = attn_out_10 + output_20
        attn_out_10 = output_20 = None
        output_21 = torch.nn.functional.layer_norm(
            attn_out_11,
            (768,),
            l_self_modules_layer_modules_3_modules_rel_attn_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_3_modules_rel_attn_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        attn_out_11 = l_self_modules_layer_modules_3_modules_rel_attn_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_3_modules_rel_attn_modules_layer_norm_parameters_bias_ = (None)
        output_22 = torch._C._nn.linear(
            output_21,
            l_self_modules_layer_modules_3_modules_ff_modules_layer_1_parameters_weight_,
            l_self_modules_layer_modules_3_modules_ff_modules_layer_1_parameters_bias_,
        )
        l_self_modules_layer_modules_3_modules_ff_modules_layer_1_parameters_weight_ = (
            l_self_modules_layer_modules_3_modules_ff_modules_layer_1_parameters_bias_
        ) = None
        output_23 = torch._C._nn.gelu(output_22)
        output_22 = None
        output_24 = torch.nn.functional.dropout(output_23, 0.1, False, False)
        output_23 = None
        output_25 = torch._C._nn.linear(
            output_24,
            l_self_modules_layer_modules_3_modules_ff_modules_layer_2_parameters_weight_,
            l_self_modules_layer_modules_3_modules_ff_modules_layer_2_parameters_bias_,
        )
        output_24 = (
            l_self_modules_layer_modules_3_modules_ff_modules_layer_2_parameters_weight_
        ) = (
            l_self_modules_layer_modules_3_modules_ff_modules_layer_2_parameters_bias_
        ) = None
        output_26 = torch.nn.functional.dropout(output_25, 0.1, False, False)
        output_25 = None
        add_28 = output_26 + output_21
        output_26 = output_21 = None
        output_27 = torch.nn.functional.layer_norm(
            add_28,
            (768,),
            l_self_modules_layer_modules_3_modules_ff_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_3_modules_ff_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_28 = l_self_modules_layer_modules_3_modules_ff_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_3_modules_ff_modules_layer_norm_parameters_bias_ = (None)
        new_mem_4 = output_27[slice(0, None, None)]
        detach_4 = new_mem_4.detach()
        new_mem_4 = None
        q_head_h_4 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_27,
            l_self_modules_layer_modules_4_modules_rel_attn_parameters_q_,
        )
        l_self_modules_layer_modules_4_modules_rel_attn_parameters_q_ = None
        k_head_h_4 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_27,
            l_self_modules_layer_modules_4_modules_rel_attn_parameters_k_,
        )
        l_self_modules_layer_modules_4_modules_rel_attn_parameters_k_ = None
        v_head_h_4 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_27,
            l_self_modules_layer_modules_4_modules_rel_attn_parameters_v_,
        )
        l_self_modules_layer_modules_4_modules_rel_attn_parameters_v_ = None
        type_5 = pos_emb_4.type(torch.float32)
        k_head_r_4 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            type_5,
            l_self_modules_layer_modules_4_modules_rel_attn_parameters_r_,
        )
        type_5 = l_self_modules_layer_modules_4_modules_rel_attn_parameters_r_ = None
        add_29 = (
            q_head_h_4
            + l_self_modules_layer_modules_4_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_4_modules_rel_attn_parameters_r_w_bias_ = None
        ac_4 = torch.functional.einsum("ibnd,jbnd->bnij", add_29, k_head_h_4)
        add_29 = k_head_h_4 = None
        add_30 = (
            q_head_h_4
            + l_self_modules_layer_modules_4_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_4_modules_rel_attn_parameters_r_r_bias_ = None
        bd_4 = torch.functional.einsum("ibnd,jbnd->bnij", add_30, k_head_r_4)
        add_30 = k_head_r_4 = None
        x_16 = bd_4.reshape(1, 12, 26, 13)
        bd_4 = None
        x_17 = x_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        x_16 = None
        x_18 = x_17.reshape(1, 12, 13, 25)
        x_17 = None
        arange_6 = torch.arange(
            13, device=device(type="cuda", index=0), dtype=torch.int64
        )
        x_19 = torch.index_select(x_18, 3, arange_6)
        x_18 = arange_6 = None
        add_31 = (
            q_head_h_4
            + l_self_modules_layer_modules_4_modules_rel_attn_parameters_r_s_bias_
        )
        q_head_h_4 = (
            l_self_modules_layer_modules_4_modules_rel_attn_parameters_r_s_bias_
        ) = None
        ef_8 = torch.functional.einsum(
            "ibnd,snd->ibns",
            add_31,
            l_self_modules_layer_modules_4_modules_rel_attn_parameters_seg_embed_,
        )
        add_31 = (
            l_self_modules_layer_modules_4_modules_rel_attn_parameters_seg_embed_
        ) = None
        ef_9 = torch.functional.einsum("ijbs,ibns->bnij", seg_mat_1, ef_8)
        ef_8 = None
        add_32 = ac_4 + x_19
        ac_4 = x_19 = None
        add_33 = add_32 + ef_9
        add_32 = ef_9 = None
        attn_score_8 = add_33 * 0.125
        add_33 = None
        einsum_53 = torch.functional.einsum("ijbn->bnij", non_tgt_mask_1)
        mul_9 = 1e30 * einsum_53
        einsum_53 = None
        attn_score_9 = attn_score_8 - mul_9
        attn_score_8 = mul_9 = None
        attn_prob_8 = torch.nn.functional.softmax(attn_score_9, dim=3)
        attn_score_9 = None
        attn_prob_9 = torch.nn.functional.dropout(attn_prob_8, 0.1, False, False)
        attn_prob_8 = None
        attn_vec_4 = torch.functional.einsum("bnij,jbnd->ibnd", attn_prob_9, v_head_h_4)
        attn_prob_9 = v_head_h_4 = None
        attn_out_12 = torch.functional.einsum(
            "ibnd,hnd->ibh",
            attn_vec_4,
            l_self_modules_layer_modules_4_modules_rel_attn_parameters_o_,
        )
        attn_vec_4 = (
            l_self_modules_layer_modules_4_modules_rel_attn_parameters_o_
        ) = None
        attn_out_13 = torch.nn.functional.dropout(attn_out_12, 0.1, False, False)
        attn_out_12 = None
        attn_out_14 = attn_out_13 + output_27
        attn_out_13 = output_27 = None
        output_28 = torch.nn.functional.layer_norm(
            attn_out_14,
            (768,),
            l_self_modules_layer_modules_4_modules_rel_attn_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_4_modules_rel_attn_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        attn_out_14 = l_self_modules_layer_modules_4_modules_rel_attn_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_4_modules_rel_attn_modules_layer_norm_parameters_bias_ = (None)
        output_29 = torch._C._nn.linear(
            output_28,
            l_self_modules_layer_modules_4_modules_ff_modules_layer_1_parameters_weight_,
            l_self_modules_layer_modules_4_modules_ff_modules_layer_1_parameters_bias_,
        )
        l_self_modules_layer_modules_4_modules_ff_modules_layer_1_parameters_weight_ = (
            l_self_modules_layer_modules_4_modules_ff_modules_layer_1_parameters_bias_
        ) = None
        output_30 = torch._C._nn.gelu(output_29)
        output_29 = None
        output_31 = torch.nn.functional.dropout(output_30, 0.1, False, False)
        output_30 = None
        output_32 = torch._C._nn.linear(
            output_31,
            l_self_modules_layer_modules_4_modules_ff_modules_layer_2_parameters_weight_,
            l_self_modules_layer_modules_4_modules_ff_modules_layer_2_parameters_bias_,
        )
        output_31 = (
            l_self_modules_layer_modules_4_modules_ff_modules_layer_2_parameters_weight_
        ) = (
            l_self_modules_layer_modules_4_modules_ff_modules_layer_2_parameters_bias_
        ) = None
        output_33 = torch.nn.functional.dropout(output_32, 0.1, False, False)
        output_32 = None
        add_35 = output_33 + output_28
        output_33 = output_28 = None
        output_34 = torch.nn.functional.layer_norm(
            add_35,
            (768,),
            l_self_modules_layer_modules_4_modules_ff_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_4_modules_ff_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_35 = l_self_modules_layer_modules_4_modules_ff_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_4_modules_ff_modules_layer_norm_parameters_bias_ = (None)
        new_mem_5 = output_34[slice(0, None, None)]
        detach_5 = new_mem_5.detach()
        new_mem_5 = None
        q_head_h_5 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_34,
            l_self_modules_layer_modules_5_modules_rel_attn_parameters_q_,
        )
        l_self_modules_layer_modules_5_modules_rel_attn_parameters_q_ = None
        k_head_h_5 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_34,
            l_self_modules_layer_modules_5_modules_rel_attn_parameters_k_,
        )
        l_self_modules_layer_modules_5_modules_rel_attn_parameters_k_ = None
        v_head_h_5 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_34,
            l_self_modules_layer_modules_5_modules_rel_attn_parameters_v_,
        )
        l_self_modules_layer_modules_5_modules_rel_attn_parameters_v_ = None
        type_6 = pos_emb_4.type(torch.float32)
        k_head_r_5 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            type_6,
            l_self_modules_layer_modules_5_modules_rel_attn_parameters_r_,
        )
        type_6 = l_self_modules_layer_modules_5_modules_rel_attn_parameters_r_ = None
        add_36 = (
            q_head_h_5
            + l_self_modules_layer_modules_5_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_5_modules_rel_attn_parameters_r_w_bias_ = None
        ac_5 = torch.functional.einsum("ibnd,jbnd->bnij", add_36, k_head_h_5)
        add_36 = k_head_h_5 = None
        add_37 = (
            q_head_h_5
            + l_self_modules_layer_modules_5_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_5_modules_rel_attn_parameters_r_r_bias_ = None
        bd_5 = torch.functional.einsum("ibnd,jbnd->bnij", add_37, k_head_r_5)
        add_37 = k_head_r_5 = None
        x_20 = bd_5.reshape(1, 12, 26, 13)
        bd_5 = None
        x_21 = x_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        x_20 = None
        x_22 = x_21.reshape(1, 12, 13, 25)
        x_21 = None
        arange_7 = torch.arange(
            13, device=device(type="cuda", index=0), dtype=torch.int64
        )
        x_23 = torch.index_select(x_22, 3, arange_7)
        x_22 = arange_7 = None
        add_38 = (
            q_head_h_5
            + l_self_modules_layer_modules_5_modules_rel_attn_parameters_r_s_bias_
        )
        q_head_h_5 = (
            l_self_modules_layer_modules_5_modules_rel_attn_parameters_r_s_bias_
        ) = None
        ef_10 = torch.functional.einsum(
            "ibnd,snd->ibns",
            add_38,
            l_self_modules_layer_modules_5_modules_rel_attn_parameters_seg_embed_,
        )
        add_38 = (
            l_self_modules_layer_modules_5_modules_rel_attn_parameters_seg_embed_
        ) = None
        ef_11 = torch.functional.einsum("ijbs,ibns->bnij", seg_mat_1, ef_10)
        ef_10 = None
        add_39 = ac_5 + x_23
        ac_5 = x_23 = None
        add_40 = add_39 + ef_11
        add_39 = ef_11 = None
        attn_score_10 = add_40 * 0.125
        add_40 = None
        einsum_64 = torch.functional.einsum("ijbn->bnij", non_tgt_mask_1)
        mul_11 = 1e30 * einsum_64
        einsum_64 = None
        attn_score_11 = attn_score_10 - mul_11
        attn_score_10 = mul_11 = None
        attn_prob_10 = torch.nn.functional.softmax(attn_score_11, dim=3)
        attn_score_11 = None
        attn_prob_11 = torch.nn.functional.dropout(attn_prob_10, 0.1, False, False)
        attn_prob_10 = None
        attn_vec_5 = torch.functional.einsum(
            "bnij,jbnd->ibnd", attn_prob_11, v_head_h_5
        )
        attn_prob_11 = v_head_h_5 = None
        attn_out_15 = torch.functional.einsum(
            "ibnd,hnd->ibh",
            attn_vec_5,
            l_self_modules_layer_modules_5_modules_rel_attn_parameters_o_,
        )
        attn_vec_5 = (
            l_self_modules_layer_modules_5_modules_rel_attn_parameters_o_
        ) = None
        attn_out_16 = torch.nn.functional.dropout(attn_out_15, 0.1, False, False)
        attn_out_15 = None
        attn_out_17 = attn_out_16 + output_34
        attn_out_16 = output_34 = None
        output_35 = torch.nn.functional.layer_norm(
            attn_out_17,
            (768,),
            l_self_modules_layer_modules_5_modules_rel_attn_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_5_modules_rel_attn_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        attn_out_17 = l_self_modules_layer_modules_5_modules_rel_attn_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_5_modules_rel_attn_modules_layer_norm_parameters_bias_ = (None)
        output_36 = torch._C._nn.linear(
            output_35,
            l_self_modules_layer_modules_5_modules_ff_modules_layer_1_parameters_weight_,
            l_self_modules_layer_modules_5_modules_ff_modules_layer_1_parameters_bias_,
        )
        l_self_modules_layer_modules_5_modules_ff_modules_layer_1_parameters_weight_ = (
            l_self_modules_layer_modules_5_modules_ff_modules_layer_1_parameters_bias_
        ) = None
        output_37 = torch._C._nn.gelu(output_36)
        output_36 = None
        output_38 = torch.nn.functional.dropout(output_37, 0.1, False, False)
        output_37 = None
        output_39 = torch._C._nn.linear(
            output_38,
            l_self_modules_layer_modules_5_modules_ff_modules_layer_2_parameters_weight_,
            l_self_modules_layer_modules_5_modules_ff_modules_layer_2_parameters_bias_,
        )
        output_38 = (
            l_self_modules_layer_modules_5_modules_ff_modules_layer_2_parameters_weight_
        ) = (
            l_self_modules_layer_modules_5_modules_ff_modules_layer_2_parameters_bias_
        ) = None
        output_40 = torch.nn.functional.dropout(output_39, 0.1, False, False)
        output_39 = None
        add_42 = output_40 + output_35
        output_40 = output_35 = None
        output_41 = torch.nn.functional.layer_norm(
            add_42,
            (768,),
            l_self_modules_layer_modules_5_modules_ff_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_5_modules_ff_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_42 = l_self_modules_layer_modules_5_modules_ff_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_5_modules_ff_modules_layer_norm_parameters_bias_ = (None)
        new_mem_6 = output_41[slice(0, None, None)]
        detach_6 = new_mem_6.detach()
        new_mem_6 = None
        q_head_h_6 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_41,
            l_self_modules_layer_modules_6_modules_rel_attn_parameters_q_,
        )
        l_self_modules_layer_modules_6_modules_rel_attn_parameters_q_ = None
        k_head_h_6 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_41,
            l_self_modules_layer_modules_6_modules_rel_attn_parameters_k_,
        )
        l_self_modules_layer_modules_6_modules_rel_attn_parameters_k_ = None
        v_head_h_6 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_41,
            l_self_modules_layer_modules_6_modules_rel_attn_parameters_v_,
        )
        l_self_modules_layer_modules_6_modules_rel_attn_parameters_v_ = None
        type_7 = pos_emb_4.type(torch.float32)
        k_head_r_6 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            type_7,
            l_self_modules_layer_modules_6_modules_rel_attn_parameters_r_,
        )
        type_7 = l_self_modules_layer_modules_6_modules_rel_attn_parameters_r_ = None
        add_43 = (
            q_head_h_6
            + l_self_modules_layer_modules_6_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_6_modules_rel_attn_parameters_r_w_bias_ = None
        ac_6 = torch.functional.einsum("ibnd,jbnd->bnij", add_43, k_head_h_6)
        add_43 = k_head_h_6 = None
        add_44 = (
            q_head_h_6
            + l_self_modules_layer_modules_6_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_6_modules_rel_attn_parameters_r_r_bias_ = None
        bd_6 = torch.functional.einsum("ibnd,jbnd->bnij", add_44, k_head_r_6)
        add_44 = k_head_r_6 = None
        x_24 = bd_6.reshape(1, 12, 26, 13)
        bd_6 = None
        x_25 = x_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        x_24 = None
        x_26 = x_25.reshape(1, 12, 13, 25)
        x_25 = None
        arange_8 = torch.arange(
            13, device=device(type="cuda", index=0), dtype=torch.int64
        )
        x_27 = torch.index_select(x_26, 3, arange_8)
        x_26 = arange_8 = None
        add_45 = (
            q_head_h_6
            + l_self_modules_layer_modules_6_modules_rel_attn_parameters_r_s_bias_
        )
        q_head_h_6 = (
            l_self_modules_layer_modules_6_modules_rel_attn_parameters_r_s_bias_
        ) = None
        ef_12 = torch.functional.einsum(
            "ibnd,snd->ibns",
            add_45,
            l_self_modules_layer_modules_6_modules_rel_attn_parameters_seg_embed_,
        )
        add_45 = (
            l_self_modules_layer_modules_6_modules_rel_attn_parameters_seg_embed_
        ) = None
        ef_13 = torch.functional.einsum("ijbs,ibns->bnij", seg_mat_1, ef_12)
        ef_12 = None
        add_46 = ac_6 + x_27
        ac_6 = x_27 = None
        add_47 = add_46 + ef_13
        add_46 = ef_13 = None
        attn_score_12 = add_47 * 0.125
        add_47 = None
        einsum_75 = torch.functional.einsum("ijbn->bnij", non_tgt_mask_1)
        mul_13 = 1e30 * einsum_75
        einsum_75 = None
        attn_score_13 = attn_score_12 - mul_13
        attn_score_12 = mul_13 = None
        attn_prob_12 = torch.nn.functional.softmax(attn_score_13, dim=3)
        attn_score_13 = None
        attn_prob_13 = torch.nn.functional.dropout(attn_prob_12, 0.1, False, False)
        attn_prob_12 = None
        attn_vec_6 = torch.functional.einsum(
            "bnij,jbnd->ibnd", attn_prob_13, v_head_h_6
        )
        attn_prob_13 = v_head_h_6 = None
        attn_out_18 = torch.functional.einsum(
            "ibnd,hnd->ibh",
            attn_vec_6,
            l_self_modules_layer_modules_6_modules_rel_attn_parameters_o_,
        )
        attn_vec_6 = (
            l_self_modules_layer_modules_6_modules_rel_attn_parameters_o_
        ) = None
        attn_out_19 = torch.nn.functional.dropout(attn_out_18, 0.1, False, False)
        attn_out_18 = None
        attn_out_20 = attn_out_19 + output_41
        attn_out_19 = output_41 = None
        output_42 = torch.nn.functional.layer_norm(
            attn_out_20,
            (768,),
            l_self_modules_layer_modules_6_modules_rel_attn_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_6_modules_rel_attn_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        attn_out_20 = l_self_modules_layer_modules_6_modules_rel_attn_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_6_modules_rel_attn_modules_layer_norm_parameters_bias_ = (None)
        output_43 = torch._C._nn.linear(
            output_42,
            l_self_modules_layer_modules_6_modules_ff_modules_layer_1_parameters_weight_,
            l_self_modules_layer_modules_6_modules_ff_modules_layer_1_parameters_bias_,
        )
        l_self_modules_layer_modules_6_modules_ff_modules_layer_1_parameters_weight_ = (
            l_self_modules_layer_modules_6_modules_ff_modules_layer_1_parameters_bias_
        ) = None
        output_44 = torch._C._nn.gelu(output_43)
        output_43 = None
        output_45 = torch.nn.functional.dropout(output_44, 0.1, False, False)
        output_44 = None
        output_46 = torch._C._nn.linear(
            output_45,
            l_self_modules_layer_modules_6_modules_ff_modules_layer_2_parameters_weight_,
            l_self_modules_layer_modules_6_modules_ff_modules_layer_2_parameters_bias_,
        )
        output_45 = (
            l_self_modules_layer_modules_6_modules_ff_modules_layer_2_parameters_weight_
        ) = (
            l_self_modules_layer_modules_6_modules_ff_modules_layer_2_parameters_bias_
        ) = None
        output_47 = torch.nn.functional.dropout(output_46, 0.1, False, False)
        output_46 = None
        add_49 = output_47 + output_42
        output_47 = output_42 = None
        output_48 = torch.nn.functional.layer_norm(
            add_49,
            (768,),
            l_self_modules_layer_modules_6_modules_ff_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_6_modules_ff_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_49 = l_self_modules_layer_modules_6_modules_ff_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_6_modules_ff_modules_layer_norm_parameters_bias_ = (None)
        new_mem_7 = output_48[slice(0, None, None)]
        detach_7 = new_mem_7.detach()
        new_mem_7 = None
        q_head_h_7 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_48,
            l_self_modules_layer_modules_7_modules_rel_attn_parameters_q_,
        )
        l_self_modules_layer_modules_7_modules_rel_attn_parameters_q_ = None
        k_head_h_7 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_48,
            l_self_modules_layer_modules_7_modules_rel_attn_parameters_k_,
        )
        l_self_modules_layer_modules_7_modules_rel_attn_parameters_k_ = None
        v_head_h_7 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_48,
            l_self_modules_layer_modules_7_modules_rel_attn_parameters_v_,
        )
        l_self_modules_layer_modules_7_modules_rel_attn_parameters_v_ = None
        type_8 = pos_emb_4.type(torch.float32)
        k_head_r_7 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            type_8,
            l_self_modules_layer_modules_7_modules_rel_attn_parameters_r_,
        )
        type_8 = l_self_modules_layer_modules_7_modules_rel_attn_parameters_r_ = None
        add_50 = (
            q_head_h_7
            + l_self_modules_layer_modules_7_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_7_modules_rel_attn_parameters_r_w_bias_ = None
        ac_7 = torch.functional.einsum("ibnd,jbnd->bnij", add_50, k_head_h_7)
        add_50 = k_head_h_7 = None
        add_51 = (
            q_head_h_7
            + l_self_modules_layer_modules_7_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_7_modules_rel_attn_parameters_r_r_bias_ = None
        bd_7 = torch.functional.einsum("ibnd,jbnd->bnij", add_51, k_head_r_7)
        add_51 = k_head_r_7 = None
        x_28 = bd_7.reshape(1, 12, 26, 13)
        bd_7 = None
        x_29 = x_28[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        x_28 = None
        x_30 = x_29.reshape(1, 12, 13, 25)
        x_29 = None
        arange_9 = torch.arange(
            13, device=device(type="cuda", index=0), dtype=torch.int64
        )
        x_31 = torch.index_select(x_30, 3, arange_9)
        x_30 = arange_9 = None
        add_52 = (
            q_head_h_7
            + l_self_modules_layer_modules_7_modules_rel_attn_parameters_r_s_bias_
        )
        q_head_h_7 = (
            l_self_modules_layer_modules_7_modules_rel_attn_parameters_r_s_bias_
        ) = None
        ef_14 = torch.functional.einsum(
            "ibnd,snd->ibns",
            add_52,
            l_self_modules_layer_modules_7_modules_rel_attn_parameters_seg_embed_,
        )
        add_52 = (
            l_self_modules_layer_modules_7_modules_rel_attn_parameters_seg_embed_
        ) = None
        ef_15 = torch.functional.einsum("ijbs,ibns->bnij", seg_mat_1, ef_14)
        ef_14 = None
        add_53 = ac_7 + x_31
        ac_7 = x_31 = None
        add_54 = add_53 + ef_15
        add_53 = ef_15 = None
        attn_score_14 = add_54 * 0.125
        add_54 = None
        einsum_86 = torch.functional.einsum("ijbn->bnij", non_tgt_mask_1)
        mul_15 = 1e30 * einsum_86
        einsum_86 = None
        attn_score_15 = attn_score_14 - mul_15
        attn_score_14 = mul_15 = None
        attn_prob_14 = torch.nn.functional.softmax(attn_score_15, dim=3)
        attn_score_15 = None
        attn_prob_15 = torch.nn.functional.dropout(attn_prob_14, 0.1, False, False)
        attn_prob_14 = None
        attn_vec_7 = torch.functional.einsum(
            "bnij,jbnd->ibnd", attn_prob_15, v_head_h_7
        )
        attn_prob_15 = v_head_h_7 = None
        attn_out_21 = torch.functional.einsum(
            "ibnd,hnd->ibh",
            attn_vec_7,
            l_self_modules_layer_modules_7_modules_rel_attn_parameters_o_,
        )
        attn_vec_7 = (
            l_self_modules_layer_modules_7_modules_rel_attn_parameters_o_
        ) = None
        attn_out_22 = torch.nn.functional.dropout(attn_out_21, 0.1, False, False)
        attn_out_21 = None
        attn_out_23 = attn_out_22 + output_48
        attn_out_22 = output_48 = None
        output_49 = torch.nn.functional.layer_norm(
            attn_out_23,
            (768,),
            l_self_modules_layer_modules_7_modules_rel_attn_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_7_modules_rel_attn_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        attn_out_23 = l_self_modules_layer_modules_7_modules_rel_attn_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_7_modules_rel_attn_modules_layer_norm_parameters_bias_ = (None)
        output_50 = torch._C._nn.linear(
            output_49,
            l_self_modules_layer_modules_7_modules_ff_modules_layer_1_parameters_weight_,
            l_self_modules_layer_modules_7_modules_ff_modules_layer_1_parameters_bias_,
        )
        l_self_modules_layer_modules_7_modules_ff_modules_layer_1_parameters_weight_ = (
            l_self_modules_layer_modules_7_modules_ff_modules_layer_1_parameters_bias_
        ) = None
        output_51 = torch._C._nn.gelu(output_50)
        output_50 = None
        output_52 = torch.nn.functional.dropout(output_51, 0.1, False, False)
        output_51 = None
        output_53 = torch._C._nn.linear(
            output_52,
            l_self_modules_layer_modules_7_modules_ff_modules_layer_2_parameters_weight_,
            l_self_modules_layer_modules_7_modules_ff_modules_layer_2_parameters_bias_,
        )
        output_52 = (
            l_self_modules_layer_modules_7_modules_ff_modules_layer_2_parameters_weight_
        ) = (
            l_self_modules_layer_modules_7_modules_ff_modules_layer_2_parameters_bias_
        ) = None
        output_54 = torch.nn.functional.dropout(output_53, 0.1, False, False)
        output_53 = None
        add_56 = output_54 + output_49
        output_54 = output_49 = None
        output_55 = torch.nn.functional.layer_norm(
            add_56,
            (768,),
            l_self_modules_layer_modules_7_modules_ff_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_7_modules_ff_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_56 = l_self_modules_layer_modules_7_modules_ff_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_7_modules_ff_modules_layer_norm_parameters_bias_ = (None)
        new_mem_8 = output_55[slice(0, None, None)]
        detach_8 = new_mem_8.detach()
        new_mem_8 = None
        q_head_h_8 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_55,
            l_self_modules_layer_modules_8_modules_rel_attn_parameters_q_,
        )
        l_self_modules_layer_modules_8_modules_rel_attn_parameters_q_ = None
        k_head_h_8 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_55,
            l_self_modules_layer_modules_8_modules_rel_attn_parameters_k_,
        )
        l_self_modules_layer_modules_8_modules_rel_attn_parameters_k_ = None
        v_head_h_8 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_55,
            l_self_modules_layer_modules_8_modules_rel_attn_parameters_v_,
        )
        l_self_modules_layer_modules_8_modules_rel_attn_parameters_v_ = None
        type_9 = pos_emb_4.type(torch.float32)
        k_head_r_8 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            type_9,
            l_self_modules_layer_modules_8_modules_rel_attn_parameters_r_,
        )
        type_9 = l_self_modules_layer_modules_8_modules_rel_attn_parameters_r_ = None
        add_57 = (
            q_head_h_8
            + l_self_modules_layer_modules_8_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_8_modules_rel_attn_parameters_r_w_bias_ = None
        ac_8 = torch.functional.einsum("ibnd,jbnd->bnij", add_57, k_head_h_8)
        add_57 = k_head_h_8 = None
        add_58 = (
            q_head_h_8
            + l_self_modules_layer_modules_8_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_8_modules_rel_attn_parameters_r_r_bias_ = None
        bd_8 = torch.functional.einsum("ibnd,jbnd->bnij", add_58, k_head_r_8)
        add_58 = k_head_r_8 = None
        x_32 = bd_8.reshape(1, 12, 26, 13)
        bd_8 = None
        x_33 = x_32[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        x_32 = None
        x_34 = x_33.reshape(1, 12, 13, 25)
        x_33 = None
        arange_10 = torch.arange(
            13, device=device(type="cuda", index=0), dtype=torch.int64
        )
        x_35 = torch.index_select(x_34, 3, arange_10)
        x_34 = arange_10 = None
        add_59 = (
            q_head_h_8
            + l_self_modules_layer_modules_8_modules_rel_attn_parameters_r_s_bias_
        )
        q_head_h_8 = (
            l_self_modules_layer_modules_8_modules_rel_attn_parameters_r_s_bias_
        ) = None
        ef_16 = torch.functional.einsum(
            "ibnd,snd->ibns",
            add_59,
            l_self_modules_layer_modules_8_modules_rel_attn_parameters_seg_embed_,
        )
        add_59 = (
            l_self_modules_layer_modules_8_modules_rel_attn_parameters_seg_embed_
        ) = None
        ef_17 = torch.functional.einsum("ijbs,ibns->bnij", seg_mat_1, ef_16)
        ef_16 = None
        add_60 = ac_8 + x_35
        ac_8 = x_35 = None
        add_61 = add_60 + ef_17
        add_60 = ef_17 = None
        attn_score_16 = add_61 * 0.125
        add_61 = None
        einsum_97 = torch.functional.einsum("ijbn->bnij", non_tgt_mask_1)
        mul_17 = 1e30 * einsum_97
        einsum_97 = None
        attn_score_17 = attn_score_16 - mul_17
        attn_score_16 = mul_17 = None
        attn_prob_16 = torch.nn.functional.softmax(attn_score_17, dim=3)
        attn_score_17 = None
        attn_prob_17 = torch.nn.functional.dropout(attn_prob_16, 0.1, False, False)
        attn_prob_16 = None
        attn_vec_8 = torch.functional.einsum(
            "bnij,jbnd->ibnd", attn_prob_17, v_head_h_8
        )
        attn_prob_17 = v_head_h_8 = None
        attn_out_24 = torch.functional.einsum(
            "ibnd,hnd->ibh",
            attn_vec_8,
            l_self_modules_layer_modules_8_modules_rel_attn_parameters_o_,
        )
        attn_vec_8 = (
            l_self_modules_layer_modules_8_modules_rel_attn_parameters_o_
        ) = None
        attn_out_25 = torch.nn.functional.dropout(attn_out_24, 0.1, False, False)
        attn_out_24 = None
        attn_out_26 = attn_out_25 + output_55
        attn_out_25 = output_55 = None
        output_56 = torch.nn.functional.layer_norm(
            attn_out_26,
            (768,),
            l_self_modules_layer_modules_8_modules_rel_attn_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_8_modules_rel_attn_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        attn_out_26 = l_self_modules_layer_modules_8_modules_rel_attn_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_8_modules_rel_attn_modules_layer_norm_parameters_bias_ = (None)
        output_57 = torch._C._nn.linear(
            output_56,
            l_self_modules_layer_modules_8_modules_ff_modules_layer_1_parameters_weight_,
            l_self_modules_layer_modules_8_modules_ff_modules_layer_1_parameters_bias_,
        )
        l_self_modules_layer_modules_8_modules_ff_modules_layer_1_parameters_weight_ = (
            l_self_modules_layer_modules_8_modules_ff_modules_layer_1_parameters_bias_
        ) = None
        output_58 = torch._C._nn.gelu(output_57)
        output_57 = None
        output_59 = torch.nn.functional.dropout(output_58, 0.1, False, False)
        output_58 = None
        output_60 = torch._C._nn.linear(
            output_59,
            l_self_modules_layer_modules_8_modules_ff_modules_layer_2_parameters_weight_,
            l_self_modules_layer_modules_8_modules_ff_modules_layer_2_parameters_bias_,
        )
        output_59 = (
            l_self_modules_layer_modules_8_modules_ff_modules_layer_2_parameters_weight_
        ) = (
            l_self_modules_layer_modules_8_modules_ff_modules_layer_2_parameters_bias_
        ) = None
        output_61 = torch.nn.functional.dropout(output_60, 0.1, False, False)
        output_60 = None
        add_63 = output_61 + output_56
        output_61 = output_56 = None
        output_62 = torch.nn.functional.layer_norm(
            add_63,
            (768,),
            l_self_modules_layer_modules_8_modules_ff_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_8_modules_ff_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_63 = l_self_modules_layer_modules_8_modules_ff_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_8_modules_ff_modules_layer_norm_parameters_bias_ = (None)
        new_mem_9 = output_62[slice(0, None, None)]
        detach_9 = new_mem_9.detach()
        new_mem_9 = None
        q_head_h_9 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_62,
            l_self_modules_layer_modules_9_modules_rel_attn_parameters_q_,
        )
        l_self_modules_layer_modules_9_modules_rel_attn_parameters_q_ = None
        k_head_h_9 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_62,
            l_self_modules_layer_modules_9_modules_rel_attn_parameters_k_,
        )
        l_self_modules_layer_modules_9_modules_rel_attn_parameters_k_ = None
        v_head_h_9 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_62,
            l_self_modules_layer_modules_9_modules_rel_attn_parameters_v_,
        )
        l_self_modules_layer_modules_9_modules_rel_attn_parameters_v_ = None
        type_10 = pos_emb_4.type(torch.float32)
        k_head_r_9 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            type_10,
            l_self_modules_layer_modules_9_modules_rel_attn_parameters_r_,
        )
        type_10 = l_self_modules_layer_modules_9_modules_rel_attn_parameters_r_ = None
        add_64 = (
            q_head_h_9
            + l_self_modules_layer_modules_9_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_9_modules_rel_attn_parameters_r_w_bias_ = None
        ac_9 = torch.functional.einsum("ibnd,jbnd->bnij", add_64, k_head_h_9)
        add_64 = k_head_h_9 = None
        add_65 = (
            q_head_h_9
            + l_self_modules_layer_modules_9_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_9_modules_rel_attn_parameters_r_r_bias_ = None
        bd_9 = torch.functional.einsum("ibnd,jbnd->bnij", add_65, k_head_r_9)
        add_65 = k_head_r_9 = None
        x_36 = bd_9.reshape(1, 12, 26, 13)
        bd_9 = None
        x_37 = x_36[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        x_36 = None
        x_38 = x_37.reshape(1, 12, 13, 25)
        x_37 = None
        arange_11 = torch.arange(
            13, device=device(type="cuda", index=0), dtype=torch.int64
        )
        x_39 = torch.index_select(x_38, 3, arange_11)
        x_38 = arange_11 = None
        add_66 = (
            q_head_h_9
            + l_self_modules_layer_modules_9_modules_rel_attn_parameters_r_s_bias_
        )
        q_head_h_9 = (
            l_self_modules_layer_modules_9_modules_rel_attn_parameters_r_s_bias_
        ) = None
        ef_18 = torch.functional.einsum(
            "ibnd,snd->ibns",
            add_66,
            l_self_modules_layer_modules_9_modules_rel_attn_parameters_seg_embed_,
        )
        add_66 = (
            l_self_modules_layer_modules_9_modules_rel_attn_parameters_seg_embed_
        ) = None
        ef_19 = torch.functional.einsum("ijbs,ibns->bnij", seg_mat_1, ef_18)
        ef_18 = None
        add_67 = ac_9 + x_39
        ac_9 = x_39 = None
        add_68 = add_67 + ef_19
        add_67 = ef_19 = None
        attn_score_18 = add_68 * 0.125
        add_68 = None
        einsum_108 = torch.functional.einsum("ijbn->bnij", non_tgt_mask_1)
        mul_19 = 1e30 * einsum_108
        einsum_108 = None
        attn_score_19 = attn_score_18 - mul_19
        attn_score_18 = mul_19 = None
        attn_prob_18 = torch.nn.functional.softmax(attn_score_19, dim=3)
        attn_score_19 = None
        attn_prob_19 = torch.nn.functional.dropout(attn_prob_18, 0.1, False, False)
        attn_prob_18 = None
        attn_vec_9 = torch.functional.einsum(
            "bnij,jbnd->ibnd", attn_prob_19, v_head_h_9
        )
        attn_prob_19 = v_head_h_9 = None
        attn_out_27 = torch.functional.einsum(
            "ibnd,hnd->ibh",
            attn_vec_9,
            l_self_modules_layer_modules_9_modules_rel_attn_parameters_o_,
        )
        attn_vec_9 = (
            l_self_modules_layer_modules_9_modules_rel_attn_parameters_o_
        ) = None
        attn_out_28 = torch.nn.functional.dropout(attn_out_27, 0.1, False, False)
        attn_out_27 = None
        attn_out_29 = attn_out_28 + output_62
        attn_out_28 = output_62 = None
        output_63 = torch.nn.functional.layer_norm(
            attn_out_29,
            (768,),
            l_self_modules_layer_modules_9_modules_rel_attn_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_9_modules_rel_attn_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        attn_out_29 = l_self_modules_layer_modules_9_modules_rel_attn_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_9_modules_rel_attn_modules_layer_norm_parameters_bias_ = (None)
        output_64 = torch._C._nn.linear(
            output_63,
            l_self_modules_layer_modules_9_modules_ff_modules_layer_1_parameters_weight_,
            l_self_modules_layer_modules_9_modules_ff_modules_layer_1_parameters_bias_,
        )
        l_self_modules_layer_modules_9_modules_ff_modules_layer_1_parameters_weight_ = (
            l_self_modules_layer_modules_9_modules_ff_modules_layer_1_parameters_bias_
        ) = None
        output_65 = torch._C._nn.gelu(output_64)
        output_64 = None
        output_66 = torch.nn.functional.dropout(output_65, 0.1, False, False)
        output_65 = None
        output_67 = torch._C._nn.linear(
            output_66,
            l_self_modules_layer_modules_9_modules_ff_modules_layer_2_parameters_weight_,
            l_self_modules_layer_modules_9_modules_ff_modules_layer_2_parameters_bias_,
        )
        output_66 = (
            l_self_modules_layer_modules_9_modules_ff_modules_layer_2_parameters_weight_
        ) = (
            l_self_modules_layer_modules_9_modules_ff_modules_layer_2_parameters_bias_
        ) = None
        output_68 = torch.nn.functional.dropout(output_67, 0.1, False, False)
        output_67 = None
        add_70 = output_68 + output_63
        output_68 = output_63 = None
        output_69 = torch.nn.functional.layer_norm(
            add_70,
            (768,),
            l_self_modules_layer_modules_9_modules_ff_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_9_modules_ff_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_70 = l_self_modules_layer_modules_9_modules_ff_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_9_modules_ff_modules_layer_norm_parameters_bias_ = (None)
        new_mem_10 = output_69[slice(0, None, None)]
        detach_10 = new_mem_10.detach()
        new_mem_10 = None
        q_head_h_10 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_69,
            l_self_modules_layer_modules_10_modules_rel_attn_parameters_q_,
        )
        l_self_modules_layer_modules_10_modules_rel_attn_parameters_q_ = None
        k_head_h_10 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_69,
            l_self_modules_layer_modules_10_modules_rel_attn_parameters_k_,
        )
        l_self_modules_layer_modules_10_modules_rel_attn_parameters_k_ = None
        v_head_h_10 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_69,
            l_self_modules_layer_modules_10_modules_rel_attn_parameters_v_,
        )
        l_self_modules_layer_modules_10_modules_rel_attn_parameters_v_ = None
        type_11 = pos_emb_4.type(torch.float32)
        k_head_r_10 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            type_11,
            l_self_modules_layer_modules_10_modules_rel_attn_parameters_r_,
        )
        type_11 = l_self_modules_layer_modules_10_modules_rel_attn_parameters_r_ = None
        add_71 = (
            q_head_h_10
            + l_self_modules_layer_modules_10_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_10_modules_rel_attn_parameters_r_w_bias_ = None
        ac_10 = torch.functional.einsum("ibnd,jbnd->bnij", add_71, k_head_h_10)
        add_71 = k_head_h_10 = None
        add_72 = (
            q_head_h_10
            + l_self_modules_layer_modules_10_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_10_modules_rel_attn_parameters_r_r_bias_ = None
        bd_10 = torch.functional.einsum("ibnd,jbnd->bnij", add_72, k_head_r_10)
        add_72 = k_head_r_10 = None
        x_40 = bd_10.reshape(1, 12, 26, 13)
        bd_10 = None
        x_41 = x_40[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        x_40 = None
        x_42 = x_41.reshape(1, 12, 13, 25)
        x_41 = None
        arange_12 = torch.arange(
            13, device=device(type="cuda", index=0), dtype=torch.int64
        )
        x_43 = torch.index_select(x_42, 3, arange_12)
        x_42 = arange_12 = None
        add_73 = (
            q_head_h_10
            + l_self_modules_layer_modules_10_modules_rel_attn_parameters_r_s_bias_
        )
        q_head_h_10 = (
            l_self_modules_layer_modules_10_modules_rel_attn_parameters_r_s_bias_
        ) = None
        ef_20 = torch.functional.einsum(
            "ibnd,snd->ibns",
            add_73,
            l_self_modules_layer_modules_10_modules_rel_attn_parameters_seg_embed_,
        )
        add_73 = (
            l_self_modules_layer_modules_10_modules_rel_attn_parameters_seg_embed_
        ) = None
        ef_21 = torch.functional.einsum("ijbs,ibns->bnij", seg_mat_1, ef_20)
        ef_20 = None
        add_74 = ac_10 + x_43
        ac_10 = x_43 = None
        add_75 = add_74 + ef_21
        add_74 = ef_21 = None
        attn_score_20 = add_75 * 0.125
        add_75 = None
        einsum_119 = torch.functional.einsum("ijbn->bnij", non_tgt_mask_1)
        mul_21 = 1e30 * einsum_119
        einsum_119 = None
        attn_score_21 = attn_score_20 - mul_21
        attn_score_20 = mul_21 = None
        attn_prob_20 = torch.nn.functional.softmax(attn_score_21, dim=3)
        attn_score_21 = None
        attn_prob_21 = torch.nn.functional.dropout(attn_prob_20, 0.1, False, False)
        attn_prob_20 = None
        attn_vec_10 = torch.functional.einsum(
            "bnij,jbnd->ibnd", attn_prob_21, v_head_h_10
        )
        attn_prob_21 = v_head_h_10 = None
        attn_out_30 = torch.functional.einsum(
            "ibnd,hnd->ibh",
            attn_vec_10,
            l_self_modules_layer_modules_10_modules_rel_attn_parameters_o_,
        )
        attn_vec_10 = (
            l_self_modules_layer_modules_10_modules_rel_attn_parameters_o_
        ) = None
        attn_out_31 = torch.nn.functional.dropout(attn_out_30, 0.1, False, False)
        attn_out_30 = None
        attn_out_32 = attn_out_31 + output_69
        attn_out_31 = output_69 = None
        output_70 = torch.nn.functional.layer_norm(
            attn_out_32,
            (768,),
            l_self_modules_layer_modules_10_modules_rel_attn_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_10_modules_rel_attn_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        attn_out_32 = l_self_modules_layer_modules_10_modules_rel_attn_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_10_modules_rel_attn_modules_layer_norm_parameters_bias_ = (None)
        output_71 = torch._C._nn.linear(
            output_70,
            l_self_modules_layer_modules_10_modules_ff_modules_layer_1_parameters_weight_,
            l_self_modules_layer_modules_10_modules_ff_modules_layer_1_parameters_bias_,
        )
        l_self_modules_layer_modules_10_modules_ff_modules_layer_1_parameters_weight_ = (
            l_self_modules_layer_modules_10_modules_ff_modules_layer_1_parameters_bias_
        ) = None
        output_72 = torch._C._nn.gelu(output_71)
        output_71 = None
        output_73 = torch.nn.functional.dropout(output_72, 0.1, False, False)
        output_72 = None
        output_74 = torch._C._nn.linear(
            output_73,
            l_self_modules_layer_modules_10_modules_ff_modules_layer_2_parameters_weight_,
            l_self_modules_layer_modules_10_modules_ff_modules_layer_2_parameters_bias_,
        )
        output_73 = l_self_modules_layer_modules_10_modules_ff_modules_layer_2_parameters_weight_ = (
            l_self_modules_layer_modules_10_modules_ff_modules_layer_2_parameters_bias_
        ) = None
        output_75 = torch.nn.functional.dropout(output_74, 0.1, False, False)
        output_74 = None
        add_77 = output_75 + output_70
        output_75 = output_70 = None
        output_76 = torch.nn.functional.layer_norm(
            add_77,
            (768,),
            l_self_modules_layer_modules_10_modules_ff_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_10_modules_ff_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_77 = l_self_modules_layer_modules_10_modules_ff_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_10_modules_ff_modules_layer_norm_parameters_bias_ = (None)
        new_mem_11 = output_76[slice(0, None, None)]
        detach_11 = new_mem_11.detach()
        new_mem_11 = None
        q_head_h_11 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_76,
            l_self_modules_layer_modules_11_modules_rel_attn_parameters_q_,
        )
        l_self_modules_layer_modules_11_modules_rel_attn_parameters_q_ = None
        k_head_h_11 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_76,
            l_self_modules_layer_modules_11_modules_rel_attn_parameters_k_,
        )
        l_self_modules_layer_modules_11_modules_rel_attn_parameters_k_ = None
        v_head_h_11 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_76,
            l_self_modules_layer_modules_11_modules_rel_attn_parameters_v_,
        )
        l_self_modules_layer_modules_11_modules_rel_attn_parameters_v_ = None
        type_12 = pos_emb_4.type(torch.float32)
        pos_emb_4 = None
        k_head_r_11 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            type_12,
            l_self_modules_layer_modules_11_modules_rel_attn_parameters_r_,
        )
        type_12 = l_self_modules_layer_modules_11_modules_rel_attn_parameters_r_ = None
        add_78 = (
            q_head_h_11
            + l_self_modules_layer_modules_11_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_11_modules_rel_attn_parameters_r_w_bias_ = None
        ac_11 = torch.functional.einsum("ibnd,jbnd->bnij", add_78, k_head_h_11)
        add_78 = k_head_h_11 = None
        add_79 = (
            q_head_h_11
            + l_self_modules_layer_modules_11_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_11_modules_rel_attn_parameters_r_r_bias_ = None
        bd_11 = torch.functional.einsum("ibnd,jbnd->bnij", add_79, k_head_r_11)
        add_79 = k_head_r_11 = None
        x_44 = bd_11.reshape(1, 12, 26, 13)
        bd_11 = None
        x_45 = x_44[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        x_44 = None
        x_46 = x_45.reshape(1, 12, 13, 25)
        x_45 = None
        arange_13 = torch.arange(
            13, device=device(type="cuda", index=0), dtype=torch.int64
        )
        x_47 = torch.index_select(x_46, 3, arange_13)
        x_46 = arange_13 = None
        add_80 = (
            q_head_h_11
            + l_self_modules_layer_modules_11_modules_rel_attn_parameters_r_s_bias_
        )
        q_head_h_11 = (
            l_self_modules_layer_modules_11_modules_rel_attn_parameters_r_s_bias_
        ) = None
        ef_22 = torch.functional.einsum(
            "ibnd,snd->ibns",
            add_80,
            l_self_modules_layer_modules_11_modules_rel_attn_parameters_seg_embed_,
        )
        add_80 = (
            l_self_modules_layer_modules_11_modules_rel_attn_parameters_seg_embed_
        ) = None
        ef_23 = torch.functional.einsum("ijbs,ibns->bnij", seg_mat_1, ef_22)
        seg_mat_1 = ef_22 = None
        add_81 = ac_11 + x_47
        ac_11 = x_47 = None
        add_82 = add_81 + ef_23
        add_81 = ef_23 = None
        attn_score_22 = add_82 * 0.125
        add_82 = None
        einsum_130 = torch.functional.einsum("ijbn->bnij", non_tgt_mask_1)
        non_tgt_mask_1 = None
        mul_23 = 1e30 * einsum_130
        einsum_130 = None
        attn_score_23 = attn_score_22 - mul_23
        attn_score_22 = mul_23 = None
        attn_prob_22 = torch.nn.functional.softmax(attn_score_23, dim=3)
        attn_score_23 = None
        attn_prob_23 = torch.nn.functional.dropout(attn_prob_22, 0.1, False, False)
        attn_prob_22 = None
        attn_vec_11 = torch.functional.einsum(
            "bnij,jbnd->ibnd", attn_prob_23, v_head_h_11
        )
        attn_prob_23 = v_head_h_11 = None
        attn_out_33 = torch.functional.einsum(
            "ibnd,hnd->ibh",
            attn_vec_11,
            l_self_modules_layer_modules_11_modules_rel_attn_parameters_o_,
        )
        attn_vec_11 = (
            l_self_modules_layer_modules_11_modules_rel_attn_parameters_o_
        ) = None
        attn_out_34 = torch.nn.functional.dropout(attn_out_33, 0.1, False, False)
        attn_out_33 = None
        attn_out_35 = attn_out_34 + output_76
        attn_out_34 = output_76 = None
        output_77 = torch.nn.functional.layer_norm(
            attn_out_35,
            (768,),
            l_self_modules_layer_modules_11_modules_rel_attn_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_11_modules_rel_attn_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        attn_out_35 = l_self_modules_layer_modules_11_modules_rel_attn_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_11_modules_rel_attn_modules_layer_norm_parameters_bias_ = (None)
        output_78 = torch._C._nn.linear(
            output_77,
            l_self_modules_layer_modules_11_modules_ff_modules_layer_1_parameters_weight_,
            l_self_modules_layer_modules_11_modules_ff_modules_layer_1_parameters_bias_,
        )
        l_self_modules_layer_modules_11_modules_ff_modules_layer_1_parameters_weight_ = (
            l_self_modules_layer_modules_11_modules_ff_modules_layer_1_parameters_bias_
        ) = None
        output_79 = torch._C._nn.gelu(output_78)
        output_78 = None
        output_80 = torch.nn.functional.dropout(output_79, 0.1, False, False)
        output_79 = None
        output_81 = torch._C._nn.linear(
            output_80,
            l_self_modules_layer_modules_11_modules_ff_modules_layer_2_parameters_weight_,
            l_self_modules_layer_modules_11_modules_ff_modules_layer_2_parameters_bias_,
        )
        output_80 = l_self_modules_layer_modules_11_modules_ff_modules_layer_2_parameters_weight_ = (
            l_self_modules_layer_modules_11_modules_ff_modules_layer_2_parameters_bias_
        ) = None
        output_82 = torch.nn.functional.dropout(output_81, 0.1, False, False)
        output_81 = None
        add_84 = output_82 + output_77
        output_82 = output_77 = None
        output_83 = torch.nn.functional.layer_norm(
            add_84,
            (768,),
            l_self_modules_layer_modules_11_modules_ff_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_11_modules_ff_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_84 = l_self_modules_layer_modules_11_modules_ff_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_11_modules_ff_modules_layer_norm_parameters_bias_ = (None)
        output_84 = torch.nn.functional.dropout(output_83, 0.1, False, False)
        output_83 = None
        permute = output_84.permute(1, 0, 2)
        output_84 = None
        output_85 = permute.contiguous()
        permute = None
        return (
            output_85,
            detach,
            detach_1,
            detach_2,
            detach_3,
            detach_4,
            detach_5,
            detach_6,
            detach_7,
            detach_8,
            detach_9,
            detach_10,
            detach_11,
        )
