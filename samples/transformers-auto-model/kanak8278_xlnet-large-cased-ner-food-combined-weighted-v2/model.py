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
        L_self_modules_layer_modules_12_modules_rel_attn_parameters_q_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_12_modules_rel_attn_parameters_k_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_12_modules_rel_attn_parameters_v_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_12_modules_rel_attn_parameters_r_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_12_modules_rel_attn_parameters_r_w_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_12_modules_rel_attn_parameters_r_r_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_12_modules_rel_attn_parameters_r_s_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_12_modules_rel_attn_parameters_seg_embed_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_12_modules_rel_attn_parameters_o_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_12_modules_rel_attn_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_12_modules_rel_attn_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_12_modules_ff_modules_layer_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_12_modules_ff_modules_layer_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_12_modules_ff_modules_layer_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_12_modules_ff_modules_layer_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_12_modules_ff_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_12_modules_ff_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_rel_attn_parameters_q_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_rel_attn_parameters_k_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_rel_attn_parameters_v_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_rel_attn_parameters_r_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_rel_attn_parameters_r_w_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_rel_attn_parameters_r_r_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_rel_attn_parameters_r_s_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_rel_attn_parameters_seg_embed_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_rel_attn_parameters_o_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_rel_attn_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_rel_attn_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_ff_modules_layer_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_ff_modules_layer_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_ff_modules_layer_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_ff_modules_layer_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_ff_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_ff_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_rel_attn_parameters_q_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_rel_attn_parameters_k_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_rel_attn_parameters_v_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_rel_attn_parameters_r_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_rel_attn_parameters_r_w_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_rel_attn_parameters_r_r_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_rel_attn_parameters_r_s_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_rel_attn_parameters_seg_embed_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_rel_attn_parameters_o_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_rel_attn_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_rel_attn_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_ff_modules_layer_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_ff_modules_layer_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_ff_modules_layer_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_ff_modules_layer_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_ff_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_ff_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_rel_attn_parameters_q_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_rel_attn_parameters_k_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_rel_attn_parameters_v_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_rel_attn_parameters_r_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_rel_attn_parameters_r_w_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_rel_attn_parameters_r_r_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_rel_attn_parameters_r_s_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_rel_attn_parameters_seg_embed_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_rel_attn_parameters_o_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_rel_attn_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_rel_attn_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_ff_modules_layer_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_ff_modules_layer_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_ff_modules_layer_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_ff_modules_layer_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_ff_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_ff_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_16_modules_rel_attn_parameters_q_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_16_modules_rel_attn_parameters_k_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_16_modules_rel_attn_parameters_v_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_16_modules_rel_attn_parameters_r_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_16_modules_rel_attn_parameters_r_w_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_16_modules_rel_attn_parameters_r_r_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_16_modules_rel_attn_parameters_r_s_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_16_modules_rel_attn_parameters_seg_embed_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_16_modules_rel_attn_parameters_o_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_16_modules_rel_attn_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_16_modules_rel_attn_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_16_modules_ff_modules_layer_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_16_modules_ff_modules_layer_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_16_modules_ff_modules_layer_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_16_modules_ff_modules_layer_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_16_modules_ff_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_16_modules_ff_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_17_modules_rel_attn_parameters_q_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_17_modules_rel_attn_parameters_k_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_17_modules_rel_attn_parameters_v_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_17_modules_rel_attn_parameters_r_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_17_modules_rel_attn_parameters_r_w_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_17_modules_rel_attn_parameters_r_r_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_17_modules_rel_attn_parameters_r_s_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_17_modules_rel_attn_parameters_seg_embed_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_17_modules_rel_attn_parameters_o_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_17_modules_rel_attn_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_17_modules_rel_attn_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_17_modules_ff_modules_layer_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_17_modules_ff_modules_layer_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_17_modules_ff_modules_layer_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_17_modules_ff_modules_layer_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_17_modules_ff_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_17_modules_ff_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_18_modules_rel_attn_parameters_q_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_18_modules_rel_attn_parameters_k_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_18_modules_rel_attn_parameters_v_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_18_modules_rel_attn_parameters_r_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_18_modules_rel_attn_parameters_r_w_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_18_modules_rel_attn_parameters_r_r_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_18_modules_rel_attn_parameters_r_s_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_18_modules_rel_attn_parameters_seg_embed_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_18_modules_rel_attn_parameters_o_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_18_modules_rel_attn_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_18_modules_rel_attn_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_18_modules_ff_modules_layer_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_18_modules_ff_modules_layer_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_18_modules_ff_modules_layer_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_18_modules_ff_modules_layer_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_18_modules_ff_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_18_modules_ff_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_19_modules_rel_attn_parameters_q_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_19_modules_rel_attn_parameters_k_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_19_modules_rel_attn_parameters_v_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_19_modules_rel_attn_parameters_r_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_19_modules_rel_attn_parameters_r_w_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_19_modules_rel_attn_parameters_r_r_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_19_modules_rel_attn_parameters_r_s_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_19_modules_rel_attn_parameters_seg_embed_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_19_modules_rel_attn_parameters_o_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_19_modules_rel_attn_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_19_modules_rel_attn_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_19_modules_ff_modules_layer_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_19_modules_ff_modules_layer_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_19_modules_ff_modules_layer_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_19_modules_ff_modules_layer_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_19_modules_ff_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_19_modules_ff_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_20_modules_rel_attn_parameters_q_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_20_modules_rel_attn_parameters_k_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_20_modules_rel_attn_parameters_v_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_20_modules_rel_attn_parameters_r_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_20_modules_rel_attn_parameters_r_w_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_20_modules_rel_attn_parameters_r_r_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_20_modules_rel_attn_parameters_r_s_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_20_modules_rel_attn_parameters_seg_embed_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_20_modules_rel_attn_parameters_o_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_20_modules_rel_attn_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_20_modules_rel_attn_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_20_modules_ff_modules_layer_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_20_modules_ff_modules_layer_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_20_modules_ff_modules_layer_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_20_modules_ff_modules_layer_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_20_modules_ff_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_20_modules_ff_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_21_modules_rel_attn_parameters_q_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_21_modules_rel_attn_parameters_k_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_21_modules_rel_attn_parameters_v_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_21_modules_rel_attn_parameters_r_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_21_modules_rel_attn_parameters_r_w_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_21_modules_rel_attn_parameters_r_r_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_21_modules_rel_attn_parameters_r_s_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_21_modules_rel_attn_parameters_seg_embed_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_21_modules_rel_attn_parameters_o_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_21_modules_rel_attn_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_21_modules_rel_attn_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_21_modules_ff_modules_layer_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_21_modules_ff_modules_layer_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_21_modules_ff_modules_layer_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_21_modules_ff_modules_layer_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_21_modules_ff_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_21_modules_ff_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_22_modules_rel_attn_parameters_q_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_22_modules_rel_attn_parameters_k_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_22_modules_rel_attn_parameters_v_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_22_modules_rel_attn_parameters_r_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_22_modules_rel_attn_parameters_r_w_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_22_modules_rel_attn_parameters_r_r_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_22_modules_rel_attn_parameters_r_s_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_22_modules_rel_attn_parameters_seg_embed_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_22_modules_rel_attn_parameters_o_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_22_modules_rel_attn_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_22_modules_rel_attn_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_22_modules_ff_modules_layer_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_22_modules_ff_modules_layer_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_22_modules_ff_modules_layer_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_22_modules_ff_modules_layer_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_22_modules_ff_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_22_modules_ff_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_23_modules_rel_attn_parameters_q_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_23_modules_rel_attn_parameters_k_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_23_modules_rel_attn_parameters_v_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_23_modules_rel_attn_parameters_r_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_23_modules_rel_attn_parameters_r_w_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_23_modules_rel_attn_parameters_r_r_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_23_modules_rel_attn_parameters_r_s_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_23_modules_rel_attn_parameters_seg_embed_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_23_modules_rel_attn_parameters_o_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_23_modules_rel_attn_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_23_modules_rel_attn_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_23_modules_ff_modules_layer_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_23_modules_ff_modules_layer_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_23_modules_ff_modules_layer_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_23_modules_ff_modules_layer_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_23_modules_ff_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_23_modules_ff_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_layer_modules_12_modules_rel_attn_parameters_q_ = (
            L_self_modules_layer_modules_12_modules_rel_attn_parameters_q_
        )
        l_self_modules_layer_modules_12_modules_rel_attn_parameters_k_ = (
            L_self_modules_layer_modules_12_modules_rel_attn_parameters_k_
        )
        l_self_modules_layer_modules_12_modules_rel_attn_parameters_v_ = (
            L_self_modules_layer_modules_12_modules_rel_attn_parameters_v_
        )
        l_self_modules_layer_modules_12_modules_rel_attn_parameters_r_ = (
            L_self_modules_layer_modules_12_modules_rel_attn_parameters_r_
        )
        l_self_modules_layer_modules_12_modules_rel_attn_parameters_r_w_bias_ = (
            L_self_modules_layer_modules_12_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_12_modules_rel_attn_parameters_r_r_bias_ = (
            L_self_modules_layer_modules_12_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_12_modules_rel_attn_parameters_r_s_bias_ = (
            L_self_modules_layer_modules_12_modules_rel_attn_parameters_r_s_bias_
        )
        l_self_modules_layer_modules_12_modules_rel_attn_parameters_seg_embed_ = (
            L_self_modules_layer_modules_12_modules_rel_attn_parameters_seg_embed_
        )
        l_self_modules_layer_modules_12_modules_rel_attn_parameters_o_ = (
            L_self_modules_layer_modules_12_modules_rel_attn_parameters_o_
        )
        l_self_modules_layer_modules_12_modules_rel_attn_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_12_modules_rel_attn_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_12_modules_rel_attn_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_12_modules_rel_attn_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_12_modules_ff_modules_layer_1_parameters_weight_ = L_self_modules_layer_modules_12_modules_ff_modules_layer_1_parameters_weight_
        l_self_modules_layer_modules_12_modules_ff_modules_layer_1_parameters_bias_ = (
            L_self_modules_layer_modules_12_modules_ff_modules_layer_1_parameters_bias_
        )
        l_self_modules_layer_modules_12_modules_ff_modules_layer_2_parameters_weight_ = L_self_modules_layer_modules_12_modules_ff_modules_layer_2_parameters_weight_
        l_self_modules_layer_modules_12_modules_ff_modules_layer_2_parameters_bias_ = (
            L_self_modules_layer_modules_12_modules_ff_modules_layer_2_parameters_bias_
        )
        l_self_modules_layer_modules_12_modules_ff_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_12_modules_ff_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_12_modules_ff_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_12_modules_ff_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_13_modules_rel_attn_parameters_q_ = (
            L_self_modules_layer_modules_13_modules_rel_attn_parameters_q_
        )
        l_self_modules_layer_modules_13_modules_rel_attn_parameters_k_ = (
            L_self_modules_layer_modules_13_modules_rel_attn_parameters_k_
        )
        l_self_modules_layer_modules_13_modules_rel_attn_parameters_v_ = (
            L_self_modules_layer_modules_13_modules_rel_attn_parameters_v_
        )
        l_self_modules_layer_modules_13_modules_rel_attn_parameters_r_ = (
            L_self_modules_layer_modules_13_modules_rel_attn_parameters_r_
        )
        l_self_modules_layer_modules_13_modules_rel_attn_parameters_r_w_bias_ = (
            L_self_modules_layer_modules_13_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_13_modules_rel_attn_parameters_r_r_bias_ = (
            L_self_modules_layer_modules_13_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_13_modules_rel_attn_parameters_r_s_bias_ = (
            L_self_modules_layer_modules_13_modules_rel_attn_parameters_r_s_bias_
        )
        l_self_modules_layer_modules_13_modules_rel_attn_parameters_seg_embed_ = (
            L_self_modules_layer_modules_13_modules_rel_attn_parameters_seg_embed_
        )
        l_self_modules_layer_modules_13_modules_rel_attn_parameters_o_ = (
            L_self_modules_layer_modules_13_modules_rel_attn_parameters_o_
        )
        l_self_modules_layer_modules_13_modules_rel_attn_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_13_modules_rel_attn_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_13_modules_rel_attn_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_13_modules_rel_attn_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_13_modules_ff_modules_layer_1_parameters_weight_ = L_self_modules_layer_modules_13_modules_ff_modules_layer_1_parameters_weight_
        l_self_modules_layer_modules_13_modules_ff_modules_layer_1_parameters_bias_ = (
            L_self_modules_layer_modules_13_modules_ff_modules_layer_1_parameters_bias_
        )
        l_self_modules_layer_modules_13_modules_ff_modules_layer_2_parameters_weight_ = L_self_modules_layer_modules_13_modules_ff_modules_layer_2_parameters_weight_
        l_self_modules_layer_modules_13_modules_ff_modules_layer_2_parameters_bias_ = (
            L_self_modules_layer_modules_13_modules_ff_modules_layer_2_parameters_bias_
        )
        l_self_modules_layer_modules_13_modules_ff_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_13_modules_ff_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_13_modules_ff_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_13_modules_ff_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_14_modules_rel_attn_parameters_q_ = (
            L_self_modules_layer_modules_14_modules_rel_attn_parameters_q_
        )
        l_self_modules_layer_modules_14_modules_rel_attn_parameters_k_ = (
            L_self_modules_layer_modules_14_modules_rel_attn_parameters_k_
        )
        l_self_modules_layer_modules_14_modules_rel_attn_parameters_v_ = (
            L_self_modules_layer_modules_14_modules_rel_attn_parameters_v_
        )
        l_self_modules_layer_modules_14_modules_rel_attn_parameters_r_ = (
            L_self_modules_layer_modules_14_modules_rel_attn_parameters_r_
        )
        l_self_modules_layer_modules_14_modules_rel_attn_parameters_r_w_bias_ = (
            L_self_modules_layer_modules_14_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_14_modules_rel_attn_parameters_r_r_bias_ = (
            L_self_modules_layer_modules_14_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_14_modules_rel_attn_parameters_r_s_bias_ = (
            L_self_modules_layer_modules_14_modules_rel_attn_parameters_r_s_bias_
        )
        l_self_modules_layer_modules_14_modules_rel_attn_parameters_seg_embed_ = (
            L_self_modules_layer_modules_14_modules_rel_attn_parameters_seg_embed_
        )
        l_self_modules_layer_modules_14_modules_rel_attn_parameters_o_ = (
            L_self_modules_layer_modules_14_modules_rel_attn_parameters_o_
        )
        l_self_modules_layer_modules_14_modules_rel_attn_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_14_modules_rel_attn_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_14_modules_rel_attn_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_14_modules_rel_attn_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_14_modules_ff_modules_layer_1_parameters_weight_ = L_self_modules_layer_modules_14_modules_ff_modules_layer_1_parameters_weight_
        l_self_modules_layer_modules_14_modules_ff_modules_layer_1_parameters_bias_ = (
            L_self_modules_layer_modules_14_modules_ff_modules_layer_1_parameters_bias_
        )
        l_self_modules_layer_modules_14_modules_ff_modules_layer_2_parameters_weight_ = L_self_modules_layer_modules_14_modules_ff_modules_layer_2_parameters_weight_
        l_self_modules_layer_modules_14_modules_ff_modules_layer_2_parameters_bias_ = (
            L_self_modules_layer_modules_14_modules_ff_modules_layer_2_parameters_bias_
        )
        l_self_modules_layer_modules_14_modules_ff_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_14_modules_ff_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_14_modules_ff_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_14_modules_ff_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_15_modules_rel_attn_parameters_q_ = (
            L_self_modules_layer_modules_15_modules_rel_attn_parameters_q_
        )
        l_self_modules_layer_modules_15_modules_rel_attn_parameters_k_ = (
            L_self_modules_layer_modules_15_modules_rel_attn_parameters_k_
        )
        l_self_modules_layer_modules_15_modules_rel_attn_parameters_v_ = (
            L_self_modules_layer_modules_15_modules_rel_attn_parameters_v_
        )
        l_self_modules_layer_modules_15_modules_rel_attn_parameters_r_ = (
            L_self_modules_layer_modules_15_modules_rel_attn_parameters_r_
        )
        l_self_modules_layer_modules_15_modules_rel_attn_parameters_r_w_bias_ = (
            L_self_modules_layer_modules_15_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_15_modules_rel_attn_parameters_r_r_bias_ = (
            L_self_modules_layer_modules_15_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_15_modules_rel_attn_parameters_r_s_bias_ = (
            L_self_modules_layer_modules_15_modules_rel_attn_parameters_r_s_bias_
        )
        l_self_modules_layer_modules_15_modules_rel_attn_parameters_seg_embed_ = (
            L_self_modules_layer_modules_15_modules_rel_attn_parameters_seg_embed_
        )
        l_self_modules_layer_modules_15_modules_rel_attn_parameters_o_ = (
            L_self_modules_layer_modules_15_modules_rel_attn_parameters_o_
        )
        l_self_modules_layer_modules_15_modules_rel_attn_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_15_modules_rel_attn_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_15_modules_rel_attn_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_15_modules_rel_attn_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_15_modules_ff_modules_layer_1_parameters_weight_ = L_self_modules_layer_modules_15_modules_ff_modules_layer_1_parameters_weight_
        l_self_modules_layer_modules_15_modules_ff_modules_layer_1_parameters_bias_ = (
            L_self_modules_layer_modules_15_modules_ff_modules_layer_1_parameters_bias_
        )
        l_self_modules_layer_modules_15_modules_ff_modules_layer_2_parameters_weight_ = L_self_modules_layer_modules_15_modules_ff_modules_layer_2_parameters_weight_
        l_self_modules_layer_modules_15_modules_ff_modules_layer_2_parameters_bias_ = (
            L_self_modules_layer_modules_15_modules_ff_modules_layer_2_parameters_bias_
        )
        l_self_modules_layer_modules_15_modules_ff_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_15_modules_ff_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_15_modules_ff_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_15_modules_ff_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_16_modules_rel_attn_parameters_q_ = (
            L_self_modules_layer_modules_16_modules_rel_attn_parameters_q_
        )
        l_self_modules_layer_modules_16_modules_rel_attn_parameters_k_ = (
            L_self_modules_layer_modules_16_modules_rel_attn_parameters_k_
        )
        l_self_modules_layer_modules_16_modules_rel_attn_parameters_v_ = (
            L_self_modules_layer_modules_16_modules_rel_attn_parameters_v_
        )
        l_self_modules_layer_modules_16_modules_rel_attn_parameters_r_ = (
            L_self_modules_layer_modules_16_modules_rel_attn_parameters_r_
        )
        l_self_modules_layer_modules_16_modules_rel_attn_parameters_r_w_bias_ = (
            L_self_modules_layer_modules_16_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_16_modules_rel_attn_parameters_r_r_bias_ = (
            L_self_modules_layer_modules_16_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_16_modules_rel_attn_parameters_r_s_bias_ = (
            L_self_modules_layer_modules_16_modules_rel_attn_parameters_r_s_bias_
        )
        l_self_modules_layer_modules_16_modules_rel_attn_parameters_seg_embed_ = (
            L_self_modules_layer_modules_16_modules_rel_attn_parameters_seg_embed_
        )
        l_self_modules_layer_modules_16_modules_rel_attn_parameters_o_ = (
            L_self_modules_layer_modules_16_modules_rel_attn_parameters_o_
        )
        l_self_modules_layer_modules_16_modules_rel_attn_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_16_modules_rel_attn_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_16_modules_rel_attn_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_16_modules_rel_attn_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_16_modules_ff_modules_layer_1_parameters_weight_ = L_self_modules_layer_modules_16_modules_ff_modules_layer_1_parameters_weight_
        l_self_modules_layer_modules_16_modules_ff_modules_layer_1_parameters_bias_ = (
            L_self_modules_layer_modules_16_modules_ff_modules_layer_1_parameters_bias_
        )
        l_self_modules_layer_modules_16_modules_ff_modules_layer_2_parameters_weight_ = L_self_modules_layer_modules_16_modules_ff_modules_layer_2_parameters_weight_
        l_self_modules_layer_modules_16_modules_ff_modules_layer_2_parameters_bias_ = (
            L_self_modules_layer_modules_16_modules_ff_modules_layer_2_parameters_bias_
        )
        l_self_modules_layer_modules_16_modules_ff_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_16_modules_ff_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_16_modules_ff_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_16_modules_ff_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_17_modules_rel_attn_parameters_q_ = (
            L_self_modules_layer_modules_17_modules_rel_attn_parameters_q_
        )
        l_self_modules_layer_modules_17_modules_rel_attn_parameters_k_ = (
            L_self_modules_layer_modules_17_modules_rel_attn_parameters_k_
        )
        l_self_modules_layer_modules_17_modules_rel_attn_parameters_v_ = (
            L_self_modules_layer_modules_17_modules_rel_attn_parameters_v_
        )
        l_self_modules_layer_modules_17_modules_rel_attn_parameters_r_ = (
            L_self_modules_layer_modules_17_modules_rel_attn_parameters_r_
        )
        l_self_modules_layer_modules_17_modules_rel_attn_parameters_r_w_bias_ = (
            L_self_modules_layer_modules_17_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_17_modules_rel_attn_parameters_r_r_bias_ = (
            L_self_modules_layer_modules_17_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_17_modules_rel_attn_parameters_r_s_bias_ = (
            L_self_modules_layer_modules_17_modules_rel_attn_parameters_r_s_bias_
        )
        l_self_modules_layer_modules_17_modules_rel_attn_parameters_seg_embed_ = (
            L_self_modules_layer_modules_17_modules_rel_attn_parameters_seg_embed_
        )
        l_self_modules_layer_modules_17_modules_rel_attn_parameters_o_ = (
            L_self_modules_layer_modules_17_modules_rel_attn_parameters_o_
        )
        l_self_modules_layer_modules_17_modules_rel_attn_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_17_modules_rel_attn_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_17_modules_rel_attn_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_17_modules_rel_attn_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_17_modules_ff_modules_layer_1_parameters_weight_ = L_self_modules_layer_modules_17_modules_ff_modules_layer_1_parameters_weight_
        l_self_modules_layer_modules_17_modules_ff_modules_layer_1_parameters_bias_ = (
            L_self_modules_layer_modules_17_modules_ff_modules_layer_1_parameters_bias_
        )
        l_self_modules_layer_modules_17_modules_ff_modules_layer_2_parameters_weight_ = L_self_modules_layer_modules_17_modules_ff_modules_layer_2_parameters_weight_
        l_self_modules_layer_modules_17_modules_ff_modules_layer_2_parameters_bias_ = (
            L_self_modules_layer_modules_17_modules_ff_modules_layer_2_parameters_bias_
        )
        l_self_modules_layer_modules_17_modules_ff_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_17_modules_ff_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_17_modules_ff_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_17_modules_ff_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_18_modules_rel_attn_parameters_q_ = (
            L_self_modules_layer_modules_18_modules_rel_attn_parameters_q_
        )
        l_self_modules_layer_modules_18_modules_rel_attn_parameters_k_ = (
            L_self_modules_layer_modules_18_modules_rel_attn_parameters_k_
        )
        l_self_modules_layer_modules_18_modules_rel_attn_parameters_v_ = (
            L_self_modules_layer_modules_18_modules_rel_attn_parameters_v_
        )
        l_self_modules_layer_modules_18_modules_rel_attn_parameters_r_ = (
            L_self_modules_layer_modules_18_modules_rel_attn_parameters_r_
        )
        l_self_modules_layer_modules_18_modules_rel_attn_parameters_r_w_bias_ = (
            L_self_modules_layer_modules_18_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_18_modules_rel_attn_parameters_r_r_bias_ = (
            L_self_modules_layer_modules_18_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_18_modules_rel_attn_parameters_r_s_bias_ = (
            L_self_modules_layer_modules_18_modules_rel_attn_parameters_r_s_bias_
        )
        l_self_modules_layer_modules_18_modules_rel_attn_parameters_seg_embed_ = (
            L_self_modules_layer_modules_18_modules_rel_attn_parameters_seg_embed_
        )
        l_self_modules_layer_modules_18_modules_rel_attn_parameters_o_ = (
            L_self_modules_layer_modules_18_modules_rel_attn_parameters_o_
        )
        l_self_modules_layer_modules_18_modules_rel_attn_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_18_modules_rel_attn_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_18_modules_rel_attn_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_18_modules_rel_attn_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_18_modules_ff_modules_layer_1_parameters_weight_ = L_self_modules_layer_modules_18_modules_ff_modules_layer_1_parameters_weight_
        l_self_modules_layer_modules_18_modules_ff_modules_layer_1_parameters_bias_ = (
            L_self_modules_layer_modules_18_modules_ff_modules_layer_1_parameters_bias_
        )
        l_self_modules_layer_modules_18_modules_ff_modules_layer_2_parameters_weight_ = L_self_modules_layer_modules_18_modules_ff_modules_layer_2_parameters_weight_
        l_self_modules_layer_modules_18_modules_ff_modules_layer_2_parameters_bias_ = (
            L_self_modules_layer_modules_18_modules_ff_modules_layer_2_parameters_bias_
        )
        l_self_modules_layer_modules_18_modules_ff_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_18_modules_ff_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_18_modules_ff_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_18_modules_ff_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_19_modules_rel_attn_parameters_q_ = (
            L_self_modules_layer_modules_19_modules_rel_attn_parameters_q_
        )
        l_self_modules_layer_modules_19_modules_rel_attn_parameters_k_ = (
            L_self_modules_layer_modules_19_modules_rel_attn_parameters_k_
        )
        l_self_modules_layer_modules_19_modules_rel_attn_parameters_v_ = (
            L_self_modules_layer_modules_19_modules_rel_attn_parameters_v_
        )
        l_self_modules_layer_modules_19_modules_rel_attn_parameters_r_ = (
            L_self_modules_layer_modules_19_modules_rel_attn_parameters_r_
        )
        l_self_modules_layer_modules_19_modules_rel_attn_parameters_r_w_bias_ = (
            L_self_modules_layer_modules_19_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_19_modules_rel_attn_parameters_r_r_bias_ = (
            L_self_modules_layer_modules_19_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_19_modules_rel_attn_parameters_r_s_bias_ = (
            L_self_modules_layer_modules_19_modules_rel_attn_parameters_r_s_bias_
        )
        l_self_modules_layer_modules_19_modules_rel_attn_parameters_seg_embed_ = (
            L_self_modules_layer_modules_19_modules_rel_attn_parameters_seg_embed_
        )
        l_self_modules_layer_modules_19_modules_rel_attn_parameters_o_ = (
            L_self_modules_layer_modules_19_modules_rel_attn_parameters_o_
        )
        l_self_modules_layer_modules_19_modules_rel_attn_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_19_modules_rel_attn_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_19_modules_rel_attn_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_19_modules_rel_attn_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_19_modules_ff_modules_layer_1_parameters_weight_ = L_self_modules_layer_modules_19_modules_ff_modules_layer_1_parameters_weight_
        l_self_modules_layer_modules_19_modules_ff_modules_layer_1_parameters_bias_ = (
            L_self_modules_layer_modules_19_modules_ff_modules_layer_1_parameters_bias_
        )
        l_self_modules_layer_modules_19_modules_ff_modules_layer_2_parameters_weight_ = L_self_modules_layer_modules_19_modules_ff_modules_layer_2_parameters_weight_
        l_self_modules_layer_modules_19_modules_ff_modules_layer_2_parameters_bias_ = (
            L_self_modules_layer_modules_19_modules_ff_modules_layer_2_parameters_bias_
        )
        l_self_modules_layer_modules_19_modules_ff_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_19_modules_ff_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_19_modules_ff_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_19_modules_ff_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_20_modules_rel_attn_parameters_q_ = (
            L_self_modules_layer_modules_20_modules_rel_attn_parameters_q_
        )
        l_self_modules_layer_modules_20_modules_rel_attn_parameters_k_ = (
            L_self_modules_layer_modules_20_modules_rel_attn_parameters_k_
        )
        l_self_modules_layer_modules_20_modules_rel_attn_parameters_v_ = (
            L_self_modules_layer_modules_20_modules_rel_attn_parameters_v_
        )
        l_self_modules_layer_modules_20_modules_rel_attn_parameters_r_ = (
            L_self_modules_layer_modules_20_modules_rel_attn_parameters_r_
        )
        l_self_modules_layer_modules_20_modules_rel_attn_parameters_r_w_bias_ = (
            L_self_modules_layer_modules_20_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_20_modules_rel_attn_parameters_r_r_bias_ = (
            L_self_modules_layer_modules_20_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_20_modules_rel_attn_parameters_r_s_bias_ = (
            L_self_modules_layer_modules_20_modules_rel_attn_parameters_r_s_bias_
        )
        l_self_modules_layer_modules_20_modules_rel_attn_parameters_seg_embed_ = (
            L_self_modules_layer_modules_20_modules_rel_attn_parameters_seg_embed_
        )
        l_self_modules_layer_modules_20_modules_rel_attn_parameters_o_ = (
            L_self_modules_layer_modules_20_modules_rel_attn_parameters_o_
        )
        l_self_modules_layer_modules_20_modules_rel_attn_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_20_modules_rel_attn_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_20_modules_rel_attn_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_20_modules_rel_attn_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_20_modules_ff_modules_layer_1_parameters_weight_ = L_self_modules_layer_modules_20_modules_ff_modules_layer_1_parameters_weight_
        l_self_modules_layer_modules_20_modules_ff_modules_layer_1_parameters_bias_ = (
            L_self_modules_layer_modules_20_modules_ff_modules_layer_1_parameters_bias_
        )
        l_self_modules_layer_modules_20_modules_ff_modules_layer_2_parameters_weight_ = L_self_modules_layer_modules_20_modules_ff_modules_layer_2_parameters_weight_
        l_self_modules_layer_modules_20_modules_ff_modules_layer_2_parameters_bias_ = (
            L_self_modules_layer_modules_20_modules_ff_modules_layer_2_parameters_bias_
        )
        l_self_modules_layer_modules_20_modules_ff_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_20_modules_ff_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_20_modules_ff_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_20_modules_ff_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_21_modules_rel_attn_parameters_q_ = (
            L_self_modules_layer_modules_21_modules_rel_attn_parameters_q_
        )
        l_self_modules_layer_modules_21_modules_rel_attn_parameters_k_ = (
            L_self_modules_layer_modules_21_modules_rel_attn_parameters_k_
        )
        l_self_modules_layer_modules_21_modules_rel_attn_parameters_v_ = (
            L_self_modules_layer_modules_21_modules_rel_attn_parameters_v_
        )
        l_self_modules_layer_modules_21_modules_rel_attn_parameters_r_ = (
            L_self_modules_layer_modules_21_modules_rel_attn_parameters_r_
        )
        l_self_modules_layer_modules_21_modules_rel_attn_parameters_r_w_bias_ = (
            L_self_modules_layer_modules_21_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_21_modules_rel_attn_parameters_r_r_bias_ = (
            L_self_modules_layer_modules_21_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_21_modules_rel_attn_parameters_r_s_bias_ = (
            L_self_modules_layer_modules_21_modules_rel_attn_parameters_r_s_bias_
        )
        l_self_modules_layer_modules_21_modules_rel_attn_parameters_seg_embed_ = (
            L_self_modules_layer_modules_21_modules_rel_attn_parameters_seg_embed_
        )
        l_self_modules_layer_modules_21_modules_rel_attn_parameters_o_ = (
            L_self_modules_layer_modules_21_modules_rel_attn_parameters_o_
        )
        l_self_modules_layer_modules_21_modules_rel_attn_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_21_modules_rel_attn_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_21_modules_rel_attn_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_21_modules_rel_attn_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_21_modules_ff_modules_layer_1_parameters_weight_ = L_self_modules_layer_modules_21_modules_ff_modules_layer_1_parameters_weight_
        l_self_modules_layer_modules_21_modules_ff_modules_layer_1_parameters_bias_ = (
            L_self_modules_layer_modules_21_modules_ff_modules_layer_1_parameters_bias_
        )
        l_self_modules_layer_modules_21_modules_ff_modules_layer_2_parameters_weight_ = L_self_modules_layer_modules_21_modules_ff_modules_layer_2_parameters_weight_
        l_self_modules_layer_modules_21_modules_ff_modules_layer_2_parameters_bias_ = (
            L_self_modules_layer_modules_21_modules_ff_modules_layer_2_parameters_bias_
        )
        l_self_modules_layer_modules_21_modules_ff_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_21_modules_ff_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_21_modules_ff_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_21_modules_ff_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_22_modules_rel_attn_parameters_q_ = (
            L_self_modules_layer_modules_22_modules_rel_attn_parameters_q_
        )
        l_self_modules_layer_modules_22_modules_rel_attn_parameters_k_ = (
            L_self_modules_layer_modules_22_modules_rel_attn_parameters_k_
        )
        l_self_modules_layer_modules_22_modules_rel_attn_parameters_v_ = (
            L_self_modules_layer_modules_22_modules_rel_attn_parameters_v_
        )
        l_self_modules_layer_modules_22_modules_rel_attn_parameters_r_ = (
            L_self_modules_layer_modules_22_modules_rel_attn_parameters_r_
        )
        l_self_modules_layer_modules_22_modules_rel_attn_parameters_r_w_bias_ = (
            L_self_modules_layer_modules_22_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_22_modules_rel_attn_parameters_r_r_bias_ = (
            L_self_modules_layer_modules_22_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_22_modules_rel_attn_parameters_r_s_bias_ = (
            L_self_modules_layer_modules_22_modules_rel_attn_parameters_r_s_bias_
        )
        l_self_modules_layer_modules_22_modules_rel_attn_parameters_seg_embed_ = (
            L_self_modules_layer_modules_22_modules_rel_attn_parameters_seg_embed_
        )
        l_self_modules_layer_modules_22_modules_rel_attn_parameters_o_ = (
            L_self_modules_layer_modules_22_modules_rel_attn_parameters_o_
        )
        l_self_modules_layer_modules_22_modules_rel_attn_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_22_modules_rel_attn_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_22_modules_rel_attn_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_22_modules_rel_attn_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_22_modules_ff_modules_layer_1_parameters_weight_ = L_self_modules_layer_modules_22_modules_ff_modules_layer_1_parameters_weight_
        l_self_modules_layer_modules_22_modules_ff_modules_layer_1_parameters_bias_ = (
            L_self_modules_layer_modules_22_modules_ff_modules_layer_1_parameters_bias_
        )
        l_self_modules_layer_modules_22_modules_ff_modules_layer_2_parameters_weight_ = L_self_modules_layer_modules_22_modules_ff_modules_layer_2_parameters_weight_
        l_self_modules_layer_modules_22_modules_ff_modules_layer_2_parameters_bias_ = (
            L_self_modules_layer_modules_22_modules_ff_modules_layer_2_parameters_bias_
        )
        l_self_modules_layer_modules_22_modules_ff_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_22_modules_ff_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_22_modules_ff_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_22_modules_ff_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_23_modules_rel_attn_parameters_q_ = (
            L_self_modules_layer_modules_23_modules_rel_attn_parameters_q_
        )
        l_self_modules_layer_modules_23_modules_rel_attn_parameters_k_ = (
            L_self_modules_layer_modules_23_modules_rel_attn_parameters_k_
        )
        l_self_modules_layer_modules_23_modules_rel_attn_parameters_v_ = (
            L_self_modules_layer_modules_23_modules_rel_attn_parameters_v_
        )
        l_self_modules_layer_modules_23_modules_rel_attn_parameters_r_ = (
            L_self_modules_layer_modules_23_modules_rel_attn_parameters_r_
        )
        l_self_modules_layer_modules_23_modules_rel_attn_parameters_r_w_bias_ = (
            L_self_modules_layer_modules_23_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_23_modules_rel_attn_parameters_r_r_bias_ = (
            L_self_modules_layer_modules_23_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_23_modules_rel_attn_parameters_r_s_bias_ = (
            L_self_modules_layer_modules_23_modules_rel_attn_parameters_r_s_bias_
        )
        l_self_modules_layer_modules_23_modules_rel_attn_parameters_seg_embed_ = (
            L_self_modules_layer_modules_23_modules_rel_attn_parameters_seg_embed_
        )
        l_self_modules_layer_modules_23_modules_rel_attn_parameters_o_ = (
            L_self_modules_layer_modules_23_modules_rel_attn_parameters_o_
        )
        l_self_modules_layer_modules_23_modules_rel_attn_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_23_modules_rel_attn_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_23_modules_rel_attn_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_23_modules_rel_attn_modules_layer_norm_parameters_bias_
        l_self_modules_layer_modules_23_modules_ff_modules_layer_1_parameters_weight_ = L_self_modules_layer_modules_23_modules_ff_modules_layer_1_parameters_weight_
        l_self_modules_layer_modules_23_modules_ff_modules_layer_1_parameters_bias_ = (
            L_self_modules_layer_modules_23_modules_ff_modules_layer_1_parameters_bias_
        )
        l_self_modules_layer_modules_23_modules_ff_modules_layer_2_parameters_weight_ = L_self_modules_layer_modules_23_modules_ff_modules_layer_2_parameters_weight_
        l_self_modules_layer_modules_23_modules_ff_modules_layer_2_parameters_bias_ = (
            L_self_modules_layer_modules_23_modules_ff_modules_layer_2_parameters_bias_
        )
        l_self_modules_layer_modules_23_modules_ff_modules_layer_norm_parameters_weight_ = L_self_modules_layer_modules_23_modules_ff_modules_layer_norm_parameters_weight_
        l_self_modules_layer_modules_23_modules_ff_modules_layer_norm_parameters_bias_ = L_self_modules_layer_modules_23_modules_ff_modules_layer_norm_parameters_bias_
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
        arange = torch.arange(0, 1024, 2.0, dtype=torch.int64)
        freq_seq = arange.float()
        arange = None
        truediv = freq_seq / 1024
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
        x = bd.reshape(1, 16, 26, 13)
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
        x_2 = x_1.reshape(1, 16, 13, 25)
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
            (1024,),
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
            (1024,),
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
        x_4 = bd_1.reshape(1, 16, 26, 13)
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
        x_6 = x_5.reshape(1, 16, 13, 25)
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
            (1024,),
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
            (1024,),
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
        x_8 = bd_2.reshape(1, 16, 26, 13)
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
        x_10 = x_9.reshape(1, 16, 13, 25)
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
            (1024,),
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
            (1024,),
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
        x_12 = bd_3.reshape(1, 16, 26, 13)
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
        x_14 = x_13.reshape(1, 16, 13, 25)
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
            (1024,),
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
            (1024,),
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
        x_16 = bd_4.reshape(1, 16, 26, 13)
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
        x_18 = x_17.reshape(1, 16, 13, 25)
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
            (1024,),
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
            (1024,),
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
        x_20 = bd_5.reshape(1, 16, 26, 13)
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
        x_22 = x_21.reshape(1, 16, 13, 25)
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
            (1024,),
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
            (1024,),
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
        x_24 = bd_6.reshape(1, 16, 26, 13)
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
        x_26 = x_25.reshape(1, 16, 13, 25)
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
            (1024,),
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
            (1024,),
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
        x_28 = bd_7.reshape(1, 16, 26, 13)
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
        x_30 = x_29.reshape(1, 16, 13, 25)
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
            (1024,),
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
            (1024,),
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
        x_32 = bd_8.reshape(1, 16, 26, 13)
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
        x_34 = x_33.reshape(1, 16, 13, 25)
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
            (1024,),
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
            (1024,),
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
        x_36 = bd_9.reshape(1, 16, 26, 13)
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
        x_38 = x_37.reshape(1, 16, 13, 25)
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
            (1024,),
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
            (1024,),
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
        x_40 = bd_10.reshape(1, 16, 26, 13)
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
        x_42 = x_41.reshape(1, 16, 13, 25)
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
            (1024,),
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
            (1024,),
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
        x_44 = bd_11.reshape(1, 16, 26, 13)
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
        x_46 = x_45.reshape(1, 16, 13, 25)
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
        ef_22 = None
        add_81 = ac_11 + x_47
        ac_11 = x_47 = None
        add_82 = add_81 + ef_23
        add_81 = ef_23 = None
        attn_score_22 = add_82 * 0.125
        add_82 = None
        einsum_130 = torch.functional.einsum("ijbn->bnij", non_tgt_mask_1)
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
            (1024,),
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
            (1024,),
            l_self_modules_layer_modules_11_modules_ff_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_11_modules_ff_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_84 = l_self_modules_layer_modules_11_modules_ff_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_11_modules_ff_modules_layer_norm_parameters_bias_ = (None)
        new_mem_12 = output_83[slice(0, None, None)]
        detach_12 = new_mem_12.detach()
        new_mem_12 = None
        q_head_h_12 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_83,
            l_self_modules_layer_modules_12_modules_rel_attn_parameters_q_,
        )
        l_self_modules_layer_modules_12_modules_rel_attn_parameters_q_ = None
        k_head_h_12 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_83,
            l_self_modules_layer_modules_12_modules_rel_attn_parameters_k_,
        )
        l_self_modules_layer_modules_12_modules_rel_attn_parameters_k_ = None
        v_head_h_12 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_83,
            l_self_modules_layer_modules_12_modules_rel_attn_parameters_v_,
        )
        l_self_modules_layer_modules_12_modules_rel_attn_parameters_v_ = None
        type_13 = pos_emb_4.type(torch.float32)
        k_head_r_12 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            type_13,
            l_self_modules_layer_modules_12_modules_rel_attn_parameters_r_,
        )
        type_13 = l_self_modules_layer_modules_12_modules_rel_attn_parameters_r_ = None
        add_85 = (
            q_head_h_12
            + l_self_modules_layer_modules_12_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_12_modules_rel_attn_parameters_r_w_bias_ = None
        ac_12 = torch.functional.einsum("ibnd,jbnd->bnij", add_85, k_head_h_12)
        add_85 = k_head_h_12 = None
        add_86 = (
            q_head_h_12
            + l_self_modules_layer_modules_12_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_12_modules_rel_attn_parameters_r_r_bias_ = None
        bd_12 = torch.functional.einsum("ibnd,jbnd->bnij", add_86, k_head_r_12)
        add_86 = k_head_r_12 = None
        x_48 = bd_12.reshape(1, 16, 26, 13)
        bd_12 = None
        x_49 = x_48[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        x_48 = None
        x_50 = x_49.reshape(1, 16, 13, 25)
        x_49 = None
        arange_14 = torch.arange(
            13, device=device(type="cuda", index=0), dtype=torch.int64
        )
        x_51 = torch.index_select(x_50, 3, arange_14)
        x_50 = arange_14 = None
        add_87 = (
            q_head_h_12
            + l_self_modules_layer_modules_12_modules_rel_attn_parameters_r_s_bias_
        )
        q_head_h_12 = (
            l_self_modules_layer_modules_12_modules_rel_attn_parameters_r_s_bias_
        ) = None
        ef_24 = torch.functional.einsum(
            "ibnd,snd->ibns",
            add_87,
            l_self_modules_layer_modules_12_modules_rel_attn_parameters_seg_embed_,
        )
        add_87 = (
            l_self_modules_layer_modules_12_modules_rel_attn_parameters_seg_embed_
        ) = None
        ef_25 = torch.functional.einsum("ijbs,ibns->bnij", seg_mat_1, ef_24)
        ef_24 = None
        add_88 = ac_12 + x_51
        ac_12 = x_51 = None
        add_89 = add_88 + ef_25
        add_88 = ef_25 = None
        attn_score_24 = add_89 * 0.125
        add_89 = None
        einsum_141 = torch.functional.einsum("ijbn->bnij", non_tgt_mask_1)
        mul_25 = 1e30 * einsum_141
        einsum_141 = None
        attn_score_25 = attn_score_24 - mul_25
        attn_score_24 = mul_25 = None
        attn_prob_24 = torch.nn.functional.softmax(attn_score_25, dim=3)
        attn_score_25 = None
        attn_prob_25 = torch.nn.functional.dropout(attn_prob_24, 0.1, False, False)
        attn_prob_24 = None
        attn_vec_12 = torch.functional.einsum(
            "bnij,jbnd->ibnd", attn_prob_25, v_head_h_12
        )
        attn_prob_25 = v_head_h_12 = None
        attn_out_36 = torch.functional.einsum(
            "ibnd,hnd->ibh",
            attn_vec_12,
            l_self_modules_layer_modules_12_modules_rel_attn_parameters_o_,
        )
        attn_vec_12 = (
            l_self_modules_layer_modules_12_modules_rel_attn_parameters_o_
        ) = None
        attn_out_37 = torch.nn.functional.dropout(attn_out_36, 0.1, False, False)
        attn_out_36 = None
        attn_out_38 = attn_out_37 + output_83
        attn_out_37 = output_83 = None
        output_84 = torch.nn.functional.layer_norm(
            attn_out_38,
            (1024,),
            l_self_modules_layer_modules_12_modules_rel_attn_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_12_modules_rel_attn_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        attn_out_38 = l_self_modules_layer_modules_12_modules_rel_attn_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_12_modules_rel_attn_modules_layer_norm_parameters_bias_ = (None)
        output_85 = torch._C._nn.linear(
            output_84,
            l_self_modules_layer_modules_12_modules_ff_modules_layer_1_parameters_weight_,
            l_self_modules_layer_modules_12_modules_ff_modules_layer_1_parameters_bias_,
        )
        l_self_modules_layer_modules_12_modules_ff_modules_layer_1_parameters_weight_ = (
            l_self_modules_layer_modules_12_modules_ff_modules_layer_1_parameters_bias_
        ) = None
        output_86 = torch._C._nn.gelu(output_85)
        output_85 = None
        output_87 = torch.nn.functional.dropout(output_86, 0.1, False, False)
        output_86 = None
        output_88 = torch._C._nn.linear(
            output_87,
            l_self_modules_layer_modules_12_modules_ff_modules_layer_2_parameters_weight_,
            l_self_modules_layer_modules_12_modules_ff_modules_layer_2_parameters_bias_,
        )
        output_87 = l_self_modules_layer_modules_12_modules_ff_modules_layer_2_parameters_weight_ = (
            l_self_modules_layer_modules_12_modules_ff_modules_layer_2_parameters_bias_
        ) = None
        output_89 = torch.nn.functional.dropout(output_88, 0.1, False, False)
        output_88 = None
        add_91 = output_89 + output_84
        output_89 = output_84 = None
        output_90 = torch.nn.functional.layer_norm(
            add_91,
            (1024,),
            l_self_modules_layer_modules_12_modules_ff_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_12_modules_ff_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_91 = l_self_modules_layer_modules_12_modules_ff_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_12_modules_ff_modules_layer_norm_parameters_bias_ = (None)
        new_mem_13 = output_90[slice(0, None, None)]
        detach_13 = new_mem_13.detach()
        new_mem_13 = None
        q_head_h_13 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_90,
            l_self_modules_layer_modules_13_modules_rel_attn_parameters_q_,
        )
        l_self_modules_layer_modules_13_modules_rel_attn_parameters_q_ = None
        k_head_h_13 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_90,
            l_self_modules_layer_modules_13_modules_rel_attn_parameters_k_,
        )
        l_self_modules_layer_modules_13_modules_rel_attn_parameters_k_ = None
        v_head_h_13 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_90,
            l_self_modules_layer_modules_13_modules_rel_attn_parameters_v_,
        )
        l_self_modules_layer_modules_13_modules_rel_attn_parameters_v_ = None
        type_14 = pos_emb_4.type(torch.float32)
        k_head_r_13 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            type_14,
            l_self_modules_layer_modules_13_modules_rel_attn_parameters_r_,
        )
        type_14 = l_self_modules_layer_modules_13_modules_rel_attn_parameters_r_ = None
        add_92 = (
            q_head_h_13
            + l_self_modules_layer_modules_13_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_13_modules_rel_attn_parameters_r_w_bias_ = None
        ac_13 = torch.functional.einsum("ibnd,jbnd->bnij", add_92, k_head_h_13)
        add_92 = k_head_h_13 = None
        add_93 = (
            q_head_h_13
            + l_self_modules_layer_modules_13_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_13_modules_rel_attn_parameters_r_r_bias_ = None
        bd_13 = torch.functional.einsum("ibnd,jbnd->bnij", add_93, k_head_r_13)
        add_93 = k_head_r_13 = None
        x_52 = bd_13.reshape(1, 16, 26, 13)
        bd_13 = None
        x_53 = x_52[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        x_52 = None
        x_54 = x_53.reshape(1, 16, 13, 25)
        x_53 = None
        arange_15 = torch.arange(
            13, device=device(type="cuda", index=0), dtype=torch.int64
        )
        x_55 = torch.index_select(x_54, 3, arange_15)
        x_54 = arange_15 = None
        add_94 = (
            q_head_h_13
            + l_self_modules_layer_modules_13_modules_rel_attn_parameters_r_s_bias_
        )
        q_head_h_13 = (
            l_self_modules_layer_modules_13_modules_rel_attn_parameters_r_s_bias_
        ) = None
        ef_26 = torch.functional.einsum(
            "ibnd,snd->ibns",
            add_94,
            l_self_modules_layer_modules_13_modules_rel_attn_parameters_seg_embed_,
        )
        add_94 = (
            l_self_modules_layer_modules_13_modules_rel_attn_parameters_seg_embed_
        ) = None
        ef_27 = torch.functional.einsum("ijbs,ibns->bnij", seg_mat_1, ef_26)
        ef_26 = None
        add_95 = ac_13 + x_55
        ac_13 = x_55 = None
        add_96 = add_95 + ef_27
        add_95 = ef_27 = None
        attn_score_26 = add_96 * 0.125
        add_96 = None
        einsum_152 = torch.functional.einsum("ijbn->bnij", non_tgt_mask_1)
        mul_27 = 1e30 * einsum_152
        einsum_152 = None
        attn_score_27 = attn_score_26 - mul_27
        attn_score_26 = mul_27 = None
        attn_prob_26 = torch.nn.functional.softmax(attn_score_27, dim=3)
        attn_score_27 = None
        attn_prob_27 = torch.nn.functional.dropout(attn_prob_26, 0.1, False, False)
        attn_prob_26 = None
        attn_vec_13 = torch.functional.einsum(
            "bnij,jbnd->ibnd", attn_prob_27, v_head_h_13
        )
        attn_prob_27 = v_head_h_13 = None
        attn_out_39 = torch.functional.einsum(
            "ibnd,hnd->ibh",
            attn_vec_13,
            l_self_modules_layer_modules_13_modules_rel_attn_parameters_o_,
        )
        attn_vec_13 = (
            l_self_modules_layer_modules_13_modules_rel_attn_parameters_o_
        ) = None
        attn_out_40 = torch.nn.functional.dropout(attn_out_39, 0.1, False, False)
        attn_out_39 = None
        attn_out_41 = attn_out_40 + output_90
        attn_out_40 = output_90 = None
        output_91 = torch.nn.functional.layer_norm(
            attn_out_41,
            (1024,),
            l_self_modules_layer_modules_13_modules_rel_attn_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_13_modules_rel_attn_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        attn_out_41 = l_self_modules_layer_modules_13_modules_rel_attn_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_13_modules_rel_attn_modules_layer_norm_parameters_bias_ = (None)
        output_92 = torch._C._nn.linear(
            output_91,
            l_self_modules_layer_modules_13_modules_ff_modules_layer_1_parameters_weight_,
            l_self_modules_layer_modules_13_modules_ff_modules_layer_1_parameters_bias_,
        )
        l_self_modules_layer_modules_13_modules_ff_modules_layer_1_parameters_weight_ = (
            l_self_modules_layer_modules_13_modules_ff_modules_layer_1_parameters_bias_
        ) = None
        output_93 = torch._C._nn.gelu(output_92)
        output_92 = None
        output_94 = torch.nn.functional.dropout(output_93, 0.1, False, False)
        output_93 = None
        output_95 = torch._C._nn.linear(
            output_94,
            l_self_modules_layer_modules_13_modules_ff_modules_layer_2_parameters_weight_,
            l_self_modules_layer_modules_13_modules_ff_modules_layer_2_parameters_bias_,
        )
        output_94 = l_self_modules_layer_modules_13_modules_ff_modules_layer_2_parameters_weight_ = (
            l_self_modules_layer_modules_13_modules_ff_modules_layer_2_parameters_bias_
        ) = None
        output_96 = torch.nn.functional.dropout(output_95, 0.1, False, False)
        output_95 = None
        add_98 = output_96 + output_91
        output_96 = output_91 = None
        output_97 = torch.nn.functional.layer_norm(
            add_98,
            (1024,),
            l_self_modules_layer_modules_13_modules_ff_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_13_modules_ff_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_98 = l_self_modules_layer_modules_13_modules_ff_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_13_modules_ff_modules_layer_norm_parameters_bias_ = (None)
        new_mem_14 = output_97[slice(0, None, None)]
        detach_14 = new_mem_14.detach()
        new_mem_14 = None
        q_head_h_14 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_97,
            l_self_modules_layer_modules_14_modules_rel_attn_parameters_q_,
        )
        l_self_modules_layer_modules_14_modules_rel_attn_parameters_q_ = None
        k_head_h_14 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_97,
            l_self_modules_layer_modules_14_modules_rel_attn_parameters_k_,
        )
        l_self_modules_layer_modules_14_modules_rel_attn_parameters_k_ = None
        v_head_h_14 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_97,
            l_self_modules_layer_modules_14_modules_rel_attn_parameters_v_,
        )
        l_self_modules_layer_modules_14_modules_rel_attn_parameters_v_ = None
        type_15 = pos_emb_4.type(torch.float32)
        k_head_r_14 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            type_15,
            l_self_modules_layer_modules_14_modules_rel_attn_parameters_r_,
        )
        type_15 = l_self_modules_layer_modules_14_modules_rel_attn_parameters_r_ = None
        add_99 = (
            q_head_h_14
            + l_self_modules_layer_modules_14_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_14_modules_rel_attn_parameters_r_w_bias_ = None
        ac_14 = torch.functional.einsum("ibnd,jbnd->bnij", add_99, k_head_h_14)
        add_99 = k_head_h_14 = None
        add_100 = (
            q_head_h_14
            + l_self_modules_layer_modules_14_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_14_modules_rel_attn_parameters_r_r_bias_ = None
        bd_14 = torch.functional.einsum("ibnd,jbnd->bnij", add_100, k_head_r_14)
        add_100 = k_head_r_14 = None
        x_56 = bd_14.reshape(1, 16, 26, 13)
        bd_14 = None
        x_57 = x_56[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        x_56 = None
        x_58 = x_57.reshape(1, 16, 13, 25)
        x_57 = None
        arange_16 = torch.arange(
            13, device=device(type="cuda", index=0), dtype=torch.int64
        )
        x_59 = torch.index_select(x_58, 3, arange_16)
        x_58 = arange_16 = None
        add_101 = (
            q_head_h_14
            + l_self_modules_layer_modules_14_modules_rel_attn_parameters_r_s_bias_
        )
        q_head_h_14 = (
            l_self_modules_layer_modules_14_modules_rel_attn_parameters_r_s_bias_
        ) = None
        ef_28 = torch.functional.einsum(
            "ibnd,snd->ibns",
            add_101,
            l_self_modules_layer_modules_14_modules_rel_attn_parameters_seg_embed_,
        )
        add_101 = (
            l_self_modules_layer_modules_14_modules_rel_attn_parameters_seg_embed_
        ) = None
        ef_29 = torch.functional.einsum("ijbs,ibns->bnij", seg_mat_1, ef_28)
        ef_28 = None
        add_102 = ac_14 + x_59
        ac_14 = x_59 = None
        add_103 = add_102 + ef_29
        add_102 = ef_29 = None
        attn_score_28 = add_103 * 0.125
        add_103 = None
        einsum_163 = torch.functional.einsum("ijbn->bnij", non_tgt_mask_1)
        mul_29 = 1e30 * einsum_163
        einsum_163 = None
        attn_score_29 = attn_score_28 - mul_29
        attn_score_28 = mul_29 = None
        attn_prob_28 = torch.nn.functional.softmax(attn_score_29, dim=3)
        attn_score_29 = None
        attn_prob_29 = torch.nn.functional.dropout(attn_prob_28, 0.1, False, False)
        attn_prob_28 = None
        attn_vec_14 = torch.functional.einsum(
            "bnij,jbnd->ibnd", attn_prob_29, v_head_h_14
        )
        attn_prob_29 = v_head_h_14 = None
        attn_out_42 = torch.functional.einsum(
            "ibnd,hnd->ibh",
            attn_vec_14,
            l_self_modules_layer_modules_14_modules_rel_attn_parameters_o_,
        )
        attn_vec_14 = (
            l_self_modules_layer_modules_14_modules_rel_attn_parameters_o_
        ) = None
        attn_out_43 = torch.nn.functional.dropout(attn_out_42, 0.1, False, False)
        attn_out_42 = None
        attn_out_44 = attn_out_43 + output_97
        attn_out_43 = output_97 = None
        output_98 = torch.nn.functional.layer_norm(
            attn_out_44,
            (1024,),
            l_self_modules_layer_modules_14_modules_rel_attn_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_14_modules_rel_attn_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        attn_out_44 = l_self_modules_layer_modules_14_modules_rel_attn_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_14_modules_rel_attn_modules_layer_norm_parameters_bias_ = (None)
        output_99 = torch._C._nn.linear(
            output_98,
            l_self_modules_layer_modules_14_modules_ff_modules_layer_1_parameters_weight_,
            l_self_modules_layer_modules_14_modules_ff_modules_layer_1_parameters_bias_,
        )
        l_self_modules_layer_modules_14_modules_ff_modules_layer_1_parameters_weight_ = (
            l_self_modules_layer_modules_14_modules_ff_modules_layer_1_parameters_bias_
        ) = None
        output_100 = torch._C._nn.gelu(output_99)
        output_99 = None
        output_101 = torch.nn.functional.dropout(output_100, 0.1, False, False)
        output_100 = None
        output_102 = torch._C._nn.linear(
            output_101,
            l_self_modules_layer_modules_14_modules_ff_modules_layer_2_parameters_weight_,
            l_self_modules_layer_modules_14_modules_ff_modules_layer_2_parameters_bias_,
        )
        output_101 = l_self_modules_layer_modules_14_modules_ff_modules_layer_2_parameters_weight_ = (
            l_self_modules_layer_modules_14_modules_ff_modules_layer_2_parameters_bias_
        ) = None
        output_103 = torch.nn.functional.dropout(output_102, 0.1, False, False)
        output_102 = None
        add_105 = output_103 + output_98
        output_103 = output_98 = None
        output_104 = torch.nn.functional.layer_norm(
            add_105,
            (1024,),
            l_self_modules_layer_modules_14_modules_ff_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_14_modules_ff_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_105 = l_self_modules_layer_modules_14_modules_ff_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_14_modules_ff_modules_layer_norm_parameters_bias_ = (None)
        new_mem_15 = output_104[slice(0, None, None)]
        detach_15 = new_mem_15.detach()
        new_mem_15 = None
        q_head_h_15 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_104,
            l_self_modules_layer_modules_15_modules_rel_attn_parameters_q_,
        )
        l_self_modules_layer_modules_15_modules_rel_attn_parameters_q_ = None
        k_head_h_15 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_104,
            l_self_modules_layer_modules_15_modules_rel_attn_parameters_k_,
        )
        l_self_modules_layer_modules_15_modules_rel_attn_parameters_k_ = None
        v_head_h_15 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_104,
            l_self_modules_layer_modules_15_modules_rel_attn_parameters_v_,
        )
        l_self_modules_layer_modules_15_modules_rel_attn_parameters_v_ = None
        type_16 = pos_emb_4.type(torch.float32)
        k_head_r_15 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            type_16,
            l_self_modules_layer_modules_15_modules_rel_attn_parameters_r_,
        )
        type_16 = l_self_modules_layer_modules_15_modules_rel_attn_parameters_r_ = None
        add_106 = (
            q_head_h_15
            + l_self_modules_layer_modules_15_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_15_modules_rel_attn_parameters_r_w_bias_ = None
        ac_15 = torch.functional.einsum("ibnd,jbnd->bnij", add_106, k_head_h_15)
        add_106 = k_head_h_15 = None
        add_107 = (
            q_head_h_15
            + l_self_modules_layer_modules_15_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_15_modules_rel_attn_parameters_r_r_bias_ = None
        bd_15 = torch.functional.einsum("ibnd,jbnd->bnij", add_107, k_head_r_15)
        add_107 = k_head_r_15 = None
        x_60 = bd_15.reshape(1, 16, 26, 13)
        bd_15 = None
        x_61 = x_60[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        x_60 = None
        x_62 = x_61.reshape(1, 16, 13, 25)
        x_61 = None
        arange_17 = torch.arange(
            13, device=device(type="cuda", index=0), dtype=torch.int64
        )
        x_63 = torch.index_select(x_62, 3, arange_17)
        x_62 = arange_17 = None
        add_108 = (
            q_head_h_15
            + l_self_modules_layer_modules_15_modules_rel_attn_parameters_r_s_bias_
        )
        q_head_h_15 = (
            l_self_modules_layer_modules_15_modules_rel_attn_parameters_r_s_bias_
        ) = None
        ef_30 = torch.functional.einsum(
            "ibnd,snd->ibns",
            add_108,
            l_self_modules_layer_modules_15_modules_rel_attn_parameters_seg_embed_,
        )
        add_108 = (
            l_self_modules_layer_modules_15_modules_rel_attn_parameters_seg_embed_
        ) = None
        ef_31 = torch.functional.einsum("ijbs,ibns->bnij", seg_mat_1, ef_30)
        ef_30 = None
        add_109 = ac_15 + x_63
        ac_15 = x_63 = None
        add_110 = add_109 + ef_31
        add_109 = ef_31 = None
        attn_score_30 = add_110 * 0.125
        add_110 = None
        einsum_174 = torch.functional.einsum("ijbn->bnij", non_tgt_mask_1)
        mul_31 = 1e30 * einsum_174
        einsum_174 = None
        attn_score_31 = attn_score_30 - mul_31
        attn_score_30 = mul_31 = None
        attn_prob_30 = torch.nn.functional.softmax(attn_score_31, dim=3)
        attn_score_31 = None
        attn_prob_31 = torch.nn.functional.dropout(attn_prob_30, 0.1, False, False)
        attn_prob_30 = None
        attn_vec_15 = torch.functional.einsum(
            "bnij,jbnd->ibnd", attn_prob_31, v_head_h_15
        )
        attn_prob_31 = v_head_h_15 = None
        attn_out_45 = torch.functional.einsum(
            "ibnd,hnd->ibh",
            attn_vec_15,
            l_self_modules_layer_modules_15_modules_rel_attn_parameters_o_,
        )
        attn_vec_15 = (
            l_self_modules_layer_modules_15_modules_rel_attn_parameters_o_
        ) = None
        attn_out_46 = torch.nn.functional.dropout(attn_out_45, 0.1, False, False)
        attn_out_45 = None
        attn_out_47 = attn_out_46 + output_104
        attn_out_46 = output_104 = None
        output_105 = torch.nn.functional.layer_norm(
            attn_out_47,
            (1024,),
            l_self_modules_layer_modules_15_modules_rel_attn_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_15_modules_rel_attn_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        attn_out_47 = l_self_modules_layer_modules_15_modules_rel_attn_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_15_modules_rel_attn_modules_layer_norm_parameters_bias_ = (None)
        output_106 = torch._C._nn.linear(
            output_105,
            l_self_modules_layer_modules_15_modules_ff_modules_layer_1_parameters_weight_,
            l_self_modules_layer_modules_15_modules_ff_modules_layer_1_parameters_bias_,
        )
        l_self_modules_layer_modules_15_modules_ff_modules_layer_1_parameters_weight_ = (
            l_self_modules_layer_modules_15_modules_ff_modules_layer_1_parameters_bias_
        ) = None
        output_107 = torch._C._nn.gelu(output_106)
        output_106 = None
        output_108 = torch.nn.functional.dropout(output_107, 0.1, False, False)
        output_107 = None
        output_109 = torch._C._nn.linear(
            output_108,
            l_self_modules_layer_modules_15_modules_ff_modules_layer_2_parameters_weight_,
            l_self_modules_layer_modules_15_modules_ff_modules_layer_2_parameters_bias_,
        )
        output_108 = l_self_modules_layer_modules_15_modules_ff_modules_layer_2_parameters_weight_ = (
            l_self_modules_layer_modules_15_modules_ff_modules_layer_2_parameters_bias_
        ) = None
        output_110 = torch.nn.functional.dropout(output_109, 0.1, False, False)
        output_109 = None
        add_112 = output_110 + output_105
        output_110 = output_105 = None
        output_111 = torch.nn.functional.layer_norm(
            add_112,
            (1024,),
            l_self_modules_layer_modules_15_modules_ff_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_15_modules_ff_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_112 = l_self_modules_layer_modules_15_modules_ff_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_15_modules_ff_modules_layer_norm_parameters_bias_ = (None)
        new_mem_16 = output_111[slice(0, None, None)]
        detach_16 = new_mem_16.detach()
        new_mem_16 = None
        q_head_h_16 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_111,
            l_self_modules_layer_modules_16_modules_rel_attn_parameters_q_,
        )
        l_self_modules_layer_modules_16_modules_rel_attn_parameters_q_ = None
        k_head_h_16 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_111,
            l_self_modules_layer_modules_16_modules_rel_attn_parameters_k_,
        )
        l_self_modules_layer_modules_16_modules_rel_attn_parameters_k_ = None
        v_head_h_16 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_111,
            l_self_modules_layer_modules_16_modules_rel_attn_parameters_v_,
        )
        l_self_modules_layer_modules_16_modules_rel_attn_parameters_v_ = None
        type_17 = pos_emb_4.type(torch.float32)
        k_head_r_16 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            type_17,
            l_self_modules_layer_modules_16_modules_rel_attn_parameters_r_,
        )
        type_17 = l_self_modules_layer_modules_16_modules_rel_attn_parameters_r_ = None
        add_113 = (
            q_head_h_16
            + l_self_modules_layer_modules_16_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_16_modules_rel_attn_parameters_r_w_bias_ = None
        ac_16 = torch.functional.einsum("ibnd,jbnd->bnij", add_113, k_head_h_16)
        add_113 = k_head_h_16 = None
        add_114 = (
            q_head_h_16
            + l_self_modules_layer_modules_16_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_16_modules_rel_attn_parameters_r_r_bias_ = None
        bd_16 = torch.functional.einsum("ibnd,jbnd->bnij", add_114, k_head_r_16)
        add_114 = k_head_r_16 = None
        x_64 = bd_16.reshape(1, 16, 26, 13)
        bd_16 = None
        x_65 = x_64[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        x_64 = None
        x_66 = x_65.reshape(1, 16, 13, 25)
        x_65 = None
        arange_18 = torch.arange(
            13, device=device(type="cuda", index=0), dtype=torch.int64
        )
        x_67 = torch.index_select(x_66, 3, arange_18)
        x_66 = arange_18 = None
        add_115 = (
            q_head_h_16
            + l_self_modules_layer_modules_16_modules_rel_attn_parameters_r_s_bias_
        )
        q_head_h_16 = (
            l_self_modules_layer_modules_16_modules_rel_attn_parameters_r_s_bias_
        ) = None
        ef_32 = torch.functional.einsum(
            "ibnd,snd->ibns",
            add_115,
            l_self_modules_layer_modules_16_modules_rel_attn_parameters_seg_embed_,
        )
        add_115 = (
            l_self_modules_layer_modules_16_modules_rel_attn_parameters_seg_embed_
        ) = None
        ef_33 = torch.functional.einsum("ijbs,ibns->bnij", seg_mat_1, ef_32)
        ef_32 = None
        add_116 = ac_16 + x_67
        ac_16 = x_67 = None
        add_117 = add_116 + ef_33
        add_116 = ef_33 = None
        attn_score_32 = add_117 * 0.125
        add_117 = None
        einsum_185 = torch.functional.einsum("ijbn->bnij", non_tgt_mask_1)
        mul_33 = 1e30 * einsum_185
        einsum_185 = None
        attn_score_33 = attn_score_32 - mul_33
        attn_score_32 = mul_33 = None
        attn_prob_32 = torch.nn.functional.softmax(attn_score_33, dim=3)
        attn_score_33 = None
        attn_prob_33 = torch.nn.functional.dropout(attn_prob_32, 0.1, False, False)
        attn_prob_32 = None
        attn_vec_16 = torch.functional.einsum(
            "bnij,jbnd->ibnd", attn_prob_33, v_head_h_16
        )
        attn_prob_33 = v_head_h_16 = None
        attn_out_48 = torch.functional.einsum(
            "ibnd,hnd->ibh",
            attn_vec_16,
            l_self_modules_layer_modules_16_modules_rel_attn_parameters_o_,
        )
        attn_vec_16 = (
            l_self_modules_layer_modules_16_modules_rel_attn_parameters_o_
        ) = None
        attn_out_49 = torch.nn.functional.dropout(attn_out_48, 0.1, False, False)
        attn_out_48 = None
        attn_out_50 = attn_out_49 + output_111
        attn_out_49 = output_111 = None
        output_112 = torch.nn.functional.layer_norm(
            attn_out_50,
            (1024,),
            l_self_modules_layer_modules_16_modules_rel_attn_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_16_modules_rel_attn_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        attn_out_50 = l_self_modules_layer_modules_16_modules_rel_attn_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_16_modules_rel_attn_modules_layer_norm_parameters_bias_ = (None)
        output_113 = torch._C._nn.linear(
            output_112,
            l_self_modules_layer_modules_16_modules_ff_modules_layer_1_parameters_weight_,
            l_self_modules_layer_modules_16_modules_ff_modules_layer_1_parameters_bias_,
        )
        l_self_modules_layer_modules_16_modules_ff_modules_layer_1_parameters_weight_ = (
            l_self_modules_layer_modules_16_modules_ff_modules_layer_1_parameters_bias_
        ) = None
        output_114 = torch._C._nn.gelu(output_113)
        output_113 = None
        output_115 = torch.nn.functional.dropout(output_114, 0.1, False, False)
        output_114 = None
        output_116 = torch._C._nn.linear(
            output_115,
            l_self_modules_layer_modules_16_modules_ff_modules_layer_2_parameters_weight_,
            l_self_modules_layer_modules_16_modules_ff_modules_layer_2_parameters_bias_,
        )
        output_115 = l_self_modules_layer_modules_16_modules_ff_modules_layer_2_parameters_weight_ = (
            l_self_modules_layer_modules_16_modules_ff_modules_layer_2_parameters_bias_
        ) = None
        output_117 = torch.nn.functional.dropout(output_116, 0.1, False, False)
        output_116 = None
        add_119 = output_117 + output_112
        output_117 = output_112 = None
        output_118 = torch.nn.functional.layer_norm(
            add_119,
            (1024,),
            l_self_modules_layer_modules_16_modules_ff_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_16_modules_ff_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_119 = l_self_modules_layer_modules_16_modules_ff_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_16_modules_ff_modules_layer_norm_parameters_bias_ = (None)
        new_mem_17 = output_118[slice(0, None, None)]
        detach_17 = new_mem_17.detach()
        new_mem_17 = None
        q_head_h_17 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_118,
            l_self_modules_layer_modules_17_modules_rel_attn_parameters_q_,
        )
        l_self_modules_layer_modules_17_modules_rel_attn_parameters_q_ = None
        k_head_h_17 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_118,
            l_self_modules_layer_modules_17_modules_rel_attn_parameters_k_,
        )
        l_self_modules_layer_modules_17_modules_rel_attn_parameters_k_ = None
        v_head_h_17 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_118,
            l_self_modules_layer_modules_17_modules_rel_attn_parameters_v_,
        )
        l_self_modules_layer_modules_17_modules_rel_attn_parameters_v_ = None
        type_18 = pos_emb_4.type(torch.float32)
        k_head_r_17 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            type_18,
            l_self_modules_layer_modules_17_modules_rel_attn_parameters_r_,
        )
        type_18 = l_self_modules_layer_modules_17_modules_rel_attn_parameters_r_ = None
        add_120 = (
            q_head_h_17
            + l_self_modules_layer_modules_17_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_17_modules_rel_attn_parameters_r_w_bias_ = None
        ac_17 = torch.functional.einsum("ibnd,jbnd->bnij", add_120, k_head_h_17)
        add_120 = k_head_h_17 = None
        add_121 = (
            q_head_h_17
            + l_self_modules_layer_modules_17_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_17_modules_rel_attn_parameters_r_r_bias_ = None
        bd_17 = torch.functional.einsum("ibnd,jbnd->bnij", add_121, k_head_r_17)
        add_121 = k_head_r_17 = None
        x_68 = bd_17.reshape(1, 16, 26, 13)
        bd_17 = None
        x_69 = x_68[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        x_68 = None
        x_70 = x_69.reshape(1, 16, 13, 25)
        x_69 = None
        arange_19 = torch.arange(
            13, device=device(type="cuda", index=0), dtype=torch.int64
        )
        x_71 = torch.index_select(x_70, 3, arange_19)
        x_70 = arange_19 = None
        add_122 = (
            q_head_h_17
            + l_self_modules_layer_modules_17_modules_rel_attn_parameters_r_s_bias_
        )
        q_head_h_17 = (
            l_self_modules_layer_modules_17_modules_rel_attn_parameters_r_s_bias_
        ) = None
        ef_34 = torch.functional.einsum(
            "ibnd,snd->ibns",
            add_122,
            l_self_modules_layer_modules_17_modules_rel_attn_parameters_seg_embed_,
        )
        add_122 = (
            l_self_modules_layer_modules_17_modules_rel_attn_parameters_seg_embed_
        ) = None
        ef_35 = torch.functional.einsum("ijbs,ibns->bnij", seg_mat_1, ef_34)
        ef_34 = None
        add_123 = ac_17 + x_71
        ac_17 = x_71 = None
        add_124 = add_123 + ef_35
        add_123 = ef_35 = None
        attn_score_34 = add_124 * 0.125
        add_124 = None
        einsum_196 = torch.functional.einsum("ijbn->bnij", non_tgt_mask_1)
        mul_35 = 1e30 * einsum_196
        einsum_196 = None
        attn_score_35 = attn_score_34 - mul_35
        attn_score_34 = mul_35 = None
        attn_prob_34 = torch.nn.functional.softmax(attn_score_35, dim=3)
        attn_score_35 = None
        attn_prob_35 = torch.nn.functional.dropout(attn_prob_34, 0.1, False, False)
        attn_prob_34 = None
        attn_vec_17 = torch.functional.einsum(
            "bnij,jbnd->ibnd", attn_prob_35, v_head_h_17
        )
        attn_prob_35 = v_head_h_17 = None
        attn_out_51 = torch.functional.einsum(
            "ibnd,hnd->ibh",
            attn_vec_17,
            l_self_modules_layer_modules_17_modules_rel_attn_parameters_o_,
        )
        attn_vec_17 = (
            l_self_modules_layer_modules_17_modules_rel_attn_parameters_o_
        ) = None
        attn_out_52 = torch.nn.functional.dropout(attn_out_51, 0.1, False, False)
        attn_out_51 = None
        attn_out_53 = attn_out_52 + output_118
        attn_out_52 = output_118 = None
        output_119 = torch.nn.functional.layer_norm(
            attn_out_53,
            (1024,),
            l_self_modules_layer_modules_17_modules_rel_attn_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_17_modules_rel_attn_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        attn_out_53 = l_self_modules_layer_modules_17_modules_rel_attn_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_17_modules_rel_attn_modules_layer_norm_parameters_bias_ = (None)
        output_120 = torch._C._nn.linear(
            output_119,
            l_self_modules_layer_modules_17_modules_ff_modules_layer_1_parameters_weight_,
            l_self_modules_layer_modules_17_modules_ff_modules_layer_1_parameters_bias_,
        )
        l_self_modules_layer_modules_17_modules_ff_modules_layer_1_parameters_weight_ = (
            l_self_modules_layer_modules_17_modules_ff_modules_layer_1_parameters_bias_
        ) = None
        output_121 = torch._C._nn.gelu(output_120)
        output_120 = None
        output_122 = torch.nn.functional.dropout(output_121, 0.1, False, False)
        output_121 = None
        output_123 = torch._C._nn.linear(
            output_122,
            l_self_modules_layer_modules_17_modules_ff_modules_layer_2_parameters_weight_,
            l_self_modules_layer_modules_17_modules_ff_modules_layer_2_parameters_bias_,
        )
        output_122 = l_self_modules_layer_modules_17_modules_ff_modules_layer_2_parameters_weight_ = (
            l_self_modules_layer_modules_17_modules_ff_modules_layer_2_parameters_bias_
        ) = None
        output_124 = torch.nn.functional.dropout(output_123, 0.1, False, False)
        output_123 = None
        add_126 = output_124 + output_119
        output_124 = output_119 = None
        output_125 = torch.nn.functional.layer_norm(
            add_126,
            (1024,),
            l_self_modules_layer_modules_17_modules_ff_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_17_modules_ff_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_126 = l_self_modules_layer_modules_17_modules_ff_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_17_modules_ff_modules_layer_norm_parameters_bias_ = (None)
        new_mem_18 = output_125[slice(0, None, None)]
        detach_18 = new_mem_18.detach()
        new_mem_18 = None
        q_head_h_18 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_125,
            l_self_modules_layer_modules_18_modules_rel_attn_parameters_q_,
        )
        l_self_modules_layer_modules_18_modules_rel_attn_parameters_q_ = None
        k_head_h_18 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_125,
            l_self_modules_layer_modules_18_modules_rel_attn_parameters_k_,
        )
        l_self_modules_layer_modules_18_modules_rel_attn_parameters_k_ = None
        v_head_h_18 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_125,
            l_self_modules_layer_modules_18_modules_rel_attn_parameters_v_,
        )
        l_self_modules_layer_modules_18_modules_rel_attn_parameters_v_ = None
        type_19 = pos_emb_4.type(torch.float32)
        k_head_r_18 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            type_19,
            l_self_modules_layer_modules_18_modules_rel_attn_parameters_r_,
        )
        type_19 = l_self_modules_layer_modules_18_modules_rel_attn_parameters_r_ = None
        add_127 = (
            q_head_h_18
            + l_self_modules_layer_modules_18_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_18_modules_rel_attn_parameters_r_w_bias_ = None
        ac_18 = torch.functional.einsum("ibnd,jbnd->bnij", add_127, k_head_h_18)
        add_127 = k_head_h_18 = None
        add_128 = (
            q_head_h_18
            + l_self_modules_layer_modules_18_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_18_modules_rel_attn_parameters_r_r_bias_ = None
        bd_18 = torch.functional.einsum("ibnd,jbnd->bnij", add_128, k_head_r_18)
        add_128 = k_head_r_18 = None
        x_72 = bd_18.reshape(1, 16, 26, 13)
        bd_18 = None
        x_73 = x_72[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        x_72 = None
        x_74 = x_73.reshape(1, 16, 13, 25)
        x_73 = None
        arange_20 = torch.arange(
            13, device=device(type="cuda", index=0), dtype=torch.int64
        )
        x_75 = torch.index_select(x_74, 3, arange_20)
        x_74 = arange_20 = None
        add_129 = (
            q_head_h_18
            + l_self_modules_layer_modules_18_modules_rel_attn_parameters_r_s_bias_
        )
        q_head_h_18 = (
            l_self_modules_layer_modules_18_modules_rel_attn_parameters_r_s_bias_
        ) = None
        ef_36 = torch.functional.einsum(
            "ibnd,snd->ibns",
            add_129,
            l_self_modules_layer_modules_18_modules_rel_attn_parameters_seg_embed_,
        )
        add_129 = (
            l_self_modules_layer_modules_18_modules_rel_attn_parameters_seg_embed_
        ) = None
        ef_37 = torch.functional.einsum("ijbs,ibns->bnij", seg_mat_1, ef_36)
        ef_36 = None
        add_130 = ac_18 + x_75
        ac_18 = x_75 = None
        add_131 = add_130 + ef_37
        add_130 = ef_37 = None
        attn_score_36 = add_131 * 0.125
        add_131 = None
        einsum_207 = torch.functional.einsum("ijbn->bnij", non_tgt_mask_1)
        mul_37 = 1e30 * einsum_207
        einsum_207 = None
        attn_score_37 = attn_score_36 - mul_37
        attn_score_36 = mul_37 = None
        attn_prob_36 = torch.nn.functional.softmax(attn_score_37, dim=3)
        attn_score_37 = None
        attn_prob_37 = torch.nn.functional.dropout(attn_prob_36, 0.1, False, False)
        attn_prob_36 = None
        attn_vec_18 = torch.functional.einsum(
            "bnij,jbnd->ibnd", attn_prob_37, v_head_h_18
        )
        attn_prob_37 = v_head_h_18 = None
        attn_out_54 = torch.functional.einsum(
            "ibnd,hnd->ibh",
            attn_vec_18,
            l_self_modules_layer_modules_18_modules_rel_attn_parameters_o_,
        )
        attn_vec_18 = (
            l_self_modules_layer_modules_18_modules_rel_attn_parameters_o_
        ) = None
        attn_out_55 = torch.nn.functional.dropout(attn_out_54, 0.1, False, False)
        attn_out_54 = None
        attn_out_56 = attn_out_55 + output_125
        attn_out_55 = output_125 = None
        output_126 = torch.nn.functional.layer_norm(
            attn_out_56,
            (1024,),
            l_self_modules_layer_modules_18_modules_rel_attn_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_18_modules_rel_attn_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        attn_out_56 = l_self_modules_layer_modules_18_modules_rel_attn_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_18_modules_rel_attn_modules_layer_norm_parameters_bias_ = (None)
        output_127 = torch._C._nn.linear(
            output_126,
            l_self_modules_layer_modules_18_modules_ff_modules_layer_1_parameters_weight_,
            l_self_modules_layer_modules_18_modules_ff_modules_layer_1_parameters_bias_,
        )
        l_self_modules_layer_modules_18_modules_ff_modules_layer_1_parameters_weight_ = (
            l_self_modules_layer_modules_18_modules_ff_modules_layer_1_parameters_bias_
        ) = None
        output_128 = torch._C._nn.gelu(output_127)
        output_127 = None
        output_129 = torch.nn.functional.dropout(output_128, 0.1, False, False)
        output_128 = None
        output_130 = torch._C._nn.linear(
            output_129,
            l_self_modules_layer_modules_18_modules_ff_modules_layer_2_parameters_weight_,
            l_self_modules_layer_modules_18_modules_ff_modules_layer_2_parameters_bias_,
        )
        output_129 = l_self_modules_layer_modules_18_modules_ff_modules_layer_2_parameters_weight_ = (
            l_self_modules_layer_modules_18_modules_ff_modules_layer_2_parameters_bias_
        ) = None
        output_131 = torch.nn.functional.dropout(output_130, 0.1, False, False)
        output_130 = None
        add_133 = output_131 + output_126
        output_131 = output_126 = None
        output_132 = torch.nn.functional.layer_norm(
            add_133,
            (1024,),
            l_self_modules_layer_modules_18_modules_ff_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_18_modules_ff_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_133 = l_self_modules_layer_modules_18_modules_ff_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_18_modules_ff_modules_layer_norm_parameters_bias_ = (None)
        new_mem_19 = output_132[slice(0, None, None)]
        detach_19 = new_mem_19.detach()
        new_mem_19 = None
        q_head_h_19 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_132,
            l_self_modules_layer_modules_19_modules_rel_attn_parameters_q_,
        )
        l_self_modules_layer_modules_19_modules_rel_attn_parameters_q_ = None
        k_head_h_19 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_132,
            l_self_modules_layer_modules_19_modules_rel_attn_parameters_k_,
        )
        l_self_modules_layer_modules_19_modules_rel_attn_parameters_k_ = None
        v_head_h_19 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_132,
            l_self_modules_layer_modules_19_modules_rel_attn_parameters_v_,
        )
        l_self_modules_layer_modules_19_modules_rel_attn_parameters_v_ = None
        type_20 = pos_emb_4.type(torch.float32)
        k_head_r_19 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            type_20,
            l_self_modules_layer_modules_19_modules_rel_attn_parameters_r_,
        )
        type_20 = l_self_modules_layer_modules_19_modules_rel_attn_parameters_r_ = None
        add_134 = (
            q_head_h_19
            + l_self_modules_layer_modules_19_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_19_modules_rel_attn_parameters_r_w_bias_ = None
        ac_19 = torch.functional.einsum("ibnd,jbnd->bnij", add_134, k_head_h_19)
        add_134 = k_head_h_19 = None
        add_135 = (
            q_head_h_19
            + l_self_modules_layer_modules_19_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_19_modules_rel_attn_parameters_r_r_bias_ = None
        bd_19 = torch.functional.einsum("ibnd,jbnd->bnij", add_135, k_head_r_19)
        add_135 = k_head_r_19 = None
        x_76 = bd_19.reshape(1, 16, 26, 13)
        bd_19 = None
        x_77 = x_76[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        x_76 = None
        x_78 = x_77.reshape(1, 16, 13, 25)
        x_77 = None
        arange_21 = torch.arange(
            13, device=device(type="cuda", index=0), dtype=torch.int64
        )
        x_79 = torch.index_select(x_78, 3, arange_21)
        x_78 = arange_21 = None
        add_136 = (
            q_head_h_19
            + l_self_modules_layer_modules_19_modules_rel_attn_parameters_r_s_bias_
        )
        q_head_h_19 = (
            l_self_modules_layer_modules_19_modules_rel_attn_parameters_r_s_bias_
        ) = None
        ef_38 = torch.functional.einsum(
            "ibnd,snd->ibns",
            add_136,
            l_self_modules_layer_modules_19_modules_rel_attn_parameters_seg_embed_,
        )
        add_136 = (
            l_self_modules_layer_modules_19_modules_rel_attn_parameters_seg_embed_
        ) = None
        ef_39 = torch.functional.einsum("ijbs,ibns->bnij", seg_mat_1, ef_38)
        ef_38 = None
        add_137 = ac_19 + x_79
        ac_19 = x_79 = None
        add_138 = add_137 + ef_39
        add_137 = ef_39 = None
        attn_score_38 = add_138 * 0.125
        add_138 = None
        einsum_218 = torch.functional.einsum("ijbn->bnij", non_tgt_mask_1)
        mul_39 = 1e30 * einsum_218
        einsum_218 = None
        attn_score_39 = attn_score_38 - mul_39
        attn_score_38 = mul_39 = None
        attn_prob_38 = torch.nn.functional.softmax(attn_score_39, dim=3)
        attn_score_39 = None
        attn_prob_39 = torch.nn.functional.dropout(attn_prob_38, 0.1, False, False)
        attn_prob_38 = None
        attn_vec_19 = torch.functional.einsum(
            "bnij,jbnd->ibnd", attn_prob_39, v_head_h_19
        )
        attn_prob_39 = v_head_h_19 = None
        attn_out_57 = torch.functional.einsum(
            "ibnd,hnd->ibh",
            attn_vec_19,
            l_self_modules_layer_modules_19_modules_rel_attn_parameters_o_,
        )
        attn_vec_19 = (
            l_self_modules_layer_modules_19_modules_rel_attn_parameters_o_
        ) = None
        attn_out_58 = torch.nn.functional.dropout(attn_out_57, 0.1, False, False)
        attn_out_57 = None
        attn_out_59 = attn_out_58 + output_132
        attn_out_58 = output_132 = None
        output_133 = torch.nn.functional.layer_norm(
            attn_out_59,
            (1024,),
            l_self_modules_layer_modules_19_modules_rel_attn_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_19_modules_rel_attn_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        attn_out_59 = l_self_modules_layer_modules_19_modules_rel_attn_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_19_modules_rel_attn_modules_layer_norm_parameters_bias_ = (None)
        output_134 = torch._C._nn.linear(
            output_133,
            l_self_modules_layer_modules_19_modules_ff_modules_layer_1_parameters_weight_,
            l_self_modules_layer_modules_19_modules_ff_modules_layer_1_parameters_bias_,
        )
        l_self_modules_layer_modules_19_modules_ff_modules_layer_1_parameters_weight_ = (
            l_self_modules_layer_modules_19_modules_ff_modules_layer_1_parameters_bias_
        ) = None
        output_135 = torch._C._nn.gelu(output_134)
        output_134 = None
        output_136 = torch.nn.functional.dropout(output_135, 0.1, False, False)
        output_135 = None
        output_137 = torch._C._nn.linear(
            output_136,
            l_self_modules_layer_modules_19_modules_ff_modules_layer_2_parameters_weight_,
            l_self_modules_layer_modules_19_modules_ff_modules_layer_2_parameters_bias_,
        )
        output_136 = l_self_modules_layer_modules_19_modules_ff_modules_layer_2_parameters_weight_ = (
            l_self_modules_layer_modules_19_modules_ff_modules_layer_2_parameters_bias_
        ) = None
        output_138 = torch.nn.functional.dropout(output_137, 0.1, False, False)
        output_137 = None
        add_140 = output_138 + output_133
        output_138 = output_133 = None
        output_139 = torch.nn.functional.layer_norm(
            add_140,
            (1024,),
            l_self_modules_layer_modules_19_modules_ff_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_19_modules_ff_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_140 = l_self_modules_layer_modules_19_modules_ff_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_19_modules_ff_modules_layer_norm_parameters_bias_ = (None)
        new_mem_20 = output_139[slice(0, None, None)]
        detach_20 = new_mem_20.detach()
        new_mem_20 = None
        q_head_h_20 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_139,
            l_self_modules_layer_modules_20_modules_rel_attn_parameters_q_,
        )
        l_self_modules_layer_modules_20_modules_rel_attn_parameters_q_ = None
        k_head_h_20 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_139,
            l_self_modules_layer_modules_20_modules_rel_attn_parameters_k_,
        )
        l_self_modules_layer_modules_20_modules_rel_attn_parameters_k_ = None
        v_head_h_20 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_139,
            l_self_modules_layer_modules_20_modules_rel_attn_parameters_v_,
        )
        l_self_modules_layer_modules_20_modules_rel_attn_parameters_v_ = None
        type_21 = pos_emb_4.type(torch.float32)
        k_head_r_20 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            type_21,
            l_self_modules_layer_modules_20_modules_rel_attn_parameters_r_,
        )
        type_21 = l_self_modules_layer_modules_20_modules_rel_attn_parameters_r_ = None
        add_141 = (
            q_head_h_20
            + l_self_modules_layer_modules_20_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_20_modules_rel_attn_parameters_r_w_bias_ = None
        ac_20 = torch.functional.einsum("ibnd,jbnd->bnij", add_141, k_head_h_20)
        add_141 = k_head_h_20 = None
        add_142 = (
            q_head_h_20
            + l_self_modules_layer_modules_20_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_20_modules_rel_attn_parameters_r_r_bias_ = None
        bd_20 = torch.functional.einsum("ibnd,jbnd->bnij", add_142, k_head_r_20)
        add_142 = k_head_r_20 = None
        x_80 = bd_20.reshape(1, 16, 26, 13)
        bd_20 = None
        x_81 = x_80[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        x_80 = None
        x_82 = x_81.reshape(1, 16, 13, 25)
        x_81 = None
        arange_22 = torch.arange(
            13, device=device(type="cuda", index=0), dtype=torch.int64
        )
        x_83 = torch.index_select(x_82, 3, arange_22)
        x_82 = arange_22 = None
        add_143 = (
            q_head_h_20
            + l_self_modules_layer_modules_20_modules_rel_attn_parameters_r_s_bias_
        )
        q_head_h_20 = (
            l_self_modules_layer_modules_20_modules_rel_attn_parameters_r_s_bias_
        ) = None
        ef_40 = torch.functional.einsum(
            "ibnd,snd->ibns",
            add_143,
            l_self_modules_layer_modules_20_modules_rel_attn_parameters_seg_embed_,
        )
        add_143 = (
            l_self_modules_layer_modules_20_modules_rel_attn_parameters_seg_embed_
        ) = None
        ef_41 = torch.functional.einsum("ijbs,ibns->bnij", seg_mat_1, ef_40)
        ef_40 = None
        add_144 = ac_20 + x_83
        ac_20 = x_83 = None
        add_145 = add_144 + ef_41
        add_144 = ef_41 = None
        attn_score_40 = add_145 * 0.125
        add_145 = None
        einsum_229 = torch.functional.einsum("ijbn->bnij", non_tgt_mask_1)
        mul_41 = 1e30 * einsum_229
        einsum_229 = None
        attn_score_41 = attn_score_40 - mul_41
        attn_score_40 = mul_41 = None
        attn_prob_40 = torch.nn.functional.softmax(attn_score_41, dim=3)
        attn_score_41 = None
        attn_prob_41 = torch.nn.functional.dropout(attn_prob_40, 0.1, False, False)
        attn_prob_40 = None
        attn_vec_20 = torch.functional.einsum(
            "bnij,jbnd->ibnd", attn_prob_41, v_head_h_20
        )
        attn_prob_41 = v_head_h_20 = None
        attn_out_60 = torch.functional.einsum(
            "ibnd,hnd->ibh",
            attn_vec_20,
            l_self_modules_layer_modules_20_modules_rel_attn_parameters_o_,
        )
        attn_vec_20 = (
            l_self_modules_layer_modules_20_modules_rel_attn_parameters_o_
        ) = None
        attn_out_61 = torch.nn.functional.dropout(attn_out_60, 0.1, False, False)
        attn_out_60 = None
        attn_out_62 = attn_out_61 + output_139
        attn_out_61 = output_139 = None
        output_140 = torch.nn.functional.layer_norm(
            attn_out_62,
            (1024,),
            l_self_modules_layer_modules_20_modules_rel_attn_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_20_modules_rel_attn_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        attn_out_62 = l_self_modules_layer_modules_20_modules_rel_attn_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_20_modules_rel_attn_modules_layer_norm_parameters_bias_ = (None)
        output_141 = torch._C._nn.linear(
            output_140,
            l_self_modules_layer_modules_20_modules_ff_modules_layer_1_parameters_weight_,
            l_self_modules_layer_modules_20_modules_ff_modules_layer_1_parameters_bias_,
        )
        l_self_modules_layer_modules_20_modules_ff_modules_layer_1_parameters_weight_ = (
            l_self_modules_layer_modules_20_modules_ff_modules_layer_1_parameters_bias_
        ) = None
        output_142 = torch._C._nn.gelu(output_141)
        output_141 = None
        output_143 = torch.nn.functional.dropout(output_142, 0.1, False, False)
        output_142 = None
        output_144 = torch._C._nn.linear(
            output_143,
            l_self_modules_layer_modules_20_modules_ff_modules_layer_2_parameters_weight_,
            l_self_modules_layer_modules_20_modules_ff_modules_layer_2_parameters_bias_,
        )
        output_143 = l_self_modules_layer_modules_20_modules_ff_modules_layer_2_parameters_weight_ = (
            l_self_modules_layer_modules_20_modules_ff_modules_layer_2_parameters_bias_
        ) = None
        output_145 = torch.nn.functional.dropout(output_144, 0.1, False, False)
        output_144 = None
        add_147 = output_145 + output_140
        output_145 = output_140 = None
        output_146 = torch.nn.functional.layer_norm(
            add_147,
            (1024,),
            l_self_modules_layer_modules_20_modules_ff_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_20_modules_ff_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_147 = l_self_modules_layer_modules_20_modules_ff_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_20_modules_ff_modules_layer_norm_parameters_bias_ = (None)
        new_mem_21 = output_146[slice(0, None, None)]
        detach_21 = new_mem_21.detach()
        new_mem_21 = None
        q_head_h_21 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_146,
            l_self_modules_layer_modules_21_modules_rel_attn_parameters_q_,
        )
        l_self_modules_layer_modules_21_modules_rel_attn_parameters_q_ = None
        k_head_h_21 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_146,
            l_self_modules_layer_modules_21_modules_rel_attn_parameters_k_,
        )
        l_self_modules_layer_modules_21_modules_rel_attn_parameters_k_ = None
        v_head_h_21 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_146,
            l_self_modules_layer_modules_21_modules_rel_attn_parameters_v_,
        )
        l_self_modules_layer_modules_21_modules_rel_attn_parameters_v_ = None
        type_22 = pos_emb_4.type(torch.float32)
        k_head_r_21 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            type_22,
            l_self_modules_layer_modules_21_modules_rel_attn_parameters_r_,
        )
        type_22 = l_self_modules_layer_modules_21_modules_rel_attn_parameters_r_ = None
        add_148 = (
            q_head_h_21
            + l_self_modules_layer_modules_21_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_21_modules_rel_attn_parameters_r_w_bias_ = None
        ac_21 = torch.functional.einsum("ibnd,jbnd->bnij", add_148, k_head_h_21)
        add_148 = k_head_h_21 = None
        add_149 = (
            q_head_h_21
            + l_self_modules_layer_modules_21_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_21_modules_rel_attn_parameters_r_r_bias_ = None
        bd_21 = torch.functional.einsum("ibnd,jbnd->bnij", add_149, k_head_r_21)
        add_149 = k_head_r_21 = None
        x_84 = bd_21.reshape(1, 16, 26, 13)
        bd_21 = None
        x_85 = x_84[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        x_84 = None
        x_86 = x_85.reshape(1, 16, 13, 25)
        x_85 = None
        arange_23 = torch.arange(
            13, device=device(type="cuda", index=0), dtype=torch.int64
        )
        x_87 = torch.index_select(x_86, 3, arange_23)
        x_86 = arange_23 = None
        add_150 = (
            q_head_h_21
            + l_self_modules_layer_modules_21_modules_rel_attn_parameters_r_s_bias_
        )
        q_head_h_21 = (
            l_self_modules_layer_modules_21_modules_rel_attn_parameters_r_s_bias_
        ) = None
        ef_42 = torch.functional.einsum(
            "ibnd,snd->ibns",
            add_150,
            l_self_modules_layer_modules_21_modules_rel_attn_parameters_seg_embed_,
        )
        add_150 = (
            l_self_modules_layer_modules_21_modules_rel_attn_parameters_seg_embed_
        ) = None
        ef_43 = torch.functional.einsum("ijbs,ibns->bnij", seg_mat_1, ef_42)
        ef_42 = None
        add_151 = ac_21 + x_87
        ac_21 = x_87 = None
        add_152 = add_151 + ef_43
        add_151 = ef_43 = None
        attn_score_42 = add_152 * 0.125
        add_152 = None
        einsum_240 = torch.functional.einsum("ijbn->bnij", non_tgt_mask_1)
        mul_43 = 1e30 * einsum_240
        einsum_240 = None
        attn_score_43 = attn_score_42 - mul_43
        attn_score_42 = mul_43 = None
        attn_prob_42 = torch.nn.functional.softmax(attn_score_43, dim=3)
        attn_score_43 = None
        attn_prob_43 = torch.nn.functional.dropout(attn_prob_42, 0.1, False, False)
        attn_prob_42 = None
        attn_vec_21 = torch.functional.einsum(
            "bnij,jbnd->ibnd", attn_prob_43, v_head_h_21
        )
        attn_prob_43 = v_head_h_21 = None
        attn_out_63 = torch.functional.einsum(
            "ibnd,hnd->ibh",
            attn_vec_21,
            l_self_modules_layer_modules_21_modules_rel_attn_parameters_o_,
        )
        attn_vec_21 = (
            l_self_modules_layer_modules_21_modules_rel_attn_parameters_o_
        ) = None
        attn_out_64 = torch.nn.functional.dropout(attn_out_63, 0.1, False, False)
        attn_out_63 = None
        attn_out_65 = attn_out_64 + output_146
        attn_out_64 = output_146 = None
        output_147 = torch.nn.functional.layer_norm(
            attn_out_65,
            (1024,),
            l_self_modules_layer_modules_21_modules_rel_attn_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_21_modules_rel_attn_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        attn_out_65 = l_self_modules_layer_modules_21_modules_rel_attn_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_21_modules_rel_attn_modules_layer_norm_parameters_bias_ = (None)
        output_148 = torch._C._nn.linear(
            output_147,
            l_self_modules_layer_modules_21_modules_ff_modules_layer_1_parameters_weight_,
            l_self_modules_layer_modules_21_modules_ff_modules_layer_1_parameters_bias_,
        )
        l_self_modules_layer_modules_21_modules_ff_modules_layer_1_parameters_weight_ = (
            l_self_modules_layer_modules_21_modules_ff_modules_layer_1_parameters_bias_
        ) = None
        output_149 = torch._C._nn.gelu(output_148)
        output_148 = None
        output_150 = torch.nn.functional.dropout(output_149, 0.1, False, False)
        output_149 = None
        output_151 = torch._C._nn.linear(
            output_150,
            l_self_modules_layer_modules_21_modules_ff_modules_layer_2_parameters_weight_,
            l_self_modules_layer_modules_21_modules_ff_modules_layer_2_parameters_bias_,
        )
        output_150 = l_self_modules_layer_modules_21_modules_ff_modules_layer_2_parameters_weight_ = (
            l_self_modules_layer_modules_21_modules_ff_modules_layer_2_parameters_bias_
        ) = None
        output_152 = torch.nn.functional.dropout(output_151, 0.1, False, False)
        output_151 = None
        add_154 = output_152 + output_147
        output_152 = output_147 = None
        output_153 = torch.nn.functional.layer_norm(
            add_154,
            (1024,),
            l_self_modules_layer_modules_21_modules_ff_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_21_modules_ff_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_154 = l_self_modules_layer_modules_21_modules_ff_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_21_modules_ff_modules_layer_norm_parameters_bias_ = (None)
        new_mem_22 = output_153[slice(0, None, None)]
        detach_22 = new_mem_22.detach()
        new_mem_22 = None
        q_head_h_22 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_153,
            l_self_modules_layer_modules_22_modules_rel_attn_parameters_q_,
        )
        l_self_modules_layer_modules_22_modules_rel_attn_parameters_q_ = None
        k_head_h_22 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_153,
            l_self_modules_layer_modules_22_modules_rel_attn_parameters_k_,
        )
        l_self_modules_layer_modules_22_modules_rel_attn_parameters_k_ = None
        v_head_h_22 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_153,
            l_self_modules_layer_modules_22_modules_rel_attn_parameters_v_,
        )
        l_self_modules_layer_modules_22_modules_rel_attn_parameters_v_ = None
        type_23 = pos_emb_4.type(torch.float32)
        k_head_r_22 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            type_23,
            l_self_modules_layer_modules_22_modules_rel_attn_parameters_r_,
        )
        type_23 = l_self_modules_layer_modules_22_modules_rel_attn_parameters_r_ = None
        add_155 = (
            q_head_h_22
            + l_self_modules_layer_modules_22_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_22_modules_rel_attn_parameters_r_w_bias_ = None
        ac_22 = torch.functional.einsum("ibnd,jbnd->bnij", add_155, k_head_h_22)
        add_155 = k_head_h_22 = None
        add_156 = (
            q_head_h_22
            + l_self_modules_layer_modules_22_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_22_modules_rel_attn_parameters_r_r_bias_ = None
        bd_22 = torch.functional.einsum("ibnd,jbnd->bnij", add_156, k_head_r_22)
        add_156 = k_head_r_22 = None
        x_88 = bd_22.reshape(1, 16, 26, 13)
        bd_22 = None
        x_89 = x_88[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        x_88 = None
        x_90 = x_89.reshape(1, 16, 13, 25)
        x_89 = None
        arange_24 = torch.arange(
            13, device=device(type="cuda", index=0), dtype=torch.int64
        )
        x_91 = torch.index_select(x_90, 3, arange_24)
        x_90 = arange_24 = None
        add_157 = (
            q_head_h_22
            + l_self_modules_layer_modules_22_modules_rel_attn_parameters_r_s_bias_
        )
        q_head_h_22 = (
            l_self_modules_layer_modules_22_modules_rel_attn_parameters_r_s_bias_
        ) = None
        ef_44 = torch.functional.einsum(
            "ibnd,snd->ibns",
            add_157,
            l_self_modules_layer_modules_22_modules_rel_attn_parameters_seg_embed_,
        )
        add_157 = (
            l_self_modules_layer_modules_22_modules_rel_attn_parameters_seg_embed_
        ) = None
        ef_45 = torch.functional.einsum("ijbs,ibns->bnij", seg_mat_1, ef_44)
        ef_44 = None
        add_158 = ac_22 + x_91
        ac_22 = x_91 = None
        add_159 = add_158 + ef_45
        add_158 = ef_45 = None
        attn_score_44 = add_159 * 0.125
        add_159 = None
        einsum_251 = torch.functional.einsum("ijbn->bnij", non_tgt_mask_1)
        mul_45 = 1e30 * einsum_251
        einsum_251 = None
        attn_score_45 = attn_score_44 - mul_45
        attn_score_44 = mul_45 = None
        attn_prob_44 = torch.nn.functional.softmax(attn_score_45, dim=3)
        attn_score_45 = None
        attn_prob_45 = torch.nn.functional.dropout(attn_prob_44, 0.1, False, False)
        attn_prob_44 = None
        attn_vec_22 = torch.functional.einsum(
            "bnij,jbnd->ibnd", attn_prob_45, v_head_h_22
        )
        attn_prob_45 = v_head_h_22 = None
        attn_out_66 = torch.functional.einsum(
            "ibnd,hnd->ibh",
            attn_vec_22,
            l_self_modules_layer_modules_22_modules_rel_attn_parameters_o_,
        )
        attn_vec_22 = (
            l_self_modules_layer_modules_22_modules_rel_attn_parameters_o_
        ) = None
        attn_out_67 = torch.nn.functional.dropout(attn_out_66, 0.1, False, False)
        attn_out_66 = None
        attn_out_68 = attn_out_67 + output_153
        attn_out_67 = output_153 = None
        output_154 = torch.nn.functional.layer_norm(
            attn_out_68,
            (1024,),
            l_self_modules_layer_modules_22_modules_rel_attn_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_22_modules_rel_attn_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        attn_out_68 = l_self_modules_layer_modules_22_modules_rel_attn_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_22_modules_rel_attn_modules_layer_norm_parameters_bias_ = (None)
        output_155 = torch._C._nn.linear(
            output_154,
            l_self_modules_layer_modules_22_modules_ff_modules_layer_1_parameters_weight_,
            l_self_modules_layer_modules_22_modules_ff_modules_layer_1_parameters_bias_,
        )
        l_self_modules_layer_modules_22_modules_ff_modules_layer_1_parameters_weight_ = (
            l_self_modules_layer_modules_22_modules_ff_modules_layer_1_parameters_bias_
        ) = None
        output_156 = torch._C._nn.gelu(output_155)
        output_155 = None
        output_157 = torch.nn.functional.dropout(output_156, 0.1, False, False)
        output_156 = None
        output_158 = torch._C._nn.linear(
            output_157,
            l_self_modules_layer_modules_22_modules_ff_modules_layer_2_parameters_weight_,
            l_self_modules_layer_modules_22_modules_ff_modules_layer_2_parameters_bias_,
        )
        output_157 = l_self_modules_layer_modules_22_modules_ff_modules_layer_2_parameters_weight_ = (
            l_self_modules_layer_modules_22_modules_ff_modules_layer_2_parameters_bias_
        ) = None
        output_159 = torch.nn.functional.dropout(output_158, 0.1, False, False)
        output_158 = None
        add_161 = output_159 + output_154
        output_159 = output_154 = None
        output_160 = torch.nn.functional.layer_norm(
            add_161,
            (1024,),
            l_self_modules_layer_modules_22_modules_ff_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_22_modules_ff_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_161 = l_self_modules_layer_modules_22_modules_ff_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_22_modules_ff_modules_layer_norm_parameters_bias_ = (None)
        new_mem_23 = output_160[slice(0, None, None)]
        detach_23 = new_mem_23.detach()
        new_mem_23 = None
        q_head_h_23 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_160,
            l_self_modules_layer_modules_23_modules_rel_attn_parameters_q_,
        )
        l_self_modules_layer_modules_23_modules_rel_attn_parameters_q_ = None
        k_head_h_23 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_160,
            l_self_modules_layer_modules_23_modules_rel_attn_parameters_k_,
        )
        l_self_modules_layer_modules_23_modules_rel_attn_parameters_k_ = None
        v_head_h_23 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            output_160,
            l_self_modules_layer_modules_23_modules_rel_attn_parameters_v_,
        )
        l_self_modules_layer_modules_23_modules_rel_attn_parameters_v_ = None
        type_24 = pos_emb_4.type(torch.float32)
        pos_emb_4 = None
        k_head_r_23 = torch.functional.einsum(
            "ibh,hnd->ibnd",
            type_24,
            l_self_modules_layer_modules_23_modules_rel_attn_parameters_r_,
        )
        type_24 = l_self_modules_layer_modules_23_modules_rel_attn_parameters_r_ = None
        add_162 = (
            q_head_h_23
            + l_self_modules_layer_modules_23_modules_rel_attn_parameters_r_w_bias_
        )
        l_self_modules_layer_modules_23_modules_rel_attn_parameters_r_w_bias_ = None
        ac_23 = torch.functional.einsum("ibnd,jbnd->bnij", add_162, k_head_h_23)
        add_162 = k_head_h_23 = None
        add_163 = (
            q_head_h_23
            + l_self_modules_layer_modules_23_modules_rel_attn_parameters_r_r_bias_
        )
        l_self_modules_layer_modules_23_modules_rel_attn_parameters_r_r_bias_ = None
        bd_23 = torch.functional.einsum("ibnd,jbnd->bnij", add_163, k_head_r_23)
        add_163 = k_head_r_23 = None
        x_92 = bd_23.reshape(1, 16, 26, 13)
        bd_23 = None
        x_93 = x_92[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        x_92 = None
        x_94 = x_93.reshape(1, 16, 13, 25)
        x_93 = None
        arange_25 = torch.arange(
            13, device=device(type="cuda", index=0), dtype=torch.int64
        )
        x_95 = torch.index_select(x_94, 3, arange_25)
        x_94 = arange_25 = None
        add_164 = (
            q_head_h_23
            + l_self_modules_layer_modules_23_modules_rel_attn_parameters_r_s_bias_
        )
        q_head_h_23 = (
            l_self_modules_layer_modules_23_modules_rel_attn_parameters_r_s_bias_
        ) = None
        ef_46 = torch.functional.einsum(
            "ibnd,snd->ibns",
            add_164,
            l_self_modules_layer_modules_23_modules_rel_attn_parameters_seg_embed_,
        )
        add_164 = (
            l_self_modules_layer_modules_23_modules_rel_attn_parameters_seg_embed_
        ) = None
        ef_47 = torch.functional.einsum("ijbs,ibns->bnij", seg_mat_1, ef_46)
        seg_mat_1 = ef_46 = None
        add_165 = ac_23 + x_95
        ac_23 = x_95 = None
        add_166 = add_165 + ef_47
        add_165 = ef_47 = None
        attn_score_46 = add_166 * 0.125
        add_166 = None
        einsum_262 = torch.functional.einsum("ijbn->bnij", non_tgt_mask_1)
        non_tgt_mask_1 = None
        mul_47 = 1e30 * einsum_262
        einsum_262 = None
        attn_score_47 = attn_score_46 - mul_47
        attn_score_46 = mul_47 = None
        attn_prob_46 = torch.nn.functional.softmax(attn_score_47, dim=3)
        attn_score_47 = None
        attn_prob_47 = torch.nn.functional.dropout(attn_prob_46, 0.1, False, False)
        attn_prob_46 = None
        attn_vec_23 = torch.functional.einsum(
            "bnij,jbnd->ibnd", attn_prob_47, v_head_h_23
        )
        attn_prob_47 = v_head_h_23 = None
        attn_out_69 = torch.functional.einsum(
            "ibnd,hnd->ibh",
            attn_vec_23,
            l_self_modules_layer_modules_23_modules_rel_attn_parameters_o_,
        )
        attn_vec_23 = (
            l_self_modules_layer_modules_23_modules_rel_attn_parameters_o_
        ) = None
        attn_out_70 = torch.nn.functional.dropout(attn_out_69, 0.1, False, False)
        attn_out_69 = None
        attn_out_71 = attn_out_70 + output_160
        attn_out_70 = output_160 = None
        output_161 = torch.nn.functional.layer_norm(
            attn_out_71,
            (1024,),
            l_self_modules_layer_modules_23_modules_rel_attn_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_23_modules_rel_attn_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        attn_out_71 = l_self_modules_layer_modules_23_modules_rel_attn_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_23_modules_rel_attn_modules_layer_norm_parameters_bias_ = (None)
        output_162 = torch._C._nn.linear(
            output_161,
            l_self_modules_layer_modules_23_modules_ff_modules_layer_1_parameters_weight_,
            l_self_modules_layer_modules_23_modules_ff_modules_layer_1_parameters_bias_,
        )
        l_self_modules_layer_modules_23_modules_ff_modules_layer_1_parameters_weight_ = (
            l_self_modules_layer_modules_23_modules_ff_modules_layer_1_parameters_bias_
        ) = None
        output_163 = torch._C._nn.gelu(output_162)
        output_162 = None
        output_164 = torch.nn.functional.dropout(output_163, 0.1, False, False)
        output_163 = None
        output_165 = torch._C._nn.linear(
            output_164,
            l_self_modules_layer_modules_23_modules_ff_modules_layer_2_parameters_weight_,
            l_self_modules_layer_modules_23_modules_ff_modules_layer_2_parameters_bias_,
        )
        output_164 = l_self_modules_layer_modules_23_modules_ff_modules_layer_2_parameters_weight_ = (
            l_self_modules_layer_modules_23_modules_ff_modules_layer_2_parameters_bias_
        ) = None
        output_166 = torch.nn.functional.dropout(output_165, 0.1, False, False)
        output_165 = None
        add_168 = output_166 + output_161
        output_166 = output_161 = None
        output_167 = torch.nn.functional.layer_norm(
            add_168,
            (1024,),
            l_self_modules_layer_modules_23_modules_ff_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_modules_23_modules_ff_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_168 = l_self_modules_layer_modules_23_modules_ff_modules_layer_norm_parameters_weight_ = l_self_modules_layer_modules_23_modules_ff_modules_layer_norm_parameters_bias_ = (None)
        output_168 = torch.nn.functional.dropout(output_167, 0.1, False, False)
        output_167 = None
        permute = output_168.permute(1, 0, 2)
        output_168 = None
        output_169 = permute.contiguous()
        permute = None
        return (
            output_169,
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
            detach_12,
            detach_13,
            detach_14,
            detach_15,
            detach_16,
            detach_17,
            detach_18,
            detach_19,
            detach_20,
            detach_21,
            detach_22,
            detach_23,
        )
