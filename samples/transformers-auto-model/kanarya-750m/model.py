import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_self_modules_transformer_modules_wte_parameters_weight_: torch.nn.parameter.Parameter,
        L_attention_mask_: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_attn_embed_positions: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_0_modules_attn_scale_attn: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_fc_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_fc_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_fc_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_fc_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_attn_embed_positions: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_1_modules_attn_scale_attn: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_fc_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_fc_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_fc_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_fc_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_attn_embed_positions: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_2_modules_attn_scale_attn: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_fc_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_fc_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_fc_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_fc_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_attn_embed_positions: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_3_modules_attn_scale_attn: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_fc_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_fc_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_fc_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_fc_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_attn_embed_positions: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_4_modules_attn_scale_attn: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_fc_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_fc_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_fc_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_fc_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_attn_embed_positions: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_5_modules_attn_scale_attn: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_fc_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_fc_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_fc_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_fc_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_attn_embed_positions: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_6_modules_attn_scale_attn: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_fc_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_fc_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_fc_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_fc_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_attn_embed_positions: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_7_modules_attn_scale_attn: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_fc_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_fc_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_fc_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_fc_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_attn_embed_positions: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_8_modules_attn_scale_attn: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_fc_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_fc_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_fc_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_fc_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_attn_embed_positions: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_9_modules_attn_scale_attn: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_fc_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_fc_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_fc_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_fc_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_attn_embed_positions: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_10_modules_attn_scale_attn: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_fc_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_fc_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_fc_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_fc_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_attn_embed_positions: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_11_modules_attn_scale_attn: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_fc_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_fc_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_fc_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_fc_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_ln_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_ln_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_lm_head_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_lm_head_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_input_ids_ = L_input_ids_
        l_self_modules_transformer_modules_wte_parameters_weight_ = (
            L_self_modules_transformer_modules_wte_parameters_weight_
        )
        l_attention_mask_ = L_attention_mask_
        l_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_attn_embed_positions = (
            L_self_modules_transformer_modules_h_modules_0_modules_attn_embed_positions
        )
        l_self_modules_transformer_modules_h_modules_0_modules_attn_scale_attn = (
            L_self_modules_transformer_modules_h_modules_0_modules_attn_scale_attn
        )
        l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_fc_in_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_fc_in_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_fc_in_parameters_bias_ = L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_fc_in_parameters_bias_
        l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_fc_out_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_fc_out_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_fc_out_parameters_bias_ = L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_fc_out_parameters_bias_
        l_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_attn_embed_positions = (
            L_self_modules_transformer_modules_h_modules_1_modules_attn_embed_positions
        )
        l_self_modules_transformer_modules_h_modules_1_modules_attn_scale_attn = (
            L_self_modules_transformer_modules_h_modules_1_modules_attn_scale_attn
        )
        l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_fc_in_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_fc_in_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_fc_in_parameters_bias_ = L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_fc_in_parameters_bias_
        l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_fc_out_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_fc_out_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_fc_out_parameters_bias_ = L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_fc_out_parameters_bias_
        l_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_2_modules_attn_embed_positions = (
            L_self_modules_transformer_modules_h_modules_2_modules_attn_embed_positions
        )
        l_self_modules_transformer_modules_h_modules_2_modules_attn_scale_attn = (
            L_self_modules_transformer_modules_h_modules_2_modules_attn_scale_attn
        )
        l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_fc_in_parameters_weight_ = L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_fc_in_parameters_weight_
        l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_fc_in_parameters_bias_ = L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_fc_in_parameters_bias_
        l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_fc_out_parameters_weight_ = L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_fc_out_parameters_weight_
        l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_fc_out_parameters_bias_ = L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_fc_out_parameters_bias_
        l_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_3_modules_attn_embed_positions = (
            L_self_modules_transformer_modules_h_modules_3_modules_attn_embed_positions
        )
        l_self_modules_transformer_modules_h_modules_3_modules_attn_scale_attn = (
            L_self_modules_transformer_modules_h_modules_3_modules_attn_scale_attn
        )
        l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_fc_in_parameters_weight_ = L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_fc_in_parameters_weight_
        l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_fc_in_parameters_bias_ = L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_fc_in_parameters_bias_
        l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_fc_out_parameters_weight_ = L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_fc_out_parameters_weight_
        l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_fc_out_parameters_bias_ = L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_fc_out_parameters_bias_
        l_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_4_modules_attn_embed_positions = (
            L_self_modules_transformer_modules_h_modules_4_modules_attn_embed_positions
        )
        l_self_modules_transformer_modules_h_modules_4_modules_attn_scale_attn = (
            L_self_modules_transformer_modules_h_modules_4_modules_attn_scale_attn
        )
        l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_fc_in_parameters_weight_ = L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_fc_in_parameters_weight_
        l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_fc_in_parameters_bias_ = L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_fc_in_parameters_bias_
        l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_fc_out_parameters_weight_ = L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_fc_out_parameters_weight_
        l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_fc_out_parameters_bias_ = L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_fc_out_parameters_bias_
        l_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_5_modules_attn_embed_positions = (
            L_self_modules_transformer_modules_h_modules_5_modules_attn_embed_positions
        )
        l_self_modules_transformer_modules_h_modules_5_modules_attn_scale_attn = (
            L_self_modules_transformer_modules_h_modules_5_modules_attn_scale_attn
        )
        l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_fc_in_parameters_weight_ = L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_fc_in_parameters_weight_
        l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_fc_in_parameters_bias_ = L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_fc_in_parameters_bias_
        l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_fc_out_parameters_weight_ = L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_fc_out_parameters_weight_
        l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_fc_out_parameters_bias_ = L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_fc_out_parameters_bias_
        l_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_6_modules_attn_embed_positions = (
            L_self_modules_transformer_modules_h_modules_6_modules_attn_embed_positions
        )
        l_self_modules_transformer_modules_h_modules_6_modules_attn_scale_attn = (
            L_self_modules_transformer_modules_h_modules_6_modules_attn_scale_attn
        )
        l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_fc_in_parameters_weight_ = L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_fc_in_parameters_weight_
        l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_fc_in_parameters_bias_ = L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_fc_in_parameters_bias_
        l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_fc_out_parameters_weight_ = L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_fc_out_parameters_weight_
        l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_fc_out_parameters_bias_ = L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_fc_out_parameters_bias_
        l_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_7_modules_attn_embed_positions = (
            L_self_modules_transformer_modules_h_modules_7_modules_attn_embed_positions
        )
        l_self_modules_transformer_modules_h_modules_7_modules_attn_scale_attn = (
            L_self_modules_transformer_modules_h_modules_7_modules_attn_scale_attn
        )
        l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_fc_in_parameters_weight_ = L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_fc_in_parameters_weight_
        l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_fc_in_parameters_bias_ = L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_fc_in_parameters_bias_
        l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_fc_out_parameters_weight_ = L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_fc_out_parameters_weight_
        l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_fc_out_parameters_bias_ = L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_fc_out_parameters_bias_
        l_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_8_modules_attn_embed_positions = (
            L_self_modules_transformer_modules_h_modules_8_modules_attn_embed_positions
        )
        l_self_modules_transformer_modules_h_modules_8_modules_attn_scale_attn = (
            L_self_modules_transformer_modules_h_modules_8_modules_attn_scale_attn
        )
        l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_fc_in_parameters_weight_ = L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_fc_in_parameters_weight_
        l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_fc_in_parameters_bias_ = L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_fc_in_parameters_bias_
        l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_fc_out_parameters_weight_ = L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_fc_out_parameters_weight_
        l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_fc_out_parameters_bias_ = L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_fc_out_parameters_bias_
        l_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_9_modules_attn_embed_positions = (
            L_self_modules_transformer_modules_h_modules_9_modules_attn_embed_positions
        )
        l_self_modules_transformer_modules_h_modules_9_modules_attn_scale_attn = (
            L_self_modules_transformer_modules_h_modules_9_modules_attn_scale_attn
        )
        l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_fc_in_parameters_weight_ = L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_fc_in_parameters_weight_
        l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_fc_in_parameters_bias_ = L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_fc_in_parameters_bias_
        l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_fc_out_parameters_weight_ = L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_fc_out_parameters_weight_
        l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_fc_out_parameters_bias_ = L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_fc_out_parameters_bias_
        l_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_bias_ = L_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_bias_
        l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_10_modules_attn_embed_positions = (
            L_self_modules_transformer_modules_h_modules_10_modules_attn_embed_positions
        )
        l_self_modules_transformer_modules_h_modules_10_modules_attn_scale_attn = (
            L_self_modules_transformer_modules_h_modules_10_modules_attn_scale_attn
        )
        l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_fc_in_parameters_weight_ = L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_fc_in_parameters_weight_
        l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_fc_in_parameters_bias_ = L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_fc_in_parameters_bias_
        l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_fc_out_parameters_weight_ = L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_fc_out_parameters_weight_
        l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_fc_out_parameters_bias_ = L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_fc_out_parameters_bias_
        l_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_bias_ = L_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_bias_
        l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_11_modules_attn_embed_positions = (
            L_self_modules_transformer_modules_h_modules_11_modules_attn_embed_positions
        )
        l_self_modules_transformer_modules_h_modules_11_modules_attn_scale_attn = (
            L_self_modules_transformer_modules_h_modules_11_modules_attn_scale_attn
        )
        l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_fc_in_parameters_weight_ = L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_fc_in_parameters_weight_
        l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_fc_in_parameters_bias_ = L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_fc_in_parameters_bias_
        l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_fc_out_parameters_weight_ = L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_fc_out_parameters_weight_
        l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_fc_out_parameters_bias_ = L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_fc_out_parameters_bias_
        l_self_modules_transformer_modules_ln_f_parameters_weight_ = (
            L_self_modules_transformer_modules_ln_f_parameters_weight_
        )
        l_self_modules_transformer_modules_ln_f_parameters_bias_ = (
            L_self_modules_transformer_modules_ln_f_parameters_bias_
        )
        l_self_modules_lm_head_parameters_weight_ = (
            L_self_modules_lm_head_parameters_weight_
        )
        l_self_modules_lm_head_parameters_bias_ = (
            L_self_modules_lm_head_parameters_bias_
        )
        inputs_embeds = torch.nn.functional.embedding(
            l_input_ids_,
            l_self_modules_transformer_modules_wte_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        l_input_ids_ = l_self_modules_transformer_modules_wte_parameters_weight_ = None
        cache_position = torch.arange(0, 36, device=device(type="cpu"))
        position_ids = cache_position.unsqueeze(0)
        causal_mask = torch.full(
            (36, 36),
            fill_value=-65504.0,
            dtype=torch.float16,
            device=device(type="cpu"),
        )
        causal_mask_1 = torch.triu(causal_mask, diagonal=1)
        causal_mask = None
        arange_1 = torch.arange(36, device=device(type="cpu"))
        reshape = cache_position.reshape(-1, 1)
        cache_position = None
        gt = arange_1 > reshape
        arange_1 = reshape = None
        causal_mask_1 *= gt
        causal_mask_2 = causal_mask_1
        causal_mask_1 = gt = None
        getitem = causal_mask_2[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        causal_mask_2 = None
        causal_mask_3 = getitem.expand(1, 1, -1, -1)
        getitem = None
        causal_mask_4 = causal_mask_3.clone()
        causal_mask_3 = None
        getitem_1 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 36, None),
            )
        ]
        getitem_2 = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        l_attention_mask_ = None
        to = getitem_2.to(device(type="cpu"))
        getitem_2 = None
        padding_mask = getitem_1 + to
        getitem_1 = to = None
        padding_mask_1 = padding_mask == 0
        padding_mask = None
        getitem_3 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 36, None),
            )
        ]
        masked_fill = getitem_3.masked_fill(padding_mask_1, -65504.0)
        getitem_3 = padding_mask_1 = None
        causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 36, None),
            )
        ] = masked_fill
        setitem = causal_mask_4
        masked_fill = setitem = None
        hidden_states = torch.nn.functional.dropout(inputs_embeds, 0.0, False, False)
        inputs_embeds = None
        hidden_states_1 = torch.nn.functional.layer_norm(
            hidden_states,
            (2048,),
            l_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_bias_
        ) = None
        query = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        tensor = query.view((1, 36, 16, 128))
        query = None
        tensor_1 = key.view((1, 36, 16, 128))
        key = None
        tensor_2 = value.view((1, 36, 16, 128))
        value = None
        value_1 = tensor_2.permute(0, 2, 1, 3)
        tensor_2 = None
        embed_positions = l_self_modules_transformer_modules_h_modules_0_modules_attn_embed_positions.repeat(
            1, 1, 1
        )
        l_self_modules_transformer_modules_h_modules_0_modules_attn_embed_positions = (
            None
        )
        unsqueeze_1 = position_ids.unsqueeze(-1)
        repeated_position_ids = unsqueeze_1.repeat(1, 1, 64)
        unsqueeze_1 = None
        sincos = torch.gather(embed_positions, 1, repeated_position_ids)
        embed_positions = repeated_position_ids = None
        split = torch.functional.split(sincos, 32, dim=-1)
        sincos = None
        sin = split[0]
        cos = split[1]
        split = None
        k_rot = tensor_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 64, None),
            )
        ]
        k_pass = tensor_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(64, None, None),
            )
        ]
        tensor_1 = None
        q_rot = tensor[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 64, None),
            )
        ]
        q_pass = tensor[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(64, None, None),
            )
        ]
        tensor = None
        getitem_10 = sin[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_1 = torch.repeat_interleave(getitem_10, 2, 3)
        getitem_10 = None
        getitem_11 = cos[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_1 = torch.repeat_interleave(getitem_11, 2, 3)
        getitem_11 = None
        mul = k_rot * cos_1
        cos_1 = None
        x1 = k_rot[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2 = k_rot[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        k_rot = None
        neg = -x2
        x2 = None
        x = torch.stack((neg, x1), dim=-1)
        neg = x1 = None
        flatten = x.flatten(-2)
        x = None
        mul_1 = flatten * sin_1
        flatten = sin_1 = None
        k_rot_1 = mul + mul_1
        mul = mul_1 = None
        getitem_14 = sin[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin = None
        sin_2 = torch.repeat_interleave(getitem_14, 2, 3)
        getitem_14 = None
        getitem_15 = cos[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos = None
        cos_2 = torch.repeat_interleave(getitem_15, 2, 3)
        getitem_15 = None
        mul_2 = q_rot * cos_2
        cos_2 = None
        x1_1 = q_rot[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_1 = q_rot[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        q_rot = None
        neg_1 = -x2_1
        x2_1 = None
        x_1 = torch.stack((neg_1, x1_1), dim=-1)
        neg_1 = x1_1 = None
        flatten_1 = x_1.flatten(-2)
        x_1 = None
        mul_3 = flatten_1 * sin_2
        flatten_1 = sin_2 = None
        q_rot_1 = mul_2 + mul_3
        mul_2 = mul_3 = None
        key_1 = torch.cat([k_rot_1, k_pass], dim=-1)
        k_rot_1 = k_pass = None
        query_1 = torch.cat([q_rot_1, q_pass], dim=-1)
        q_rot_1 = q_pass = None
        key_2 = key_1.permute(0, 2, 1, 3)
        key_1 = None
        query_2 = query_1.permute(0, 2, 1, 3)
        query_1 = None
        query_3 = query_2.to(torch.float32)
        query_2 = None
        key_3 = key_2.to(torch.float32)
        key_2 = None
        transpose = key_3.transpose(-1, -2)
        key_3 = None
        attn_weights = torch.matmul(query_3, transpose)
        query_3 = transpose = None
        attn_weights_1 = (
            attn_weights
            / l_self_modules_transformer_modules_h_modules_0_modules_attn_scale_attn
        )
        attn_weights = (
            l_self_modules_transformer_modules_h_modules_0_modules_attn_scale_attn
        ) = None
        causal_mask_5 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 36, None),
            )
        ]
        attn_weights_2 = attn_weights_1 + causal_mask_5
        attn_weights_1 = causal_mask_5 = None
        attn_weights_3 = torch.nn.functional.softmax(attn_weights_2, dim=-1)
        attn_weights_2 = None
        attn_weights_4 = attn_weights_3.to(torch.float16)
        attn_weights_3 = None
        attn_weights_5 = torch.nn.functional.dropout(attn_weights_4, 0.0, False, False)
        attn_weights_4 = None
        attn_output = torch.matmul(attn_weights_5, value_1)
        attn_weights_5 = value_1 = None
        permute_3 = attn_output.permute(0, 2, 1, 3)
        attn_output = None
        tensor_3 = permute_3.contiguous()
        permute_3 = None
        attn_output_1 = tensor_3.view((1, 36, 2048))
        tensor_3 = None
        attn_output_2 = torch._C._nn.linear(
            attn_output_1,
            l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_out_proj_parameters_weight_,
            None,
        )
        attn_output_1 = l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_out_proj_parameters_weight_ = (None)
        attn_output_3 = torch.nn.functional.dropout(attn_output_2, 0.0, False, False)
        attn_output_2 = None
        hidden_states_2 = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_fc_in_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_fc_in_parameters_bias_,
        )
        hidden_states_1 = l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_fc_in_parameters_weight_ = l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_fc_in_parameters_bias_ = (None)
        mul_4 = 0.5 * hidden_states_2
        pow_1 = torch.pow(hidden_states_2, 3.0)
        mul_5 = 0.044715 * pow_1
        pow_1 = None
        add_4 = hidden_states_2 + mul_5
        hidden_states_2 = mul_5 = None
        mul_6 = 0.7978845608028654 * add_4
        add_4 = None
        tanh = torch.tanh(mul_6)
        mul_6 = None
        add_5 = 1.0 + tanh
        tanh = None
        hidden_states_3 = mul_4 * add_5
        mul_4 = add_5 = None
        hidden_states_4 = torch._C._nn.linear(
            hidden_states_3,
            l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_fc_out_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_fc_out_parameters_bias_,
        )
        hidden_states_3 = l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_fc_out_parameters_weight_ = l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_fc_out_parameters_bias_ = (None)
        hidden_states_5 = torch.nn.functional.dropout(
            hidden_states_4, 0.0, False, False
        )
        hidden_states_4 = None
        add_6 = attn_output_3 + hidden_states_5
        attn_output_3 = hidden_states_5 = None
        hidden_states_6 = add_6 + hidden_states
        add_6 = hidden_states = None
        hidden_states_7 = torch.nn.functional.layer_norm(
            hidden_states_6,
            (2048,),
            l_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_bias_
        ) = None
        query_4 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_4 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_2 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        tensor_4 = query_4.view((1, 36, 16, 128))
        query_4 = None
        tensor_5 = key_4.view((1, 36, 16, 128))
        key_4 = None
        tensor_6 = value_2.view((1, 36, 16, 128))
        value_2 = None
        value_3 = tensor_6.permute(0, 2, 1, 3)
        tensor_6 = None
        embed_positions_1 = l_self_modules_transformer_modules_h_modules_1_modules_attn_embed_positions.repeat(
            1, 1, 1
        )
        l_self_modules_transformer_modules_h_modules_1_modules_attn_embed_positions = (
            None
        )
        unsqueeze_2 = position_ids.unsqueeze(-1)
        repeated_position_ids_1 = unsqueeze_2.repeat(1, 1, 64)
        unsqueeze_2 = None
        sincos_1 = torch.gather(embed_positions_1, 1, repeated_position_ids_1)
        embed_positions_1 = repeated_position_ids_1 = None
        split_1 = torch.functional.split(sincos_1, 32, dim=-1)
        sincos_1 = None
        sin_3 = split_1[0]
        cos_3 = split_1[1]
        split_1 = None
        k_rot_2 = tensor_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 64, None),
            )
        ]
        k_pass_1 = tensor_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(64, None, None),
            )
        ]
        tensor_5 = None
        q_rot_2 = tensor_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 64, None),
            )
        ]
        q_pass_1 = tensor_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(64, None, None),
            )
        ]
        tensor_4 = None
        getitem_25 = sin_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_4 = torch.repeat_interleave(getitem_25, 2, 3)
        getitem_25 = None
        getitem_26 = cos_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_4 = torch.repeat_interleave(getitem_26, 2, 3)
        getitem_26 = None
        mul_8 = k_rot_2 * cos_4
        cos_4 = None
        x1_2 = k_rot_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_2 = k_rot_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        k_rot_2 = None
        neg_2 = -x2_2
        x2_2 = None
        x_2 = torch.stack((neg_2, x1_2), dim=-1)
        neg_2 = x1_2 = None
        flatten_2 = x_2.flatten(-2)
        x_2 = None
        mul_9 = flatten_2 * sin_4
        flatten_2 = sin_4 = None
        k_rot_3 = mul_8 + mul_9
        mul_8 = mul_9 = None
        getitem_29 = sin_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_3 = None
        sin_5 = torch.repeat_interleave(getitem_29, 2, 3)
        getitem_29 = None
        getitem_30 = cos_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_3 = None
        cos_5 = torch.repeat_interleave(getitem_30, 2, 3)
        getitem_30 = None
        mul_10 = q_rot_2 * cos_5
        cos_5 = None
        x1_3 = q_rot_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_3 = q_rot_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        q_rot_2 = None
        neg_3 = -x2_3
        x2_3 = None
        x_3 = torch.stack((neg_3, x1_3), dim=-1)
        neg_3 = x1_3 = None
        flatten_3 = x_3.flatten(-2)
        x_3 = None
        mul_11 = flatten_3 * sin_5
        flatten_3 = sin_5 = None
        q_rot_3 = mul_10 + mul_11
        mul_10 = mul_11 = None
        key_5 = torch.cat([k_rot_3, k_pass_1], dim=-1)
        k_rot_3 = k_pass_1 = None
        query_5 = torch.cat([q_rot_3, q_pass_1], dim=-1)
        q_rot_3 = q_pass_1 = None
        key_6 = key_5.permute(0, 2, 1, 3)
        key_5 = None
        query_6 = query_5.permute(0, 2, 1, 3)
        query_5 = None
        query_7 = query_6.to(torch.float32)
        query_6 = None
        key_7 = key_6.to(torch.float32)
        key_6 = None
        transpose_1 = key_7.transpose(-1, -2)
        key_7 = None
        attn_weights_6 = torch.matmul(query_7, transpose_1)
        query_7 = transpose_1 = None
        attn_weights_7 = (
            attn_weights_6
            / l_self_modules_transformer_modules_h_modules_1_modules_attn_scale_attn
        )
        attn_weights_6 = (
            l_self_modules_transformer_modules_h_modules_1_modules_attn_scale_attn
        ) = None
        causal_mask_6 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 36, None),
            )
        ]
        attn_weights_8 = attn_weights_7 + causal_mask_6
        attn_weights_7 = causal_mask_6 = None
        attn_weights_9 = torch.nn.functional.softmax(attn_weights_8, dim=-1)
        attn_weights_8 = None
        attn_weights_10 = attn_weights_9.to(torch.float16)
        attn_weights_9 = None
        attn_weights_11 = torch.nn.functional.dropout(
            attn_weights_10, 0.0, False, False
        )
        attn_weights_10 = None
        attn_output_4 = torch.matmul(attn_weights_11, value_3)
        attn_weights_11 = value_3 = None
        permute_7 = attn_output_4.permute(0, 2, 1, 3)
        attn_output_4 = None
        tensor_7 = permute_7.contiguous()
        permute_7 = None
        attn_output_5 = tensor_7.view((1, 36, 2048))
        tensor_7 = None
        attn_output_6 = torch._C._nn.linear(
            attn_output_5,
            l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_out_proj_parameters_weight_,
            None,
        )
        attn_output_5 = l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_out_proj_parameters_weight_ = (None)
        attn_output_7 = torch.nn.functional.dropout(attn_output_6, 0.0, False, False)
        attn_output_6 = None
        hidden_states_8 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_fc_in_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_fc_in_parameters_bias_,
        )
        hidden_states_7 = l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_fc_in_parameters_weight_ = l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_fc_in_parameters_bias_ = (None)
        mul_12 = 0.5 * hidden_states_8
        pow_2 = torch.pow(hidden_states_8, 3.0)
        mul_13 = 0.044715 * pow_2
        pow_2 = None
        add_11 = hidden_states_8 + mul_13
        hidden_states_8 = mul_13 = None
        mul_14 = 0.7978845608028654 * add_11
        add_11 = None
        tanh_1 = torch.tanh(mul_14)
        mul_14 = None
        add_12 = 1.0 + tanh_1
        tanh_1 = None
        hidden_states_9 = mul_12 * add_12
        mul_12 = add_12 = None
        hidden_states_10 = torch._C._nn.linear(
            hidden_states_9,
            l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_fc_out_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_fc_out_parameters_bias_,
        )
        hidden_states_9 = l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_fc_out_parameters_weight_ = l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_fc_out_parameters_bias_ = (None)
        hidden_states_11 = torch.nn.functional.dropout(
            hidden_states_10, 0.0, False, False
        )
        hidden_states_10 = None
        add_13 = attn_output_7 + hidden_states_11
        attn_output_7 = hidden_states_11 = None
        hidden_states_12 = add_13 + hidden_states_6
        add_13 = hidden_states_6 = None
        hidden_states_13 = torch.nn.functional.layer_norm(
            hidden_states_12,
            (2048,),
            l_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_bias_
        ) = None
        query_8 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_8 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_4 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        tensor_8 = query_8.view((1, 36, 16, 128))
        query_8 = None
        tensor_9 = key_8.view((1, 36, 16, 128))
        key_8 = None
        tensor_10 = value_4.view((1, 36, 16, 128))
        value_4 = None
        value_5 = tensor_10.permute(0, 2, 1, 3)
        tensor_10 = None
        embed_positions_2 = l_self_modules_transformer_modules_h_modules_2_modules_attn_embed_positions.repeat(
            1, 1, 1
        )
        l_self_modules_transformer_modules_h_modules_2_modules_attn_embed_positions = (
            None
        )
        unsqueeze_3 = position_ids.unsqueeze(-1)
        repeated_position_ids_2 = unsqueeze_3.repeat(1, 1, 64)
        unsqueeze_3 = None
        sincos_2 = torch.gather(embed_positions_2, 1, repeated_position_ids_2)
        embed_positions_2 = repeated_position_ids_2 = None
        split_2 = torch.functional.split(sincos_2, 32, dim=-1)
        sincos_2 = None
        sin_6 = split_2[0]
        cos_6 = split_2[1]
        split_2 = None
        k_rot_4 = tensor_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 64, None),
            )
        ]
        k_pass_2 = tensor_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(64, None, None),
            )
        ]
        tensor_9 = None
        q_rot_4 = tensor_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 64, None),
            )
        ]
        q_pass_2 = tensor_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(64, None, None),
            )
        ]
        tensor_8 = None
        getitem_40 = sin_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_7 = torch.repeat_interleave(getitem_40, 2, 3)
        getitem_40 = None
        getitem_41 = cos_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_7 = torch.repeat_interleave(getitem_41, 2, 3)
        getitem_41 = None
        mul_16 = k_rot_4 * cos_7
        cos_7 = None
        x1_4 = k_rot_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_4 = k_rot_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        k_rot_4 = None
        neg_4 = -x2_4
        x2_4 = None
        x_4 = torch.stack((neg_4, x1_4), dim=-1)
        neg_4 = x1_4 = None
        flatten_4 = x_4.flatten(-2)
        x_4 = None
        mul_17 = flatten_4 * sin_7
        flatten_4 = sin_7 = None
        k_rot_5 = mul_16 + mul_17
        mul_16 = mul_17 = None
        getitem_44 = sin_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_6 = None
        sin_8 = torch.repeat_interleave(getitem_44, 2, 3)
        getitem_44 = None
        getitem_45 = cos_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_6 = None
        cos_8 = torch.repeat_interleave(getitem_45, 2, 3)
        getitem_45 = None
        mul_18 = q_rot_4 * cos_8
        cos_8 = None
        x1_5 = q_rot_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_5 = q_rot_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        q_rot_4 = None
        neg_5 = -x2_5
        x2_5 = None
        x_5 = torch.stack((neg_5, x1_5), dim=-1)
        neg_5 = x1_5 = None
        flatten_5 = x_5.flatten(-2)
        x_5 = None
        mul_19 = flatten_5 * sin_8
        flatten_5 = sin_8 = None
        q_rot_5 = mul_18 + mul_19
        mul_18 = mul_19 = None
        key_9 = torch.cat([k_rot_5, k_pass_2], dim=-1)
        k_rot_5 = k_pass_2 = None
        query_9 = torch.cat([q_rot_5, q_pass_2], dim=-1)
        q_rot_5 = q_pass_2 = None
        key_10 = key_9.permute(0, 2, 1, 3)
        key_9 = None
        query_10 = query_9.permute(0, 2, 1, 3)
        query_9 = None
        query_11 = query_10.to(torch.float32)
        query_10 = None
        key_11 = key_10.to(torch.float32)
        key_10 = None
        transpose_2 = key_11.transpose(-1, -2)
        key_11 = None
        attn_weights_12 = torch.matmul(query_11, transpose_2)
        query_11 = transpose_2 = None
        attn_weights_13 = (
            attn_weights_12
            / l_self_modules_transformer_modules_h_modules_2_modules_attn_scale_attn
        )
        attn_weights_12 = (
            l_self_modules_transformer_modules_h_modules_2_modules_attn_scale_attn
        ) = None
        causal_mask_7 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 36, None),
            )
        ]
        attn_weights_14 = attn_weights_13 + causal_mask_7
        attn_weights_13 = causal_mask_7 = None
        attn_weights_15 = torch.nn.functional.softmax(attn_weights_14, dim=-1)
        attn_weights_14 = None
        attn_weights_16 = attn_weights_15.to(torch.float16)
        attn_weights_15 = None
        attn_weights_17 = torch.nn.functional.dropout(
            attn_weights_16, 0.0, False, False
        )
        attn_weights_16 = None
        attn_output_8 = torch.matmul(attn_weights_17, value_5)
        attn_weights_17 = value_5 = None
        permute_11 = attn_output_8.permute(0, 2, 1, 3)
        attn_output_8 = None
        tensor_11 = permute_11.contiguous()
        permute_11 = None
        attn_output_9 = tensor_11.view((1, 36, 2048))
        tensor_11 = None
        attn_output_10 = torch._C._nn.linear(
            attn_output_9,
            l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_out_proj_parameters_weight_,
            None,
        )
        attn_output_9 = l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_out_proj_parameters_weight_ = (None)
        attn_output_11 = torch.nn.functional.dropout(attn_output_10, 0.0, False, False)
        attn_output_10 = None
        hidden_states_14 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_fc_in_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_fc_in_parameters_bias_,
        )
        hidden_states_13 = l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_fc_in_parameters_weight_ = l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_fc_in_parameters_bias_ = (None)
        mul_20 = 0.5 * hidden_states_14
        pow_3 = torch.pow(hidden_states_14, 3.0)
        mul_21 = 0.044715 * pow_3
        pow_3 = None
        add_18 = hidden_states_14 + mul_21
        hidden_states_14 = mul_21 = None
        mul_22 = 0.7978845608028654 * add_18
        add_18 = None
        tanh_2 = torch.tanh(mul_22)
        mul_22 = None
        add_19 = 1.0 + tanh_2
        tanh_2 = None
        hidden_states_15 = mul_20 * add_19
        mul_20 = add_19 = None
        hidden_states_16 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_fc_out_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_fc_out_parameters_bias_,
        )
        hidden_states_15 = l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_fc_out_parameters_weight_ = l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_fc_out_parameters_bias_ = (None)
        hidden_states_17 = torch.nn.functional.dropout(
            hidden_states_16, 0.0, False, False
        )
        hidden_states_16 = None
        add_20 = attn_output_11 + hidden_states_17
        attn_output_11 = hidden_states_17 = None
        hidden_states_18 = add_20 + hidden_states_12
        add_20 = hidden_states_12 = None
        hidden_states_19 = torch.nn.functional.layer_norm(
            hidden_states_18,
            (2048,),
            l_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_bias_
        ) = None
        query_12 = torch._C._nn.linear(
            hidden_states_19,
            l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_12 = torch._C._nn.linear(
            hidden_states_19,
            l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_6 = torch._C._nn.linear(
            hidden_states_19,
            l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        tensor_12 = query_12.view((1, 36, 16, 128))
        query_12 = None
        tensor_13 = key_12.view((1, 36, 16, 128))
        key_12 = None
        tensor_14 = value_6.view((1, 36, 16, 128))
        value_6 = None
        value_7 = tensor_14.permute(0, 2, 1, 3)
        tensor_14 = None
        embed_positions_3 = l_self_modules_transformer_modules_h_modules_3_modules_attn_embed_positions.repeat(
            1, 1, 1
        )
        l_self_modules_transformer_modules_h_modules_3_modules_attn_embed_positions = (
            None
        )
        unsqueeze_4 = position_ids.unsqueeze(-1)
        repeated_position_ids_3 = unsqueeze_4.repeat(1, 1, 64)
        unsqueeze_4 = None
        sincos_3 = torch.gather(embed_positions_3, 1, repeated_position_ids_3)
        embed_positions_3 = repeated_position_ids_3 = None
        split_3 = torch.functional.split(sincos_3, 32, dim=-1)
        sincos_3 = None
        sin_9 = split_3[0]
        cos_9 = split_3[1]
        split_3 = None
        k_rot_6 = tensor_13[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 64, None),
            )
        ]
        k_pass_3 = tensor_13[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(64, None, None),
            )
        ]
        tensor_13 = None
        q_rot_6 = tensor_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 64, None),
            )
        ]
        q_pass_3 = tensor_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(64, None, None),
            )
        ]
        tensor_12 = None
        getitem_55 = sin_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_10 = torch.repeat_interleave(getitem_55, 2, 3)
        getitem_55 = None
        getitem_56 = cos_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_10 = torch.repeat_interleave(getitem_56, 2, 3)
        getitem_56 = None
        mul_24 = k_rot_6 * cos_10
        cos_10 = None
        x1_6 = k_rot_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_6 = k_rot_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        k_rot_6 = None
        neg_6 = -x2_6
        x2_6 = None
        x_6 = torch.stack((neg_6, x1_6), dim=-1)
        neg_6 = x1_6 = None
        flatten_6 = x_6.flatten(-2)
        x_6 = None
        mul_25 = flatten_6 * sin_10
        flatten_6 = sin_10 = None
        k_rot_7 = mul_24 + mul_25
        mul_24 = mul_25 = None
        getitem_59 = sin_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_9 = None
        sin_11 = torch.repeat_interleave(getitem_59, 2, 3)
        getitem_59 = None
        getitem_60 = cos_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_9 = None
        cos_11 = torch.repeat_interleave(getitem_60, 2, 3)
        getitem_60 = None
        mul_26 = q_rot_6 * cos_11
        cos_11 = None
        x1_7 = q_rot_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_7 = q_rot_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        q_rot_6 = None
        neg_7 = -x2_7
        x2_7 = None
        x_7 = torch.stack((neg_7, x1_7), dim=-1)
        neg_7 = x1_7 = None
        flatten_7 = x_7.flatten(-2)
        x_7 = None
        mul_27 = flatten_7 * sin_11
        flatten_7 = sin_11 = None
        q_rot_7 = mul_26 + mul_27
        mul_26 = mul_27 = None
        key_13 = torch.cat([k_rot_7, k_pass_3], dim=-1)
        k_rot_7 = k_pass_3 = None
        query_13 = torch.cat([q_rot_7, q_pass_3], dim=-1)
        q_rot_7 = q_pass_3 = None
        key_14 = key_13.permute(0, 2, 1, 3)
        key_13 = None
        query_14 = query_13.permute(0, 2, 1, 3)
        query_13 = None
        query_15 = query_14.to(torch.float32)
        query_14 = None
        key_15 = key_14.to(torch.float32)
        key_14 = None
        transpose_3 = key_15.transpose(-1, -2)
        key_15 = None
        attn_weights_18 = torch.matmul(query_15, transpose_3)
        query_15 = transpose_3 = None
        attn_weights_19 = (
            attn_weights_18
            / l_self_modules_transformer_modules_h_modules_3_modules_attn_scale_attn
        )
        attn_weights_18 = (
            l_self_modules_transformer_modules_h_modules_3_modules_attn_scale_attn
        ) = None
        causal_mask_8 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 36, None),
            )
        ]
        attn_weights_20 = attn_weights_19 + causal_mask_8
        attn_weights_19 = causal_mask_8 = None
        attn_weights_21 = torch.nn.functional.softmax(attn_weights_20, dim=-1)
        attn_weights_20 = None
        attn_weights_22 = attn_weights_21.to(torch.float16)
        attn_weights_21 = None
        attn_weights_23 = torch.nn.functional.dropout(
            attn_weights_22, 0.0, False, False
        )
        attn_weights_22 = None
        attn_output_12 = torch.matmul(attn_weights_23, value_7)
        attn_weights_23 = value_7 = None
        permute_15 = attn_output_12.permute(0, 2, 1, 3)
        attn_output_12 = None
        tensor_15 = permute_15.contiguous()
        permute_15 = None
        attn_output_13 = tensor_15.view((1, 36, 2048))
        tensor_15 = None
        attn_output_14 = torch._C._nn.linear(
            attn_output_13,
            l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_out_proj_parameters_weight_,
            None,
        )
        attn_output_13 = l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_out_proj_parameters_weight_ = (None)
        attn_output_15 = torch.nn.functional.dropout(attn_output_14, 0.0, False, False)
        attn_output_14 = None
        hidden_states_20 = torch._C._nn.linear(
            hidden_states_19,
            l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_fc_in_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_fc_in_parameters_bias_,
        )
        hidden_states_19 = l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_fc_in_parameters_weight_ = l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_fc_in_parameters_bias_ = (None)
        mul_28 = 0.5 * hidden_states_20
        pow_4 = torch.pow(hidden_states_20, 3.0)
        mul_29 = 0.044715 * pow_4
        pow_4 = None
        add_25 = hidden_states_20 + mul_29
        hidden_states_20 = mul_29 = None
        mul_30 = 0.7978845608028654 * add_25
        add_25 = None
        tanh_3 = torch.tanh(mul_30)
        mul_30 = None
        add_26 = 1.0 + tanh_3
        tanh_3 = None
        hidden_states_21 = mul_28 * add_26
        mul_28 = add_26 = None
        hidden_states_22 = torch._C._nn.linear(
            hidden_states_21,
            l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_fc_out_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_fc_out_parameters_bias_,
        )
        hidden_states_21 = l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_fc_out_parameters_weight_ = l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_fc_out_parameters_bias_ = (None)
        hidden_states_23 = torch.nn.functional.dropout(
            hidden_states_22, 0.0, False, False
        )
        hidden_states_22 = None
        add_27 = attn_output_15 + hidden_states_23
        attn_output_15 = hidden_states_23 = None
        hidden_states_24 = add_27 + hidden_states_18
        add_27 = hidden_states_18 = None
        hidden_states_25 = torch.nn.functional.layer_norm(
            hidden_states_24,
            (2048,),
            l_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_bias_
        ) = None
        query_16 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_16 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_8 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        tensor_16 = query_16.view((1, 36, 16, 128))
        query_16 = None
        tensor_17 = key_16.view((1, 36, 16, 128))
        key_16 = None
        tensor_18 = value_8.view((1, 36, 16, 128))
        value_8 = None
        value_9 = tensor_18.permute(0, 2, 1, 3)
        tensor_18 = None
        embed_positions_4 = l_self_modules_transformer_modules_h_modules_4_modules_attn_embed_positions.repeat(
            1, 1, 1
        )
        l_self_modules_transformer_modules_h_modules_4_modules_attn_embed_positions = (
            None
        )
        unsqueeze_5 = position_ids.unsqueeze(-1)
        repeated_position_ids_4 = unsqueeze_5.repeat(1, 1, 64)
        unsqueeze_5 = None
        sincos_4 = torch.gather(embed_positions_4, 1, repeated_position_ids_4)
        embed_positions_4 = repeated_position_ids_4 = None
        split_4 = torch.functional.split(sincos_4, 32, dim=-1)
        sincos_4 = None
        sin_12 = split_4[0]
        cos_12 = split_4[1]
        split_4 = None
        k_rot_8 = tensor_17[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 64, None),
            )
        ]
        k_pass_4 = tensor_17[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(64, None, None),
            )
        ]
        tensor_17 = None
        q_rot_8 = tensor_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 64, None),
            )
        ]
        q_pass_4 = tensor_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(64, None, None),
            )
        ]
        tensor_16 = None
        getitem_70 = sin_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_13 = torch.repeat_interleave(getitem_70, 2, 3)
        getitem_70 = None
        getitem_71 = cos_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_13 = torch.repeat_interleave(getitem_71, 2, 3)
        getitem_71 = None
        mul_32 = k_rot_8 * cos_13
        cos_13 = None
        x1_8 = k_rot_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_8 = k_rot_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        k_rot_8 = None
        neg_8 = -x2_8
        x2_8 = None
        x_8 = torch.stack((neg_8, x1_8), dim=-1)
        neg_8 = x1_8 = None
        flatten_8 = x_8.flatten(-2)
        x_8 = None
        mul_33 = flatten_8 * sin_13
        flatten_8 = sin_13 = None
        k_rot_9 = mul_32 + mul_33
        mul_32 = mul_33 = None
        getitem_74 = sin_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_12 = None
        sin_14 = torch.repeat_interleave(getitem_74, 2, 3)
        getitem_74 = None
        getitem_75 = cos_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_12 = None
        cos_14 = torch.repeat_interleave(getitem_75, 2, 3)
        getitem_75 = None
        mul_34 = q_rot_8 * cos_14
        cos_14 = None
        x1_9 = q_rot_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_9 = q_rot_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        q_rot_8 = None
        neg_9 = -x2_9
        x2_9 = None
        x_9 = torch.stack((neg_9, x1_9), dim=-1)
        neg_9 = x1_9 = None
        flatten_9 = x_9.flatten(-2)
        x_9 = None
        mul_35 = flatten_9 * sin_14
        flatten_9 = sin_14 = None
        q_rot_9 = mul_34 + mul_35
        mul_34 = mul_35 = None
        key_17 = torch.cat([k_rot_9, k_pass_4], dim=-1)
        k_rot_9 = k_pass_4 = None
        query_17 = torch.cat([q_rot_9, q_pass_4], dim=-1)
        q_rot_9 = q_pass_4 = None
        key_18 = key_17.permute(0, 2, 1, 3)
        key_17 = None
        query_18 = query_17.permute(0, 2, 1, 3)
        query_17 = None
        query_19 = query_18.to(torch.float32)
        query_18 = None
        key_19 = key_18.to(torch.float32)
        key_18 = None
        transpose_4 = key_19.transpose(-1, -2)
        key_19 = None
        attn_weights_24 = torch.matmul(query_19, transpose_4)
        query_19 = transpose_4 = None
        attn_weights_25 = (
            attn_weights_24
            / l_self_modules_transformer_modules_h_modules_4_modules_attn_scale_attn
        )
        attn_weights_24 = (
            l_self_modules_transformer_modules_h_modules_4_modules_attn_scale_attn
        ) = None
        causal_mask_9 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 36, None),
            )
        ]
        attn_weights_26 = attn_weights_25 + causal_mask_9
        attn_weights_25 = causal_mask_9 = None
        attn_weights_27 = torch.nn.functional.softmax(attn_weights_26, dim=-1)
        attn_weights_26 = None
        attn_weights_28 = attn_weights_27.to(torch.float16)
        attn_weights_27 = None
        attn_weights_29 = torch.nn.functional.dropout(
            attn_weights_28, 0.0, False, False
        )
        attn_weights_28 = None
        attn_output_16 = torch.matmul(attn_weights_29, value_9)
        attn_weights_29 = value_9 = None
        permute_19 = attn_output_16.permute(0, 2, 1, 3)
        attn_output_16 = None
        tensor_19 = permute_19.contiguous()
        permute_19 = None
        attn_output_17 = tensor_19.view((1, 36, 2048))
        tensor_19 = None
        attn_output_18 = torch._C._nn.linear(
            attn_output_17,
            l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_out_proj_parameters_weight_,
            None,
        )
        attn_output_17 = l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_out_proj_parameters_weight_ = (None)
        attn_output_19 = torch.nn.functional.dropout(attn_output_18, 0.0, False, False)
        attn_output_18 = None
        hidden_states_26 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_fc_in_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_fc_in_parameters_bias_,
        )
        hidden_states_25 = l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_fc_in_parameters_weight_ = l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_fc_in_parameters_bias_ = (None)
        mul_36 = 0.5 * hidden_states_26
        pow_5 = torch.pow(hidden_states_26, 3.0)
        mul_37 = 0.044715 * pow_5
        pow_5 = None
        add_32 = hidden_states_26 + mul_37
        hidden_states_26 = mul_37 = None
        mul_38 = 0.7978845608028654 * add_32
        add_32 = None
        tanh_4 = torch.tanh(mul_38)
        mul_38 = None
        add_33 = 1.0 + tanh_4
        tanh_4 = None
        hidden_states_27 = mul_36 * add_33
        mul_36 = add_33 = None
        hidden_states_28 = torch._C._nn.linear(
            hidden_states_27,
            l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_fc_out_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_fc_out_parameters_bias_,
        )
        hidden_states_27 = l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_fc_out_parameters_weight_ = l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_fc_out_parameters_bias_ = (None)
        hidden_states_29 = torch.nn.functional.dropout(
            hidden_states_28, 0.0, False, False
        )
        hidden_states_28 = None
        add_34 = attn_output_19 + hidden_states_29
        attn_output_19 = hidden_states_29 = None
        hidden_states_30 = add_34 + hidden_states_24
        add_34 = hidden_states_24 = None
        hidden_states_31 = torch.nn.functional.layer_norm(
            hidden_states_30,
            (2048,),
            l_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_bias_
        ) = None
        query_20 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_20 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_10 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        tensor_20 = query_20.view((1, 36, 16, 128))
        query_20 = None
        tensor_21 = key_20.view((1, 36, 16, 128))
        key_20 = None
        tensor_22 = value_10.view((1, 36, 16, 128))
        value_10 = None
        value_11 = tensor_22.permute(0, 2, 1, 3)
        tensor_22 = None
        embed_positions_5 = l_self_modules_transformer_modules_h_modules_5_modules_attn_embed_positions.repeat(
            1, 1, 1
        )
        l_self_modules_transformer_modules_h_modules_5_modules_attn_embed_positions = (
            None
        )
        unsqueeze_6 = position_ids.unsqueeze(-1)
        repeated_position_ids_5 = unsqueeze_6.repeat(1, 1, 64)
        unsqueeze_6 = None
        sincos_5 = torch.gather(embed_positions_5, 1, repeated_position_ids_5)
        embed_positions_5 = repeated_position_ids_5 = None
        split_5 = torch.functional.split(sincos_5, 32, dim=-1)
        sincos_5 = None
        sin_15 = split_5[0]
        cos_15 = split_5[1]
        split_5 = None
        k_rot_10 = tensor_21[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 64, None),
            )
        ]
        k_pass_5 = tensor_21[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(64, None, None),
            )
        ]
        tensor_21 = None
        q_rot_10 = tensor_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 64, None),
            )
        ]
        q_pass_5 = tensor_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(64, None, None),
            )
        ]
        tensor_20 = None
        getitem_85 = sin_15[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_16 = torch.repeat_interleave(getitem_85, 2, 3)
        getitem_85 = None
        getitem_86 = cos_15[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_16 = torch.repeat_interleave(getitem_86, 2, 3)
        getitem_86 = None
        mul_40 = k_rot_10 * cos_16
        cos_16 = None
        x1_10 = k_rot_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_10 = k_rot_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        k_rot_10 = None
        neg_10 = -x2_10
        x2_10 = None
        x_10 = torch.stack((neg_10, x1_10), dim=-1)
        neg_10 = x1_10 = None
        flatten_10 = x_10.flatten(-2)
        x_10 = None
        mul_41 = flatten_10 * sin_16
        flatten_10 = sin_16 = None
        k_rot_11 = mul_40 + mul_41
        mul_40 = mul_41 = None
        getitem_89 = sin_15[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_15 = None
        sin_17 = torch.repeat_interleave(getitem_89, 2, 3)
        getitem_89 = None
        getitem_90 = cos_15[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_15 = None
        cos_17 = torch.repeat_interleave(getitem_90, 2, 3)
        getitem_90 = None
        mul_42 = q_rot_10 * cos_17
        cos_17 = None
        x1_11 = q_rot_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_11 = q_rot_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        q_rot_10 = None
        neg_11 = -x2_11
        x2_11 = None
        x_11 = torch.stack((neg_11, x1_11), dim=-1)
        neg_11 = x1_11 = None
        flatten_11 = x_11.flatten(-2)
        x_11 = None
        mul_43 = flatten_11 * sin_17
        flatten_11 = sin_17 = None
        q_rot_11 = mul_42 + mul_43
        mul_42 = mul_43 = None
        key_21 = torch.cat([k_rot_11, k_pass_5], dim=-1)
        k_rot_11 = k_pass_5 = None
        query_21 = torch.cat([q_rot_11, q_pass_5], dim=-1)
        q_rot_11 = q_pass_5 = None
        key_22 = key_21.permute(0, 2, 1, 3)
        key_21 = None
        query_22 = query_21.permute(0, 2, 1, 3)
        query_21 = None
        query_23 = query_22.to(torch.float32)
        query_22 = None
        key_23 = key_22.to(torch.float32)
        key_22 = None
        transpose_5 = key_23.transpose(-1, -2)
        key_23 = None
        attn_weights_30 = torch.matmul(query_23, transpose_5)
        query_23 = transpose_5 = None
        attn_weights_31 = (
            attn_weights_30
            / l_self_modules_transformer_modules_h_modules_5_modules_attn_scale_attn
        )
        attn_weights_30 = (
            l_self_modules_transformer_modules_h_modules_5_modules_attn_scale_attn
        ) = None
        causal_mask_10 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 36, None),
            )
        ]
        attn_weights_32 = attn_weights_31 + causal_mask_10
        attn_weights_31 = causal_mask_10 = None
        attn_weights_33 = torch.nn.functional.softmax(attn_weights_32, dim=-1)
        attn_weights_32 = None
        attn_weights_34 = attn_weights_33.to(torch.float16)
        attn_weights_33 = None
        attn_weights_35 = torch.nn.functional.dropout(
            attn_weights_34, 0.0, False, False
        )
        attn_weights_34 = None
        attn_output_20 = torch.matmul(attn_weights_35, value_11)
        attn_weights_35 = value_11 = None
        permute_23 = attn_output_20.permute(0, 2, 1, 3)
        attn_output_20 = None
        tensor_23 = permute_23.contiguous()
        permute_23 = None
        attn_output_21 = tensor_23.view((1, 36, 2048))
        tensor_23 = None
        attn_output_22 = torch._C._nn.linear(
            attn_output_21,
            l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_out_proj_parameters_weight_,
            None,
        )
        attn_output_21 = l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_out_proj_parameters_weight_ = (None)
        attn_output_23 = torch.nn.functional.dropout(attn_output_22, 0.0, False, False)
        attn_output_22 = None
        hidden_states_32 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_fc_in_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_fc_in_parameters_bias_,
        )
        hidden_states_31 = l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_fc_in_parameters_weight_ = l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_fc_in_parameters_bias_ = (None)
        mul_44 = 0.5 * hidden_states_32
        pow_6 = torch.pow(hidden_states_32, 3.0)
        mul_45 = 0.044715 * pow_6
        pow_6 = None
        add_39 = hidden_states_32 + mul_45
        hidden_states_32 = mul_45 = None
        mul_46 = 0.7978845608028654 * add_39
        add_39 = None
        tanh_5 = torch.tanh(mul_46)
        mul_46 = None
        add_40 = 1.0 + tanh_5
        tanh_5 = None
        hidden_states_33 = mul_44 * add_40
        mul_44 = add_40 = None
        hidden_states_34 = torch._C._nn.linear(
            hidden_states_33,
            l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_fc_out_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_fc_out_parameters_bias_,
        )
        hidden_states_33 = l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_fc_out_parameters_weight_ = l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_fc_out_parameters_bias_ = (None)
        hidden_states_35 = torch.nn.functional.dropout(
            hidden_states_34, 0.0, False, False
        )
        hidden_states_34 = None
        add_41 = attn_output_23 + hidden_states_35
        attn_output_23 = hidden_states_35 = None
        hidden_states_36 = add_41 + hidden_states_30
        add_41 = hidden_states_30 = None
        hidden_states_37 = torch.nn.functional.layer_norm(
            hidden_states_36,
            (2048,),
            l_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_bias_
        ) = None
        query_24 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_24 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_12 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        tensor_24 = query_24.view((1, 36, 16, 128))
        query_24 = None
        tensor_25 = key_24.view((1, 36, 16, 128))
        key_24 = None
        tensor_26 = value_12.view((1, 36, 16, 128))
        value_12 = None
        value_13 = tensor_26.permute(0, 2, 1, 3)
        tensor_26 = None
        embed_positions_6 = l_self_modules_transformer_modules_h_modules_6_modules_attn_embed_positions.repeat(
            1, 1, 1
        )
        l_self_modules_transformer_modules_h_modules_6_modules_attn_embed_positions = (
            None
        )
        unsqueeze_7 = position_ids.unsqueeze(-1)
        repeated_position_ids_6 = unsqueeze_7.repeat(1, 1, 64)
        unsqueeze_7 = None
        sincos_6 = torch.gather(embed_positions_6, 1, repeated_position_ids_6)
        embed_positions_6 = repeated_position_ids_6 = None
        split_6 = torch.functional.split(sincos_6, 32, dim=-1)
        sincos_6 = None
        sin_18 = split_6[0]
        cos_18 = split_6[1]
        split_6 = None
        k_rot_12 = tensor_25[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 64, None),
            )
        ]
        k_pass_6 = tensor_25[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(64, None, None),
            )
        ]
        tensor_25 = None
        q_rot_12 = tensor_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 64, None),
            )
        ]
        q_pass_6 = tensor_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(64, None, None),
            )
        ]
        tensor_24 = None
        getitem_100 = sin_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_19 = torch.repeat_interleave(getitem_100, 2, 3)
        getitem_100 = None
        getitem_101 = cos_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_19 = torch.repeat_interleave(getitem_101, 2, 3)
        getitem_101 = None
        mul_48 = k_rot_12 * cos_19
        cos_19 = None
        x1_12 = k_rot_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_12 = k_rot_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        k_rot_12 = None
        neg_12 = -x2_12
        x2_12 = None
        x_12 = torch.stack((neg_12, x1_12), dim=-1)
        neg_12 = x1_12 = None
        flatten_12 = x_12.flatten(-2)
        x_12 = None
        mul_49 = flatten_12 * sin_19
        flatten_12 = sin_19 = None
        k_rot_13 = mul_48 + mul_49
        mul_48 = mul_49 = None
        getitem_104 = sin_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_18 = None
        sin_20 = torch.repeat_interleave(getitem_104, 2, 3)
        getitem_104 = None
        getitem_105 = cos_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_18 = None
        cos_20 = torch.repeat_interleave(getitem_105, 2, 3)
        getitem_105 = None
        mul_50 = q_rot_12 * cos_20
        cos_20 = None
        x1_13 = q_rot_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_13 = q_rot_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        q_rot_12 = None
        neg_13 = -x2_13
        x2_13 = None
        x_13 = torch.stack((neg_13, x1_13), dim=-1)
        neg_13 = x1_13 = None
        flatten_13 = x_13.flatten(-2)
        x_13 = None
        mul_51 = flatten_13 * sin_20
        flatten_13 = sin_20 = None
        q_rot_13 = mul_50 + mul_51
        mul_50 = mul_51 = None
        key_25 = torch.cat([k_rot_13, k_pass_6], dim=-1)
        k_rot_13 = k_pass_6 = None
        query_25 = torch.cat([q_rot_13, q_pass_6], dim=-1)
        q_rot_13 = q_pass_6 = None
        key_26 = key_25.permute(0, 2, 1, 3)
        key_25 = None
        query_26 = query_25.permute(0, 2, 1, 3)
        query_25 = None
        query_27 = query_26.to(torch.float32)
        query_26 = None
        key_27 = key_26.to(torch.float32)
        key_26 = None
        transpose_6 = key_27.transpose(-1, -2)
        key_27 = None
        attn_weights_36 = torch.matmul(query_27, transpose_6)
        query_27 = transpose_6 = None
        attn_weights_37 = (
            attn_weights_36
            / l_self_modules_transformer_modules_h_modules_6_modules_attn_scale_attn
        )
        attn_weights_36 = (
            l_self_modules_transformer_modules_h_modules_6_modules_attn_scale_attn
        ) = None
        causal_mask_11 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 36, None),
            )
        ]
        attn_weights_38 = attn_weights_37 + causal_mask_11
        attn_weights_37 = causal_mask_11 = None
        attn_weights_39 = torch.nn.functional.softmax(attn_weights_38, dim=-1)
        attn_weights_38 = None
        attn_weights_40 = attn_weights_39.to(torch.float16)
        attn_weights_39 = None
        attn_weights_41 = torch.nn.functional.dropout(
            attn_weights_40, 0.0, False, False
        )
        attn_weights_40 = None
        attn_output_24 = torch.matmul(attn_weights_41, value_13)
        attn_weights_41 = value_13 = None
        permute_27 = attn_output_24.permute(0, 2, 1, 3)
        attn_output_24 = None
        tensor_27 = permute_27.contiguous()
        permute_27 = None
        attn_output_25 = tensor_27.view((1, 36, 2048))
        tensor_27 = None
        attn_output_26 = torch._C._nn.linear(
            attn_output_25,
            l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_out_proj_parameters_weight_,
            None,
        )
        attn_output_25 = l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_out_proj_parameters_weight_ = (None)
        attn_output_27 = torch.nn.functional.dropout(attn_output_26, 0.0, False, False)
        attn_output_26 = None
        hidden_states_38 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_fc_in_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_fc_in_parameters_bias_,
        )
        hidden_states_37 = l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_fc_in_parameters_weight_ = l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_fc_in_parameters_bias_ = (None)
        mul_52 = 0.5 * hidden_states_38
        pow_7 = torch.pow(hidden_states_38, 3.0)
        mul_53 = 0.044715 * pow_7
        pow_7 = None
        add_46 = hidden_states_38 + mul_53
        hidden_states_38 = mul_53 = None
        mul_54 = 0.7978845608028654 * add_46
        add_46 = None
        tanh_6 = torch.tanh(mul_54)
        mul_54 = None
        add_47 = 1.0 + tanh_6
        tanh_6 = None
        hidden_states_39 = mul_52 * add_47
        mul_52 = add_47 = None
        hidden_states_40 = torch._C._nn.linear(
            hidden_states_39,
            l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_fc_out_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_fc_out_parameters_bias_,
        )
        hidden_states_39 = l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_fc_out_parameters_weight_ = l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_fc_out_parameters_bias_ = (None)
        hidden_states_41 = torch.nn.functional.dropout(
            hidden_states_40, 0.0, False, False
        )
        hidden_states_40 = None
        add_48 = attn_output_27 + hidden_states_41
        attn_output_27 = hidden_states_41 = None
        hidden_states_42 = add_48 + hidden_states_36
        add_48 = hidden_states_36 = None
        hidden_states_43 = torch.nn.functional.layer_norm(
            hidden_states_42,
            (2048,),
            l_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_bias_
        ) = None
        query_28 = torch._C._nn.linear(
            hidden_states_43,
            l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_28 = torch._C._nn.linear(
            hidden_states_43,
            l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_14 = torch._C._nn.linear(
            hidden_states_43,
            l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        tensor_28 = query_28.view((1, 36, 16, 128))
        query_28 = None
        tensor_29 = key_28.view((1, 36, 16, 128))
        key_28 = None
        tensor_30 = value_14.view((1, 36, 16, 128))
        value_14 = None
        value_15 = tensor_30.permute(0, 2, 1, 3)
        tensor_30 = None
        embed_positions_7 = l_self_modules_transformer_modules_h_modules_7_modules_attn_embed_positions.repeat(
            1, 1, 1
        )
        l_self_modules_transformer_modules_h_modules_7_modules_attn_embed_positions = (
            None
        )
        unsqueeze_8 = position_ids.unsqueeze(-1)
        repeated_position_ids_7 = unsqueeze_8.repeat(1, 1, 64)
        unsqueeze_8 = None
        sincos_7 = torch.gather(embed_positions_7, 1, repeated_position_ids_7)
        embed_positions_7 = repeated_position_ids_7 = None
        split_7 = torch.functional.split(sincos_7, 32, dim=-1)
        sincos_7 = None
        sin_21 = split_7[0]
        cos_21 = split_7[1]
        split_7 = None
        k_rot_14 = tensor_29[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 64, None),
            )
        ]
        k_pass_7 = tensor_29[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(64, None, None),
            )
        ]
        tensor_29 = None
        q_rot_14 = tensor_28[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 64, None),
            )
        ]
        q_pass_7 = tensor_28[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(64, None, None),
            )
        ]
        tensor_28 = None
        getitem_115 = sin_21[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_22 = torch.repeat_interleave(getitem_115, 2, 3)
        getitem_115 = None
        getitem_116 = cos_21[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_22 = torch.repeat_interleave(getitem_116, 2, 3)
        getitem_116 = None
        mul_56 = k_rot_14 * cos_22
        cos_22 = None
        x1_14 = k_rot_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_14 = k_rot_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        k_rot_14 = None
        neg_14 = -x2_14
        x2_14 = None
        x_14 = torch.stack((neg_14, x1_14), dim=-1)
        neg_14 = x1_14 = None
        flatten_14 = x_14.flatten(-2)
        x_14 = None
        mul_57 = flatten_14 * sin_22
        flatten_14 = sin_22 = None
        k_rot_15 = mul_56 + mul_57
        mul_56 = mul_57 = None
        getitem_119 = sin_21[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_21 = None
        sin_23 = torch.repeat_interleave(getitem_119, 2, 3)
        getitem_119 = None
        getitem_120 = cos_21[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_21 = None
        cos_23 = torch.repeat_interleave(getitem_120, 2, 3)
        getitem_120 = None
        mul_58 = q_rot_14 * cos_23
        cos_23 = None
        x1_15 = q_rot_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_15 = q_rot_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        q_rot_14 = None
        neg_15 = -x2_15
        x2_15 = None
        x_15 = torch.stack((neg_15, x1_15), dim=-1)
        neg_15 = x1_15 = None
        flatten_15 = x_15.flatten(-2)
        x_15 = None
        mul_59 = flatten_15 * sin_23
        flatten_15 = sin_23 = None
        q_rot_15 = mul_58 + mul_59
        mul_58 = mul_59 = None
        key_29 = torch.cat([k_rot_15, k_pass_7], dim=-1)
        k_rot_15 = k_pass_7 = None
        query_29 = torch.cat([q_rot_15, q_pass_7], dim=-1)
        q_rot_15 = q_pass_7 = None
        key_30 = key_29.permute(0, 2, 1, 3)
        key_29 = None
        query_30 = query_29.permute(0, 2, 1, 3)
        query_29 = None
        query_31 = query_30.to(torch.float32)
        query_30 = None
        key_31 = key_30.to(torch.float32)
        key_30 = None
        transpose_7 = key_31.transpose(-1, -2)
        key_31 = None
        attn_weights_42 = torch.matmul(query_31, transpose_7)
        query_31 = transpose_7 = None
        attn_weights_43 = (
            attn_weights_42
            / l_self_modules_transformer_modules_h_modules_7_modules_attn_scale_attn
        )
        attn_weights_42 = (
            l_self_modules_transformer_modules_h_modules_7_modules_attn_scale_attn
        ) = None
        causal_mask_12 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 36, None),
            )
        ]
        attn_weights_44 = attn_weights_43 + causal_mask_12
        attn_weights_43 = causal_mask_12 = None
        attn_weights_45 = torch.nn.functional.softmax(attn_weights_44, dim=-1)
        attn_weights_44 = None
        attn_weights_46 = attn_weights_45.to(torch.float16)
        attn_weights_45 = None
        attn_weights_47 = torch.nn.functional.dropout(
            attn_weights_46, 0.0, False, False
        )
        attn_weights_46 = None
        attn_output_28 = torch.matmul(attn_weights_47, value_15)
        attn_weights_47 = value_15 = None
        permute_31 = attn_output_28.permute(0, 2, 1, 3)
        attn_output_28 = None
        tensor_31 = permute_31.contiguous()
        permute_31 = None
        attn_output_29 = tensor_31.view((1, 36, 2048))
        tensor_31 = None
        attn_output_30 = torch._C._nn.linear(
            attn_output_29,
            l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_out_proj_parameters_weight_,
            None,
        )
        attn_output_29 = l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_out_proj_parameters_weight_ = (None)
        attn_output_31 = torch.nn.functional.dropout(attn_output_30, 0.0, False, False)
        attn_output_30 = None
        hidden_states_44 = torch._C._nn.linear(
            hidden_states_43,
            l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_fc_in_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_fc_in_parameters_bias_,
        )
        hidden_states_43 = l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_fc_in_parameters_weight_ = l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_fc_in_parameters_bias_ = (None)
        mul_60 = 0.5 * hidden_states_44
        pow_8 = torch.pow(hidden_states_44, 3.0)
        mul_61 = 0.044715 * pow_8
        pow_8 = None
        add_53 = hidden_states_44 + mul_61
        hidden_states_44 = mul_61 = None
        mul_62 = 0.7978845608028654 * add_53
        add_53 = None
        tanh_7 = torch.tanh(mul_62)
        mul_62 = None
        add_54 = 1.0 + tanh_7
        tanh_7 = None
        hidden_states_45 = mul_60 * add_54
        mul_60 = add_54 = None
        hidden_states_46 = torch._C._nn.linear(
            hidden_states_45,
            l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_fc_out_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_fc_out_parameters_bias_,
        )
        hidden_states_45 = l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_fc_out_parameters_weight_ = l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_fc_out_parameters_bias_ = (None)
        hidden_states_47 = torch.nn.functional.dropout(
            hidden_states_46, 0.0, False, False
        )
        hidden_states_46 = None
        add_55 = attn_output_31 + hidden_states_47
        attn_output_31 = hidden_states_47 = None
        hidden_states_48 = add_55 + hidden_states_42
        add_55 = hidden_states_42 = None
        hidden_states_49 = torch.nn.functional.layer_norm(
            hidden_states_48,
            (2048,),
            l_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_bias_
        ) = None
        query_32 = torch._C._nn.linear(
            hidden_states_49,
            l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_32 = torch._C._nn.linear(
            hidden_states_49,
            l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_16 = torch._C._nn.linear(
            hidden_states_49,
            l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        tensor_32 = query_32.view((1, 36, 16, 128))
        query_32 = None
        tensor_33 = key_32.view((1, 36, 16, 128))
        key_32 = None
        tensor_34 = value_16.view((1, 36, 16, 128))
        value_16 = None
        value_17 = tensor_34.permute(0, 2, 1, 3)
        tensor_34 = None
        embed_positions_8 = l_self_modules_transformer_modules_h_modules_8_modules_attn_embed_positions.repeat(
            1, 1, 1
        )
        l_self_modules_transformer_modules_h_modules_8_modules_attn_embed_positions = (
            None
        )
        unsqueeze_9 = position_ids.unsqueeze(-1)
        repeated_position_ids_8 = unsqueeze_9.repeat(1, 1, 64)
        unsqueeze_9 = None
        sincos_8 = torch.gather(embed_positions_8, 1, repeated_position_ids_8)
        embed_positions_8 = repeated_position_ids_8 = None
        split_8 = torch.functional.split(sincos_8, 32, dim=-1)
        sincos_8 = None
        sin_24 = split_8[0]
        cos_24 = split_8[1]
        split_8 = None
        k_rot_16 = tensor_33[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 64, None),
            )
        ]
        k_pass_8 = tensor_33[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(64, None, None),
            )
        ]
        tensor_33 = None
        q_rot_16 = tensor_32[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 64, None),
            )
        ]
        q_pass_8 = tensor_32[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(64, None, None),
            )
        ]
        tensor_32 = None
        getitem_130 = sin_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_25 = torch.repeat_interleave(getitem_130, 2, 3)
        getitem_130 = None
        getitem_131 = cos_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_25 = torch.repeat_interleave(getitem_131, 2, 3)
        getitem_131 = None
        mul_64 = k_rot_16 * cos_25
        cos_25 = None
        x1_16 = k_rot_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_16 = k_rot_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        k_rot_16 = None
        neg_16 = -x2_16
        x2_16 = None
        x_16 = torch.stack((neg_16, x1_16), dim=-1)
        neg_16 = x1_16 = None
        flatten_16 = x_16.flatten(-2)
        x_16 = None
        mul_65 = flatten_16 * sin_25
        flatten_16 = sin_25 = None
        k_rot_17 = mul_64 + mul_65
        mul_64 = mul_65 = None
        getitem_134 = sin_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_24 = None
        sin_26 = torch.repeat_interleave(getitem_134, 2, 3)
        getitem_134 = None
        getitem_135 = cos_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_24 = None
        cos_26 = torch.repeat_interleave(getitem_135, 2, 3)
        getitem_135 = None
        mul_66 = q_rot_16 * cos_26
        cos_26 = None
        x1_17 = q_rot_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_17 = q_rot_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        q_rot_16 = None
        neg_17 = -x2_17
        x2_17 = None
        x_17 = torch.stack((neg_17, x1_17), dim=-1)
        neg_17 = x1_17 = None
        flatten_17 = x_17.flatten(-2)
        x_17 = None
        mul_67 = flatten_17 * sin_26
        flatten_17 = sin_26 = None
        q_rot_17 = mul_66 + mul_67
        mul_66 = mul_67 = None
        key_33 = torch.cat([k_rot_17, k_pass_8], dim=-1)
        k_rot_17 = k_pass_8 = None
        query_33 = torch.cat([q_rot_17, q_pass_8], dim=-1)
        q_rot_17 = q_pass_8 = None
        key_34 = key_33.permute(0, 2, 1, 3)
        key_33 = None
        query_34 = query_33.permute(0, 2, 1, 3)
        query_33 = None
        query_35 = query_34.to(torch.float32)
        query_34 = None
        key_35 = key_34.to(torch.float32)
        key_34 = None
        transpose_8 = key_35.transpose(-1, -2)
        key_35 = None
        attn_weights_48 = torch.matmul(query_35, transpose_8)
        query_35 = transpose_8 = None
        attn_weights_49 = (
            attn_weights_48
            / l_self_modules_transformer_modules_h_modules_8_modules_attn_scale_attn
        )
        attn_weights_48 = (
            l_self_modules_transformer_modules_h_modules_8_modules_attn_scale_attn
        ) = None
        causal_mask_13 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 36, None),
            )
        ]
        attn_weights_50 = attn_weights_49 + causal_mask_13
        attn_weights_49 = causal_mask_13 = None
        attn_weights_51 = torch.nn.functional.softmax(attn_weights_50, dim=-1)
        attn_weights_50 = None
        attn_weights_52 = attn_weights_51.to(torch.float16)
        attn_weights_51 = None
        attn_weights_53 = torch.nn.functional.dropout(
            attn_weights_52, 0.0, False, False
        )
        attn_weights_52 = None
        attn_output_32 = torch.matmul(attn_weights_53, value_17)
        attn_weights_53 = value_17 = None
        permute_35 = attn_output_32.permute(0, 2, 1, 3)
        attn_output_32 = None
        tensor_35 = permute_35.contiguous()
        permute_35 = None
        attn_output_33 = tensor_35.view((1, 36, 2048))
        tensor_35 = None
        attn_output_34 = torch._C._nn.linear(
            attn_output_33,
            l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_out_proj_parameters_weight_,
            None,
        )
        attn_output_33 = l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_out_proj_parameters_weight_ = (None)
        attn_output_35 = torch.nn.functional.dropout(attn_output_34, 0.0, False, False)
        attn_output_34 = None
        hidden_states_50 = torch._C._nn.linear(
            hidden_states_49,
            l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_fc_in_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_fc_in_parameters_bias_,
        )
        hidden_states_49 = l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_fc_in_parameters_weight_ = l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_fc_in_parameters_bias_ = (None)
        mul_68 = 0.5 * hidden_states_50
        pow_9 = torch.pow(hidden_states_50, 3.0)
        mul_69 = 0.044715 * pow_9
        pow_9 = None
        add_60 = hidden_states_50 + mul_69
        hidden_states_50 = mul_69 = None
        mul_70 = 0.7978845608028654 * add_60
        add_60 = None
        tanh_8 = torch.tanh(mul_70)
        mul_70 = None
        add_61 = 1.0 + tanh_8
        tanh_8 = None
        hidden_states_51 = mul_68 * add_61
        mul_68 = add_61 = None
        hidden_states_52 = torch._C._nn.linear(
            hidden_states_51,
            l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_fc_out_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_fc_out_parameters_bias_,
        )
        hidden_states_51 = l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_fc_out_parameters_weight_ = l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_fc_out_parameters_bias_ = (None)
        hidden_states_53 = torch.nn.functional.dropout(
            hidden_states_52, 0.0, False, False
        )
        hidden_states_52 = None
        add_62 = attn_output_35 + hidden_states_53
        attn_output_35 = hidden_states_53 = None
        hidden_states_54 = add_62 + hidden_states_48
        add_62 = hidden_states_48 = None
        hidden_states_55 = torch.nn.functional.layer_norm(
            hidden_states_54,
            (2048,),
            l_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_bias_
        ) = None
        query_36 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_36 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_18 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        tensor_36 = query_36.view((1, 36, 16, 128))
        query_36 = None
        tensor_37 = key_36.view((1, 36, 16, 128))
        key_36 = None
        tensor_38 = value_18.view((1, 36, 16, 128))
        value_18 = None
        value_19 = tensor_38.permute(0, 2, 1, 3)
        tensor_38 = None
        embed_positions_9 = l_self_modules_transformer_modules_h_modules_9_modules_attn_embed_positions.repeat(
            1, 1, 1
        )
        l_self_modules_transformer_modules_h_modules_9_modules_attn_embed_positions = (
            None
        )
        unsqueeze_10 = position_ids.unsqueeze(-1)
        repeated_position_ids_9 = unsqueeze_10.repeat(1, 1, 64)
        unsqueeze_10 = None
        sincos_9 = torch.gather(embed_positions_9, 1, repeated_position_ids_9)
        embed_positions_9 = repeated_position_ids_9 = None
        split_9 = torch.functional.split(sincos_9, 32, dim=-1)
        sincos_9 = None
        sin_27 = split_9[0]
        cos_27 = split_9[1]
        split_9 = None
        k_rot_18 = tensor_37[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 64, None),
            )
        ]
        k_pass_9 = tensor_37[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(64, None, None),
            )
        ]
        tensor_37 = None
        q_rot_18 = tensor_36[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 64, None),
            )
        ]
        q_pass_9 = tensor_36[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(64, None, None),
            )
        ]
        tensor_36 = None
        getitem_145 = sin_27[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_28 = torch.repeat_interleave(getitem_145, 2, 3)
        getitem_145 = None
        getitem_146 = cos_27[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_28 = torch.repeat_interleave(getitem_146, 2, 3)
        getitem_146 = None
        mul_72 = k_rot_18 * cos_28
        cos_28 = None
        x1_18 = k_rot_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_18 = k_rot_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        k_rot_18 = None
        neg_18 = -x2_18
        x2_18 = None
        x_18 = torch.stack((neg_18, x1_18), dim=-1)
        neg_18 = x1_18 = None
        flatten_18 = x_18.flatten(-2)
        x_18 = None
        mul_73 = flatten_18 * sin_28
        flatten_18 = sin_28 = None
        k_rot_19 = mul_72 + mul_73
        mul_72 = mul_73 = None
        getitem_149 = sin_27[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_27 = None
        sin_29 = torch.repeat_interleave(getitem_149, 2, 3)
        getitem_149 = None
        getitem_150 = cos_27[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_27 = None
        cos_29 = torch.repeat_interleave(getitem_150, 2, 3)
        getitem_150 = None
        mul_74 = q_rot_18 * cos_29
        cos_29 = None
        x1_19 = q_rot_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_19 = q_rot_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        q_rot_18 = None
        neg_19 = -x2_19
        x2_19 = None
        x_19 = torch.stack((neg_19, x1_19), dim=-1)
        neg_19 = x1_19 = None
        flatten_19 = x_19.flatten(-2)
        x_19 = None
        mul_75 = flatten_19 * sin_29
        flatten_19 = sin_29 = None
        q_rot_19 = mul_74 + mul_75
        mul_74 = mul_75 = None
        key_37 = torch.cat([k_rot_19, k_pass_9], dim=-1)
        k_rot_19 = k_pass_9 = None
        query_37 = torch.cat([q_rot_19, q_pass_9], dim=-1)
        q_rot_19 = q_pass_9 = None
        key_38 = key_37.permute(0, 2, 1, 3)
        key_37 = None
        query_38 = query_37.permute(0, 2, 1, 3)
        query_37 = None
        query_39 = query_38.to(torch.float32)
        query_38 = None
        key_39 = key_38.to(torch.float32)
        key_38 = None
        transpose_9 = key_39.transpose(-1, -2)
        key_39 = None
        attn_weights_54 = torch.matmul(query_39, transpose_9)
        query_39 = transpose_9 = None
        attn_weights_55 = (
            attn_weights_54
            / l_self_modules_transformer_modules_h_modules_9_modules_attn_scale_attn
        )
        attn_weights_54 = (
            l_self_modules_transformer_modules_h_modules_9_modules_attn_scale_attn
        ) = None
        causal_mask_14 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 36, None),
            )
        ]
        attn_weights_56 = attn_weights_55 + causal_mask_14
        attn_weights_55 = causal_mask_14 = None
        attn_weights_57 = torch.nn.functional.softmax(attn_weights_56, dim=-1)
        attn_weights_56 = None
        attn_weights_58 = attn_weights_57.to(torch.float16)
        attn_weights_57 = None
        attn_weights_59 = torch.nn.functional.dropout(
            attn_weights_58, 0.0, False, False
        )
        attn_weights_58 = None
        attn_output_36 = torch.matmul(attn_weights_59, value_19)
        attn_weights_59 = value_19 = None
        permute_39 = attn_output_36.permute(0, 2, 1, 3)
        attn_output_36 = None
        tensor_39 = permute_39.contiguous()
        permute_39 = None
        attn_output_37 = tensor_39.view((1, 36, 2048))
        tensor_39 = None
        attn_output_38 = torch._C._nn.linear(
            attn_output_37,
            l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_out_proj_parameters_weight_,
            None,
        )
        attn_output_37 = l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_out_proj_parameters_weight_ = (None)
        attn_output_39 = torch.nn.functional.dropout(attn_output_38, 0.0, False, False)
        attn_output_38 = None
        hidden_states_56 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_fc_in_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_fc_in_parameters_bias_,
        )
        hidden_states_55 = l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_fc_in_parameters_weight_ = l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_fc_in_parameters_bias_ = (None)
        mul_76 = 0.5 * hidden_states_56
        pow_10 = torch.pow(hidden_states_56, 3.0)
        mul_77 = 0.044715 * pow_10
        pow_10 = None
        add_67 = hidden_states_56 + mul_77
        hidden_states_56 = mul_77 = None
        mul_78 = 0.7978845608028654 * add_67
        add_67 = None
        tanh_9 = torch.tanh(mul_78)
        mul_78 = None
        add_68 = 1.0 + tanh_9
        tanh_9 = None
        hidden_states_57 = mul_76 * add_68
        mul_76 = add_68 = None
        hidden_states_58 = torch._C._nn.linear(
            hidden_states_57,
            l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_fc_out_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_fc_out_parameters_bias_,
        )
        hidden_states_57 = l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_fc_out_parameters_weight_ = l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_fc_out_parameters_bias_ = (None)
        hidden_states_59 = torch.nn.functional.dropout(
            hidden_states_58, 0.0, False, False
        )
        hidden_states_58 = None
        add_69 = attn_output_39 + hidden_states_59
        attn_output_39 = hidden_states_59 = None
        hidden_states_60 = add_69 + hidden_states_54
        add_69 = hidden_states_54 = None
        hidden_states_61 = torch.nn.functional.layer_norm(
            hidden_states_60,
            (2048,),
            l_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_bias_ = (None)
        query_40 = torch._C._nn.linear(
            hidden_states_61,
            l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_40 = torch._C._nn.linear(
            hidden_states_61,
            l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_20 = torch._C._nn.linear(
            hidden_states_61,
            l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        tensor_40 = query_40.view((1, 36, 16, 128))
        query_40 = None
        tensor_41 = key_40.view((1, 36, 16, 128))
        key_40 = None
        tensor_42 = value_20.view((1, 36, 16, 128))
        value_20 = None
        value_21 = tensor_42.permute(0, 2, 1, 3)
        tensor_42 = None
        embed_positions_10 = l_self_modules_transformer_modules_h_modules_10_modules_attn_embed_positions.repeat(
            1, 1, 1
        )
        l_self_modules_transformer_modules_h_modules_10_modules_attn_embed_positions = (
            None
        )
        unsqueeze_11 = position_ids.unsqueeze(-1)
        repeated_position_ids_10 = unsqueeze_11.repeat(1, 1, 64)
        unsqueeze_11 = None
        sincos_10 = torch.gather(embed_positions_10, 1, repeated_position_ids_10)
        embed_positions_10 = repeated_position_ids_10 = None
        split_10 = torch.functional.split(sincos_10, 32, dim=-1)
        sincos_10 = None
        sin_30 = split_10[0]
        cos_30 = split_10[1]
        split_10 = None
        k_rot_20 = tensor_41[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 64, None),
            )
        ]
        k_pass_10 = tensor_41[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(64, None, None),
            )
        ]
        tensor_41 = None
        q_rot_20 = tensor_40[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 64, None),
            )
        ]
        q_pass_10 = tensor_40[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(64, None, None),
            )
        ]
        tensor_40 = None
        getitem_160 = sin_30[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_31 = torch.repeat_interleave(getitem_160, 2, 3)
        getitem_160 = None
        getitem_161 = cos_30[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_31 = torch.repeat_interleave(getitem_161, 2, 3)
        getitem_161 = None
        mul_80 = k_rot_20 * cos_31
        cos_31 = None
        x1_20 = k_rot_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_20 = k_rot_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        k_rot_20 = None
        neg_20 = -x2_20
        x2_20 = None
        x_20 = torch.stack((neg_20, x1_20), dim=-1)
        neg_20 = x1_20 = None
        flatten_20 = x_20.flatten(-2)
        x_20 = None
        mul_81 = flatten_20 * sin_31
        flatten_20 = sin_31 = None
        k_rot_21 = mul_80 + mul_81
        mul_80 = mul_81 = None
        getitem_164 = sin_30[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_30 = None
        sin_32 = torch.repeat_interleave(getitem_164, 2, 3)
        getitem_164 = None
        getitem_165 = cos_30[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_30 = None
        cos_32 = torch.repeat_interleave(getitem_165, 2, 3)
        getitem_165 = None
        mul_82 = q_rot_20 * cos_32
        cos_32 = None
        x1_21 = q_rot_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_21 = q_rot_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        q_rot_20 = None
        neg_21 = -x2_21
        x2_21 = None
        x_21 = torch.stack((neg_21, x1_21), dim=-1)
        neg_21 = x1_21 = None
        flatten_21 = x_21.flatten(-2)
        x_21 = None
        mul_83 = flatten_21 * sin_32
        flatten_21 = sin_32 = None
        q_rot_21 = mul_82 + mul_83
        mul_82 = mul_83 = None
        key_41 = torch.cat([k_rot_21, k_pass_10], dim=-1)
        k_rot_21 = k_pass_10 = None
        query_41 = torch.cat([q_rot_21, q_pass_10], dim=-1)
        q_rot_21 = q_pass_10 = None
        key_42 = key_41.permute(0, 2, 1, 3)
        key_41 = None
        query_42 = query_41.permute(0, 2, 1, 3)
        query_41 = None
        query_43 = query_42.to(torch.float32)
        query_42 = None
        key_43 = key_42.to(torch.float32)
        key_42 = None
        transpose_10 = key_43.transpose(-1, -2)
        key_43 = None
        attn_weights_60 = torch.matmul(query_43, transpose_10)
        query_43 = transpose_10 = None
        attn_weights_61 = (
            attn_weights_60
            / l_self_modules_transformer_modules_h_modules_10_modules_attn_scale_attn
        )
        attn_weights_60 = (
            l_self_modules_transformer_modules_h_modules_10_modules_attn_scale_attn
        ) = None
        causal_mask_15 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 36, None),
            )
        ]
        attn_weights_62 = attn_weights_61 + causal_mask_15
        attn_weights_61 = causal_mask_15 = None
        attn_weights_63 = torch.nn.functional.softmax(attn_weights_62, dim=-1)
        attn_weights_62 = None
        attn_weights_64 = attn_weights_63.to(torch.float16)
        attn_weights_63 = None
        attn_weights_65 = torch.nn.functional.dropout(
            attn_weights_64, 0.0, False, False
        )
        attn_weights_64 = None
        attn_output_40 = torch.matmul(attn_weights_65, value_21)
        attn_weights_65 = value_21 = None
        permute_43 = attn_output_40.permute(0, 2, 1, 3)
        attn_output_40 = None
        tensor_43 = permute_43.contiguous()
        permute_43 = None
        attn_output_41 = tensor_43.view((1, 36, 2048))
        tensor_43 = None
        attn_output_42 = torch._C._nn.linear(
            attn_output_41,
            l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_out_proj_parameters_weight_,
            None,
        )
        attn_output_41 = l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_out_proj_parameters_weight_ = (None)
        attn_output_43 = torch.nn.functional.dropout(attn_output_42, 0.0, False, False)
        attn_output_42 = None
        hidden_states_62 = torch._C._nn.linear(
            hidden_states_61,
            l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_fc_in_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_fc_in_parameters_bias_,
        )
        hidden_states_61 = l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_fc_in_parameters_weight_ = l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_fc_in_parameters_bias_ = (None)
        mul_84 = 0.5 * hidden_states_62
        pow_11 = torch.pow(hidden_states_62, 3.0)
        mul_85 = 0.044715 * pow_11
        pow_11 = None
        add_74 = hidden_states_62 + mul_85
        hidden_states_62 = mul_85 = None
        mul_86 = 0.7978845608028654 * add_74
        add_74 = None
        tanh_10 = torch.tanh(mul_86)
        mul_86 = None
        add_75 = 1.0 + tanh_10
        tanh_10 = None
        hidden_states_63 = mul_84 * add_75
        mul_84 = add_75 = None
        hidden_states_64 = torch._C._nn.linear(
            hidden_states_63,
            l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_fc_out_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_fc_out_parameters_bias_,
        )
        hidden_states_63 = l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_fc_out_parameters_weight_ = l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_fc_out_parameters_bias_ = (None)
        hidden_states_65 = torch.nn.functional.dropout(
            hidden_states_64, 0.0, False, False
        )
        hidden_states_64 = None
        add_76 = attn_output_43 + hidden_states_65
        attn_output_43 = hidden_states_65 = None
        hidden_states_66 = add_76 + hidden_states_60
        add_76 = hidden_states_60 = None
        hidden_states_67 = torch.nn.functional.layer_norm(
            hidden_states_66,
            (2048,),
            l_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_bias_ = (None)
        query_44 = torch._C._nn.linear(
            hidden_states_67,
            l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_44 = torch._C._nn.linear(
            hidden_states_67,
            l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_22 = torch._C._nn.linear(
            hidden_states_67,
            l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        tensor_44 = query_44.view((1, 36, 16, 128))
        query_44 = None
        tensor_45 = key_44.view((1, 36, 16, 128))
        key_44 = None
        tensor_46 = value_22.view((1, 36, 16, 128))
        value_22 = None
        value_23 = tensor_46.permute(0, 2, 1, 3)
        tensor_46 = None
        embed_positions_11 = l_self_modules_transformer_modules_h_modules_11_modules_attn_embed_positions.repeat(
            1, 1, 1
        )
        l_self_modules_transformer_modules_h_modules_11_modules_attn_embed_positions = (
            None
        )
        unsqueeze_12 = position_ids.unsqueeze(-1)
        position_ids = None
        repeated_position_ids_11 = unsqueeze_12.repeat(1, 1, 64)
        unsqueeze_12 = None
        sincos_11 = torch.gather(embed_positions_11, 1, repeated_position_ids_11)
        embed_positions_11 = repeated_position_ids_11 = None
        split_11 = torch.functional.split(sincos_11, 32, dim=-1)
        sincos_11 = None
        sin_33 = split_11[0]
        cos_33 = split_11[1]
        split_11 = None
        k_rot_22 = tensor_45[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 64, None),
            )
        ]
        k_pass_11 = tensor_45[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(64, None, None),
            )
        ]
        tensor_45 = None
        q_rot_22 = tensor_44[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 64, None),
            )
        ]
        q_pass_11 = tensor_44[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(64, None, None),
            )
        ]
        tensor_44 = None
        getitem_175 = sin_33[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_34 = torch.repeat_interleave(getitem_175, 2, 3)
        getitem_175 = None
        getitem_176 = cos_33[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_34 = torch.repeat_interleave(getitem_176, 2, 3)
        getitem_176 = None
        mul_88 = k_rot_22 * cos_34
        cos_34 = None
        x1_22 = k_rot_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_22 = k_rot_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        k_rot_22 = None
        neg_22 = -x2_22
        x2_22 = None
        x_22 = torch.stack((neg_22, x1_22), dim=-1)
        neg_22 = x1_22 = None
        flatten_22 = x_22.flatten(-2)
        x_22 = None
        mul_89 = flatten_22 * sin_34
        flatten_22 = sin_34 = None
        k_rot_23 = mul_88 + mul_89
        mul_88 = mul_89 = None
        getitem_179 = sin_33[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_33 = None
        sin_35 = torch.repeat_interleave(getitem_179, 2, 3)
        getitem_179 = None
        getitem_180 = cos_33[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_33 = None
        cos_35 = torch.repeat_interleave(getitem_180, 2, 3)
        getitem_180 = None
        mul_90 = q_rot_22 * cos_35
        cos_35 = None
        x1_23 = q_rot_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_23 = q_rot_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        q_rot_22 = None
        neg_23 = -x2_23
        x2_23 = None
        x_23 = torch.stack((neg_23, x1_23), dim=-1)
        neg_23 = x1_23 = None
        flatten_23 = x_23.flatten(-2)
        x_23 = None
        mul_91 = flatten_23 * sin_35
        flatten_23 = sin_35 = None
        q_rot_23 = mul_90 + mul_91
        mul_90 = mul_91 = None
        key_45 = torch.cat([k_rot_23, k_pass_11], dim=-1)
        k_rot_23 = k_pass_11 = None
        query_45 = torch.cat([q_rot_23, q_pass_11], dim=-1)
        q_rot_23 = q_pass_11 = None
        key_46 = key_45.permute(0, 2, 1, 3)
        key_45 = None
        query_46 = query_45.permute(0, 2, 1, 3)
        query_45 = None
        query_47 = query_46.to(torch.float32)
        query_46 = None
        key_47 = key_46.to(torch.float32)
        key_46 = None
        transpose_11 = key_47.transpose(-1, -2)
        key_47 = None
        attn_weights_66 = torch.matmul(query_47, transpose_11)
        query_47 = transpose_11 = None
        attn_weights_67 = (
            attn_weights_66
            / l_self_modules_transformer_modules_h_modules_11_modules_attn_scale_attn
        )
        attn_weights_66 = (
            l_self_modules_transformer_modules_h_modules_11_modules_attn_scale_attn
        ) = None
        causal_mask_16 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 36, None),
            )
        ]
        causal_mask_4 = None
        attn_weights_68 = attn_weights_67 + causal_mask_16
        attn_weights_67 = causal_mask_16 = None
        attn_weights_69 = torch.nn.functional.softmax(attn_weights_68, dim=-1)
        attn_weights_68 = None
        attn_weights_70 = attn_weights_69.to(torch.float16)
        attn_weights_69 = None
        attn_weights_71 = torch.nn.functional.dropout(
            attn_weights_70, 0.0, False, False
        )
        attn_weights_70 = None
        attn_output_44 = torch.matmul(attn_weights_71, value_23)
        attn_weights_71 = value_23 = None
        permute_47 = attn_output_44.permute(0, 2, 1, 3)
        attn_output_44 = None
        tensor_47 = permute_47.contiguous()
        permute_47 = None
        attn_output_45 = tensor_47.view((1, 36, 2048))
        tensor_47 = None
        attn_output_46 = torch._C._nn.linear(
            attn_output_45,
            l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_out_proj_parameters_weight_,
            None,
        )
        attn_output_45 = l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_out_proj_parameters_weight_ = (None)
        attn_output_47 = torch.nn.functional.dropout(attn_output_46, 0.0, False, False)
        attn_output_46 = None
        hidden_states_68 = torch._C._nn.linear(
            hidden_states_67,
            l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_fc_in_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_fc_in_parameters_bias_,
        )
        hidden_states_67 = l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_fc_in_parameters_weight_ = l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_fc_in_parameters_bias_ = (None)
        mul_92 = 0.5 * hidden_states_68
        pow_12 = torch.pow(hidden_states_68, 3.0)
        mul_93 = 0.044715 * pow_12
        pow_12 = None
        add_81 = hidden_states_68 + mul_93
        hidden_states_68 = mul_93 = None
        mul_94 = 0.7978845608028654 * add_81
        add_81 = None
        tanh_11 = torch.tanh(mul_94)
        mul_94 = None
        add_82 = 1.0 + tanh_11
        tanh_11 = None
        hidden_states_69 = mul_92 * add_82
        mul_92 = add_82 = None
        hidden_states_70 = torch._C._nn.linear(
            hidden_states_69,
            l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_fc_out_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_fc_out_parameters_bias_,
        )
        hidden_states_69 = l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_fc_out_parameters_weight_ = l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_fc_out_parameters_bias_ = (None)
        hidden_states_71 = torch.nn.functional.dropout(
            hidden_states_70, 0.0, False, False
        )
        hidden_states_70 = None
        add_83 = attn_output_47 + hidden_states_71
        attn_output_47 = hidden_states_71 = None
        hidden_states_72 = add_83 + hidden_states_66
        add_83 = hidden_states_66 = None
        hidden_states_73 = torch.nn.functional.layer_norm(
            hidden_states_72,
            (2048,),
            l_self_modules_transformer_modules_ln_f_parameters_weight_,
            l_self_modules_transformer_modules_ln_f_parameters_bias_,
            1e-06,
        )
        hidden_states_72 = (
            l_self_modules_transformer_modules_ln_f_parameters_weight_
        ) = l_self_modules_transformer_modules_ln_f_parameters_bias_ = None
        hidden_states_74 = hidden_states_73.view((-1, 36, 2048))
        hidden_states_73 = None
        linear_72 = torch._C._nn.linear(
            hidden_states_74,
            l_self_modules_lm_head_parameters_weight_,
            l_self_modules_lm_head_parameters_bias_,
        )
        hidden_states_74 = (
            l_self_modules_lm_head_parameters_weight_
        ) = l_self_modules_lm_head_parameters_bias_ = None
        lm_logits = linear_72.to(torch.float32)
        linear_72 = None
        return (lm_logits,)
