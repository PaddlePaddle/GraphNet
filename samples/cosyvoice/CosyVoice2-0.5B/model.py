import torch

from torch import device

from torch import inf


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_stack0_: torch.Tensor,
        L_xs_: torch.Tensor,
        L_self_modules_up_encoders_modules_0_modules_norm_mha_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_0_modules_norm_mha_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_pos_emb_: torch.Tensor,
        L_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_pos_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_0_modules_self_attn_parameters_pos_bias_u_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_0_modules_self_attn_parameters_pos_bias_v_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_0_modules_norm_ff_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_0_modules_norm_ff_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_0_modules_feed_forward_modules_w_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_0_modules_feed_forward_modules_w_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_0_modules_feed_forward_modules_w_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_0_modules_feed_forward_modules_w_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_1_modules_norm_mha_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_1_modules_norm_mha_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_pos_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_1_modules_self_attn_parameters_pos_bias_u_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_1_modules_self_attn_parameters_pos_bias_v_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_1_modules_norm_ff_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_1_modules_norm_ff_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_1_modules_feed_forward_modules_w_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_1_modules_feed_forward_modules_w_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_1_modules_feed_forward_modules_w_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_1_modules_feed_forward_modules_w_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_2_modules_norm_mha_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_2_modules_norm_mha_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_pos_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_2_modules_self_attn_parameters_pos_bias_u_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_2_modules_self_attn_parameters_pos_bias_v_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_2_modules_norm_ff_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_2_modules_norm_ff_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_2_modules_feed_forward_modules_w_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_2_modules_feed_forward_modules_w_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_2_modules_feed_forward_modules_w_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_2_modules_feed_forward_modules_w_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_3_modules_norm_mha_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_3_modules_norm_mha_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_pos_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_3_modules_self_attn_parameters_pos_bias_u_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_3_modules_self_attn_parameters_pos_bias_v_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_3_modules_norm_ff_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_3_modules_norm_ff_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_3_modules_feed_forward_modules_w_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_3_modules_feed_forward_modules_w_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_3_modules_feed_forward_modules_w_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_up_encoders_modules_3_modules_feed_forward_modules_w_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_after_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_after_norm_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_stack0_ = L_stack0_
        l_xs_ = L_xs_
        l_self_modules_up_encoders_modules_0_modules_norm_mha_parameters_weight_ = (
            L_self_modules_up_encoders_modules_0_modules_norm_mha_parameters_weight_
        )
        l_self_modules_up_encoders_modules_0_modules_norm_mha_parameters_bias_ = (
            L_self_modules_up_encoders_modules_0_modules_norm_mha_parameters_bias_
        )
        l_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_q_parameters_weight_ = L_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_q_parameters_weight_
        l_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_q_parameters_bias_ = L_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_q_parameters_bias_
        l_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_k_parameters_weight_ = L_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_k_parameters_weight_
        l_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_k_parameters_bias_ = L_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_k_parameters_bias_
        l_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_v_parameters_weight_ = L_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_v_parameters_weight_
        l_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_v_parameters_bias_ = L_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_v_parameters_bias_
        l_pos_emb_ = L_pos_emb_
        l_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_pos_parameters_weight_ = L_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_pos_parameters_weight_
        l_self_modules_up_encoders_modules_0_modules_self_attn_parameters_pos_bias_u_ = L_self_modules_up_encoders_modules_0_modules_self_attn_parameters_pos_bias_u_
        l_self_modules_up_encoders_modules_0_modules_self_attn_parameters_pos_bias_v_ = L_self_modules_up_encoders_modules_0_modules_self_attn_parameters_pos_bias_v_
        l_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_out_parameters_weight_ = L_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_out_parameters_weight_
        l_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_out_parameters_bias_ = L_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_out_parameters_bias_
        l_self_modules_up_encoders_modules_0_modules_norm_ff_parameters_weight_ = (
            L_self_modules_up_encoders_modules_0_modules_norm_ff_parameters_weight_
        )
        l_self_modules_up_encoders_modules_0_modules_norm_ff_parameters_bias_ = (
            L_self_modules_up_encoders_modules_0_modules_norm_ff_parameters_bias_
        )
        l_self_modules_up_encoders_modules_0_modules_feed_forward_modules_w_1_parameters_weight_ = L_self_modules_up_encoders_modules_0_modules_feed_forward_modules_w_1_parameters_weight_
        l_self_modules_up_encoders_modules_0_modules_feed_forward_modules_w_1_parameters_bias_ = L_self_modules_up_encoders_modules_0_modules_feed_forward_modules_w_1_parameters_bias_
        l_self_modules_up_encoders_modules_0_modules_feed_forward_modules_w_2_parameters_weight_ = L_self_modules_up_encoders_modules_0_modules_feed_forward_modules_w_2_parameters_weight_
        l_self_modules_up_encoders_modules_0_modules_feed_forward_modules_w_2_parameters_bias_ = L_self_modules_up_encoders_modules_0_modules_feed_forward_modules_w_2_parameters_bias_
        l_self_modules_up_encoders_modules_1_modules_norm_mha_parameters_weight_ = (
            L_self_modules_up_encoders_modules_1_modules_norm_mha_parameters_weight_
        )
        l_self_modules_up_encoders_modules_1_modules_norm_mha_parameters_bias_ = (
            L_self_modules_up_encoders_modules_1_modules_norm_mha_parameters_bias_
        )
        l_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_q_parameters_weight_ = L_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_q_parameters_weight_
        l_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_q_parameters_bias_ = L_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_q_parameters_bias_
        l_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_k_parameters_weight_ = L_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_k_parameters_weight_
        l_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_k_parameters_bias_ = L_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_k_parameters_bias_
        l_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_v_parameters_weight_ = L_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_v_parameters_weight_
        l_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_v_parameters_bias_ = L_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_v_parameters_bias_
        l_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_pos_parameters_weight_ = L_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_pos_parameters_weight_
        l_self_modules_up_encoders_modules_1_modules_self_attn_parameters_pos_bias_u_ = L_self_modules_up_encoders_modules_1_modules_self_attn_parameters_pos_bias_u_
        l_self_modules_up_encoders_modules_1_modules_self_attn_parameters_pos_bias_v_ = L_self_modules_up_encoders_modules_1_modules_self_attn_parameters_pos_bias_v_
        l_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_out_parameters_weight_ = L_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_out_parameters_weight_
        l_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_out_parameters_bias_ = L_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_out_parameters_bias_
        l_self_modules_up_encoders_modules_1_modules_norm_ff_parameters_weight_ = (
            L_self_modules_up_encoders_modules_1_modules_norm_ff_parameters_weight_
        )
        l_self_modules_up_encoders_modules_1_modules_norm_ff_parameters_bias_ = (
            L_self_modules_up_encoders_modules_1_modules_norm_ff_parameters_bias_
        )
        l_self_modules_up_encoders_modules_1_modules_feed_forward_modules_w_1_parameters_weight_ = L_self_modules_up_encoders_modules_1_modules_feed_forward_modules_w_1_parameters_weight_
        l_self_modules_up_encoders_modules_1_modules_feed_forward_modules_w_1_parameters_bias_ = L_self_modules_up_encoders_modules_1_modules_feed_forward_modules_w_1_parameters_bias_
        l_self_modules_up_encoders_modules_1_modules_feed_forward_modules_w_2_parameters_weight_ = L_self_modules_up_encoders_modules_1_modules_feed_forward_modules_w_2_parameters_weight_
        l_self_modules_up_encoders_modules_1_modules_feed_forward_modules_w_2_parameters_bias_ = L_self_modules_up_encoders_modules_1_modules_feed_forward_modules_w_2_parameters_bias_
        l_self_modules_up_encoders_modules_2_modules_norm_mha_parameters_weight_ = (
            L_self_modules_up_encoders_modules_2_modules_norm_mha_parameters_weight_
        )
        l_self_modules_up_encoders_modules_2_modules_norm_mha_parameters_bias_ = (
            L_self_modules_up_encoders_modules_2_modules_norm_mha_parameters_bias_
        )
        l_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_q_parameters_weight_ = L_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_q_parameters_weight_
        l_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_q_parameters_bias_ = L_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_q_parameters_bias_
        l_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_k_parameters_weight_ = L_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_k_parameters_weight_
        l_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_k_parameters_bias_ = L_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_k_parameters_bias_
        l_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_v_parameters_weight_ = L_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_v_parameters_weight_
        l_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_v_parameters_bias_ = L_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_v_parameters_bias_
        l_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_pos_parameters_weight_ = L_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_pos_parameters_weight_
        l_self_modules_up_encoders_modules_2_modules_self_attn_parameters_pos_bias_u_ = L_self_modules_up_encoders_modules_2_modules_self_attn_parameters_pos_bias_u_
        l_self_modules_up_encoders_modules_2_modules_self_attn_parameters_pos_bias_v_ = L_self_modules_up_encoders_modules_2_modules_self_attn_parameters_pos_bias_v_
        l_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_out_parameters_weight_ = L_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_out_parameters_weight_
        l_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_out_parameters_bias_ = L_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_out_parameters_bias_
        l_self_modules_up_encoders_modules_2_modules_norm_ff_parameters_weight_ = (
            L_self_modules_up_encoders_modules_2_modules_norm_ff_parameters_weight_
        )
        l_self_modules_up_encoders_modules_2_modules_norm_ff_parameters_bias_ = (
            L_self_modules_up_encoders_modules_2_modules_norm_ff_parameters_bias_
        )
        l_self_modules_up_encoders_modules_2_modules_feed_forward_modules_w_1_parameters_weight_ = L_self_modules_up_encoders_modules_2_modules_feed_forward_modules_w_1_parameters_weight_
        l_self_modules_up_encoders_modules_2_modules_feed_forward_modules_w_1_parameters_bias_ = L_self_modules_up_encoders_modules_2_modules_feed_forward_modules_w_1_parameters_bias_
        l_self_modules_up_encoders_modules_2_modules_feed_forward_modules_w_2_parameters_weight_ = L_self_modules_up_encoders_modules_2_modules_feed_forward_modules_w_2_parameters_weight_
        l_self_modules_up_encoders_modules_2_modules_feed_forward_modules_w_2_parameters_bias_ = L_self_modules_up_encoders_modules_2_modules_feed_forward_modules_w_2_parameters_bias_
        l_self_modules_up_encoders_modules_3_modules_norm_mha_parameters_weight_ = (
            L_self_modules_up_encoders_modules_3_modules_norm_mha_parameters_weight_
        )
        l_self_modules_up_encoders_modules_3_modules_norm_mha_parameters_bias_ = (
            L_self_modules_up_encoders_modules_3_modules_norm_mha_parameters_bias_
        )
        l_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_q_parameters_weight_ = L_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_q_parameters_weight_
        l_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_q_parameters_bias_ = L_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_q_parameters_bias_
        l_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_k_parameters_weight_ = L_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_k_parameters_weight_
        l_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_k_parameters_bias_ = L_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_k_parameters_bias_
        l_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_v_parameters_weight_ = L_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_v_parameters_weight_
        l_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_v_parameters_bias_ = L_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_v_parameters_bias_
        l_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_pos_parameters_weight_ = L_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_pos_parameters_weight_
        l_self_modules_up_encoders_modules_3_modules_self_attn_parameters_pos_bias_u_ = L_self_modules_up_encoders_modules_3_modules_self_attn_parameters_pos_bias_u_
        l_self_modules_up_encoders_modules_3_modules_self_attn_parameters_pos_bias_v_ = L_self_modules_up_encoders_modules_3_modules_self_attn_parameters_pos_bias_v_
        l_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_out_parameters_weight_ = L_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_out_parameters_weight_
        l_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_out_parameters_bias_ = L_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_out_parameters_bias_
        l_self_modules_up_encoders_modules_3_modules_norm_ff_parameters_weight_ = (
            L_self_modules_up_encoders_modules_3_modules_norm_ff_parameters_weight_
        )
        l_self_modules_up_encoders_modules_3_modules_norm_ff_parameters_bias_ = (
            L_self_modules_up_encoders_modules_3_modules_norm_ff_parameters_bias_
        )
        l_self_modules_up_encoders_modules_3_modules_feed_forward_modules_w_1_parameters_weight_ = L_self_modules_up_encoders_modules_3_modules_feed_forward_modules_w_1_parameters_weight_
        l_self_modules_up_encoders_modules_3_modules_feed_forward_modules_w_1_parameters_bias_ = L_self_modules_up_encoders_modules_3_modules_feed_forward_modules_w_1_parameters_bias_
        l_self_modules_up_encoders_modules_3_modules_feed_forward_modules_w_2_parameters_weight_ = L_self_modules_up_encoders_modules_3_modules_feed_forward_modules_w_2_parameters_weight_
        l_self_modules_up_encoders_modules_3_modules_feed_forward_modules_w_2_parameters_bias_ = L_self_modules_up_encoders_modules_3_modules_feed_forward_modules_w_2_parameters_bias_
        l_self_modules_after_norm_parameters_weight_ = (
            L_self_modules_after_norm_parameters_weight_
        )
        l_self_modules_after_norm_parameters_bias_ = (
            L_self_modules_after_norm_parameters_bias_
        )
        x = torch.nn.functional.layer_norm(
            l_xs_,
            (512,),
            l_self_modules_up_encoders_modules_0_modules_norm_mha_parameters_weight_,
            l_self_modules_up_encoders_modules_0_modules_norm_mha_parameters_bias_,
            1e-12,
        )
        l_self_modules_up_encoders_modules_0_modules_norm_mha_parameters_weight_ = (
            l_self_modules_up_encoders_modules_0_modules_norm_mha_parameters_bias_
        ) = None
        linear = torch._C._nn.linear(
            x,
            l_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_q_parameters_weight_,
            l_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_q_parameters_weight_ = l_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q = linear.view(1, -1, 8, 64)
        linear = None
        linear_1 = torch._C._nn.linear(
            x,
            l_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_k_parameters_weight_,
            l_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_k_parameters_weight_ = l_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k = linear_1.view(1, -1, 8, 64)
        linear_1 = None
        linear_2 = torch._C._nn.linear(
            x,
            l_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_v_parameters_weight_,
            l_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x = l_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_v_parameters_weight_ = l_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v = linear_2.view(1, -1, 8, 64)
        linear_2 = None
        q_1 = q.transpose(1, 2)
        q = None
        k_1 = k.transpose(1, 2)
        k = None
        v_1 = v.transpose(1, 2)
        v = None
        q_2 = q_1.transpose(1, 2)
        q_1 = None
        new_cache = torch.cat((k_1, v_1), dim=-1)
        new_cache = None
        linear_3 = torch._C._nn.linear(
            l_pos_emb_,
            l_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p = linear_3.view(1, -1, 8, 64)
        linear_3 = None
        p_1 = p.transpose(1, 2)
        p = None
        add = (
            q_2
            + l_self_modules_up_encoders_modules_0_modules_self_attn_parameters_pos_bias_u_
        )
        l_self_modules_up_encoders_modules_0_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u = add.transpose(1, 2)
        add = None
        add_1 = (
            q_2
            + l_self_modules_up_encoders_modules_0_modules_self_attn_parameters_pos_bias_v_
        )
        q_2 = l_self_modules_up_encoders_modules_0_modules_self_attn_parameters_pos_bias_v_ = (None)
        q_with_bias_v = add_1.transpose(1, 2)
        add_1 = None
        transpose_7 = k_1.transpose(-2, -1)
        k_1 = None
        matrix_ac = torch.matmul(q_with_bias_u, transpose_7)
        q_with_bias_u = transpose_7 = None
        transpose_8 = p_1.transpose(-2, -1)
        p_1 = None
        matrix_bd = torch.matmul(q_with_bias_v, transpose_8)
        q_with_bias_v = transpose_8 = None
        zero_pad = torch.zeros(
            (1, 8, 796, 1), device=device(type="cuda", index=0), dtype=torch.float32
        )
        x_padded = torch.cat([zero_pad, matrix_bd], dim=-1)
        zero_pad = None
        x_padded_1 = x_padded.view(1, 8, 1592, 796)
        x_padded = None
        getitem = x_padded_1[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_padded_1 = None
        view_as = getitem.view_as(matrix_bd)
        getitem = matrix_bd = None
        x_1 = view_as[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 796, None),
            )
        ]
        view_as = None
        add_2 = matrix_ac + x_1
        matrix_ac = x_1 = None
        scores = add_2 / 8.0
        add_2 = None
        unsqueeze = l_stack0_.unsqueeze(1)
        mask = unsqueeze.eq(0)
        unsqueeze = None
        mask_1 = mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 796, None),
            )
        ]
        mask = None
        scores_1 = scores.masked_fill(mask_1, -inf)
        scores = None
        softmax = torch.softmax(scores_1, dim=-1)
        scores_1 = None
        attn = softmax.masked_fill(mask_1, 0.0)
        softmax = mask_1 = None
        p_attn = torch.nn.functional.dropout(attn, 0.1, False, False)
        attn = None
        x_2 = torch.matmul(p_attn, v_1)
        p_attn = v_1 = None
        transpose_9 = x_2.transpose(1, 2)
        x_2 = None
        contiguous = transpose_9.contiguous()
        transpose_9 = None
        x_3 = contiguous.view(1, -1, 512)
        contiguous = None
        x_att = torch._C._nn.linear(
            x_3,
            l_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_out_parameters_weight_,
            l_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_3 = l_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_out_parameters_weight_ = l_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_1 = torch.nn.functional.dropout(x_att, 0.1, False, False)
        x_att = None
        x_4 = l_xs_ + dropout_1
        l_xs_ = dropout_1 = None
        new_cnn_cache = torch.zeros(
            (0, 0, 0), dtype=torch.float32, device=device(type="cuda", index=0)
        )
        new_cnn_cache = None
        x_5 = torch.nn.functional.layer_norm(
            x_4,
            (512,),
            l_self_modules_up_encoders_modules_0_modules_norm_ff_parameters_weight_,
            l_self_modules_up_encoders_modules_0_modules_norm_ff_parameters_bias_,
            1e-12,
        )
        l_self_modules_up_encoders_modules_0_modules_norm_ff_parameters_weight_ = (
            l_self_modules_up_encoders_modules_0_modules_norm_ff_parameters_bias_
        ) = None
        linear_5 = torch._C._nn.linear(
            x_5,
            l_self_modules_up_encoders_modules_0_modules_feed_forward_modules_w_1_parameters_weight_,
            l_self_modules_up_encoders_modules_0_modules_feed_forward_modules_w_1_parameters_bias_,
        )
        x_5 = l_self_modules_up_encoders_modules_0_modules_feed_forward_modules_w_1_parameters_weight_ = l_self_modules_up_encoders_modules_0_modules_feed_forward_modules_w_1_parameters_bias_ = (None)
        silu = torch.nn.functional.silu(linear_5, inplace=False)
        linear_5 = None
        dropout_2 = torch.nn.functional.dropout(silu, 0.1, False, False)
        silu = None
        linear_6 = torch._C._nn.linear(
            dropout_2,
            l_self_modules_up_encoders_modules_0_modules_feed_forward_modules_w_2_parameters_weight_,
            l_self_modules_up_encoders_modules_0_modules_feed_forward_modules_w_2_parameters_bias_,
        )
        dropout_2 = l_self_modules_up_encoders_modules_0_modules_feed_forward_modules_w_2_parameters_weight_ = l_self_modules_up_encoders_modules_0_modules_feed_forward_modules_w_2_parameters_bias_ = (None)
        dropout_3 = torch.nn.functional.dropout(linear_6, 0.1, False, False)
        linear_6 = None
        mul = 1.0 * dropout_3
        dropout_3 = None
        x_6 = x_4 + mul
        x_4 = mul = None
        x_7 = torch.nn.functional.layer_norm(
            x_6,
            (512,),
            l_self_modules_up_encoders_modules_1_modules_norm_mha_parameters_weight_,
            l_self_modules_up_encoders_modules_1_modules_norm_mha_parameters_bias_,
            1e-12,
        )
        l_self_modules_up_encoders_modules_1_modules_norm_mha_parameters_weight_ = (
            l_self_modules_up_encoders_modules_1_modules_norm_mha_parameters_bias_
        ) = None
        linear_7 = torch._C._nn.linear(
            x_7,
            l_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_q_parameters_weight_,
            l_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_q_parameters_weight_ = l_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_3 = linear_7.view(1, -1, 8, 64)
        linear_7 = None
        linear_8 = torch._C._nn.linear(
            x_7,
            l_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_k_parameters_weight_,
            l_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_k_parameters_weight_ = l_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_2 = linear_8.view(1, -1, 8, 64)
        linear_8 = None
        linear_9 = torch._C._nn.linear(
            x_7,
            l_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_v_parameters_weight_,
            l_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_7 = l_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_v_parameters_weight_ = l_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_2 = linear_9.view(1, -1, 8, 64)
        linear_9 = None
        q_4 = q_3.transpose(1, 2)
        q_3 = None
        k_3 = k_2.transpose(1, 2)
        k_2 = None
        v_3 = v_2.transpose(1, 2)
        v_2 = None
        q_5 = q_4.transpose(1, 2)
        q_4 = None
        new_cache_1 = torch.cat((k_3, v_3), dim=-1)
        new_cache_1 = None
        linear_10 = torch._C._nn.linear(
            l_pos_emb_,
            l_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_2 = linear_10.view(1, -1, 8, 64)
        linear_10 = None
        p_3 = p_2.transpose(1, 2)
        p_2 = None
        add_5 = (
            q_5
            + l_self_modules_up_encoders_modules_1_modules_self_attn_parameters_pos_bias_u_
        )
        l_self_modules_up_encoders_modules_1_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_1 = add_5.transpose(1, 2)
        add_5 = None
        add_6 = (
            q_5
            + l_self_modules_up_encoders_modules_1_modules_self_attn_parameters_pos_bias_v_
        )
        q_5 = l_self_modules_up_encoders_modules_1_modules_self_attn_parameters_pos_bias_v_ = (None)
        q_with_bias_v_1 = add_6.transpose(1, 2)
        add_6 = None
        transpose_17 = k_3.transpose(-2, -1)
        k_3 = None
        matrix_ac_1 = torch.matmul(q_with_bias_u_1, transpose_17)
        q_with_bias_u_1 = transpose_17 = None
        transpose_18 = p_3.transpose(-2, -1)
        p_3 = None
        matrix_bd_1 = torch.matmul(q_with_bias_v_1, transpose_18)
        q_with_bias_v_1 = transpose_18 = None
        zero_pad_1 = torch.zeros(
            (1, 8, 796, 1), device=device(type="cuda", index=0), dtype=torch.float32
        )
        x_padded_2 = torch.cat([zero_pad_1, matrix_bd_1], dim=-1)
        zero_pad_1 = None
        x_padded_3 = x_padded_2.view(1, 8, 1592, 796)
        x_padded_2 = None
        getitem_3 = x_padded_3[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_padded_3 = None
        view_as_1 = getitem_3.view_as(matrix_bd_1)
        getitem_3 = matrix_bd_1 = None
        x_8 = view_as_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 796, None),
            )
        ]
        view_as_1 = None
        add_7 = matrix_ac_1 + x_8
        matrix_ac_1 = x_8 = None
        scores_2 = add_7 / 8.0
        add_7 = None
        unsqueeze_1 = l_stack0_.unsqueeze(1)
        mask_2 = unsqueeze_1.eq(0)
        unsqueeze_1 = None
        mask_3 = mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 796, None),
            )
        ]
        mask_2 = None
        scores_3 = scores_2.masked_fill(mask_3, -inf)
        scores_2 = None
        softmax_1 = torch.softmax(scores_3, dim=-1)
        scores_3 = None
        attn_1 = softmax_1.masked_fill(mask_3, 0.0)
        softmax_1 = mask_3 = None
        p_attn_1 = torch.nn.functional.dropout(attn_1, 0.1, False, False)
        attn_1 = None
        x_9 = torch.matmul(p_attn_1, v_3)
        p_attn_1 = v_3 = None
        transpose_19 = x_9.transpose(1, 2)
        x_9 = None
        contiguous_1 = transpose_19.contiguous()
        transpose_19 = None
        x_10 = contiguous_1.view(1, -1, 512)
        contiguous_1 = None
        x_att_1 = torch._C._nn.linear(
            x_10,
            l_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_out_parameters_weight_,
            l_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_10 = l_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_out_parameters_weight_ = l_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_5 = torch.nn.functional.dropout(x_att_1, 0.1, False, False)
        x_att_1 = None
        x_11 = x_6 + dropout_5
        x_6 = dropout_5 = None
        new_cnn_cache_1 = torch.zeros(
            (0, 0, 0), dtype=torch.float32, device=device(type="cuda", index=0)
        )
        new_cnn_cache_1 = None
        x_12 = torch.nn.functional.layer_norm(
            x_11,
            (512,),
            l_self_modules_up_encoders_modules_1_modules_norm_ff_parameters_weight_,
            l_self_modules_up_encoders_modules_1_modules_norm_ff_parameters_bias_,
            1e-12,
        )
        l_self_modules_up_encoders_modules_1_modules_norm_ff_parameters_weight_ = (
            l_self_modules_up_encoders_modules_1_modules_norm_ff_parameters_bias_
        ) = None
        linear_12 = torch._C._nn.linear(
            x_12,
            l_self_modules_up_encoders_modules_1_modules_feed_forward_modules_w_1_parameters_weight_,
            l_self_modules_up_encoders_modules_1_modules_feed_forward_modules_w_1_parameters_bias_,
        )
        x_12 = l_self_modules_up_encoders_modules_1_modules_feed_forward_modules_w_1_parameters_weight_ = l_self_modules_up_encoders_modules_1_modules_feed_forward_modules_w_1_parameters_bias_ = (None)
        silu_1 = torch.nn.functional.silu(linear_12, inplace=False)
        linear_12 = None
        dropout_6 = torch.nn.functional.dropout(silu_1, 0.1, False, False)
        silu_1 = None
        linear_13 = torch._C._nn.linear(
            dropout_6,
            l_self_modules_up_encoders_modules_1_modules_feed_forward_modules_w_2_parameters_weight_,
            l_self_modules_up_encoders_modules_1_modules_feed_forward_modules_w_2_parameters_bias_,
        )
        dropout_6 = l_self_modules_up_encoders_modules_1_modules_feed_forward_modules_w_2_parameters_weight_ = l_self_modules_up_encoders_modules_1_modules_feed_forward_modules_w_2_parameters_bias_ = (None)
        dropout_7 = torch.nn.functional.dropout(linear_13, 0.1, False, False)
        linear_13 = None
        mul_1 = 1.0 * dropout_7
        dropout_7 = None
        x_13 = x_11 + mul_1
        x_11 = mul_1 = None
        x_14 = torch.nn.functional.layer_norm(
            x_13,
            (512,),
            l_self_modules_up_encoders_modules_2_modules_norm_mha_parameters_weight_,
            l_self_modules_up_encoders_modules_2_modules_norm_mha_parameters_bias_,
            1e-12,
        )
        l_self_modules_up_encoders_modules_2_modules_norm_mha_parameters_weight_ = (
            l_self_modules_up_encoders_modules_2_modules_norm_mha_parameters_bias_
        ) = None
        linear_14 = torch._C._nn.linear(
            x_14,
            l_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_q_parameters_weight_,
            l_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_q_parameters_weight_ = l_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_6 = linear_14.view(1, -1, 8, 64)
        linear_14 = None
        linear_15 = torch._C._nn.linear(
            x_14,
            l_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_k_parameters_weight_,
            l_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_k_parameters_weight_ = l_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_4 = linear_15.view(1, -1, 8, 64)
        linear_15 = None
        linear_16 = torch._C._nn.linear(
            x_14,
            l_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_v_parameters_weight_,
            l_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_14 = l_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_v_parameters_weight_ = l_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_4 = linear_16.view(1, -1, 8, 64)
        linear_16 = None
        q_7 = q_6.transpose(1, 2)
        q_6 = None
        k_5 = k_4.transpose(1, 2)
        k_4 = None
        v_5 = v_4.transpose(1, 2)
        v_4 = None
        q_8 = q_7.transpose(1, 2)
        q_7 = None
        new_cache_2 = torch.cat((k_5, v_5), dim=-1)
        new_cache_2 = None
        linear_17 = torch._C._nn.linear(
            l_pos_emb_,
            l_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_4 = linear_17.view(1, -1, 8, 64)
        linear_17 = None
        p_5 = p_4.transpose(1, 2)
        p_4 = None
        add_10 = (
            q_8
            + l_self_modules_up_encoders_modules_2_modules_self_attn_parameters_pos_bias_u_
        )
        l_self_modules_up_encoders_modules_2_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_2 = add_10.transpose(1, 2)
        add_10 = None
        add_11 = (
            q_8
            + l_self_modules_up_encoders_modules_2_modules_self_attn_parameters_pos_bias_v_
        )
        q_8 = l_self_modules_up_encoders_modules_2_modules_self_attn_parameters_pos_bias_v_ = (None)
        q_with_bias_v_2 = add_11.transpose(1, 2)
        add_11 = None
        transpose_27 = k_5.transpose(-2, -1)
        k_5 = None
        matrix_ac_2 = torch.matmul(q_with_bias_u_2, transpose_27)
        q_with_bias_u_2 = transpose_27 = None
        transpose_28 = p_5.transpose(-2, -1)
        p_5 = None
        matrix_bd_2 = torch.matmul(q_with_bias_v_2, transpose_28)
        q_with_bias_v_2 = transpose_28 = None
        zero_pad_2 = torch.zeros(
            (1, 8, 796, 1), device=device(type="cuda", index=0), dtype=torch.float32
        )
        x_padded_4 = torch.cat([zero_pad_2, matrix_bd_2], dim=-1)
        zero_pad_2 = None
        x_padded_5 = x_padded_4.view(1, 8, 1592, 796)
        x_padded_4 = None
        getitem_6 = x_padded_5[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_padded_5 = None
        view_as_2 = getitem_6.view_as(matrix_bd_2)
        getitem_6 = matrix_bd_2 = None
        x_15 = view_as_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 796, None),
            )
        ]
        view_as_2 = None
        add_12 = matrix_ac_2 + x_15
        matrix_ac_2 = x_15 = None
        scores_4 = add_12 / 8.0
        add_12 = None
        unsqueeze_2 = l_stack0_.unsqueeze(1)
        mask_4 = unsqueeze_2.eq(0)
        unsqueeze_2 = None
        mask_5 = mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 796, None),
            )
        ]
        mask_4 = None
        scores_5 = scores_4.masked_fill(mask_5, -inf)
        scores_4 = None
        softmax_2 = torch.softmax(scores_5, dim=-1)
        scores_5 = None
        attn_2 = softmax_2.masked_fill(mask_5, 0.0)
        softmax_2 = mask_5 = None
        p_attn_2 = torch.nn.functional.dropout(attn_2, 0.1, False, False)
        attn_2 = None
        x_16 = torch.matmul(p_attn_2, v_5)
        p_attn_2 = v_5 = None
        transpose_29 = x_16.transpose(1, 2)
        x_16 = None
        contiguous_2 = transpose_29.contiguous()
        transpose_29 = None
        x_17 = contiguous_2.view(1, -1, 512)
        contiguous_2 = None
        x_att_2 = torch._C._nn.linear(
            x_17,
            l_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_out_parameters_weight_,
            l_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_17 = l_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_out_parameters_weight_ = l_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_9 = torch.nn.functional.dropout(x_att_2, 0.1, False, False)
        x_att_2 = None
        x_18 = x_13 + dropout_9
        x_13 = dropout_9 = None
        new_cnn_cache_2 = torch.zeros(
            (0, 0, 0), dtype=torch.float32, device=device(type="cuda", index=0)
        )
        new_cnn_cache_2 = None
        x_19 = torch.nn.functional.layer_norm(
            x_18,
            (512,),
            l_self_modules_up_encoders_modules_2_modules_norm_ff_parameters_weight_,
            l_self_modules_up_encoders_modules_2_modules_norm_ff_parameters_bias_,
            1e-12,
        )
        l_self_modules_up_encoders_modules_2_modules_norm_ff_parameters_weight_ = (
            l_self_modules_up_encoders_modules_2_modules_norm_ff_parameters_bias_
        ) = None
        linear_19 = torch._C._nn.linear(
            x_19,
            l_self_modules_up_encoders_modules_2_modules_feed_forward_modules_w_1_parameters_weight_,
            l_self_modules_up_encoders_modules_2_modules_feed_forward_modules_w_1_parameters_bias_,
        )
        x_19 = l_self_modules_up_encoders_modules_2_modules_feed_forward_modules_w_1_parameters_weight_ = l_self_modules_up_encoders_modules_2_modules_feed_forward_modules_w_1_parameters_bias_ = (None)
        silu_2 = torch.nn.functional.silu(linear_19, inplace=False)
        linear_19 = None
        dropout_10 = torch.nn.functional.dropout(silu_2, 0.1, False, False)
        silu_2 = None
        linear_20 = torch._C._nn.linear(
            dropout_10,
            l_self_modules_up_encoders_modules_2_modules_feed_forward_modules_w_2_parameters_weight_,
            l_self_modules_up_encoders_modules_2_modules_feed_forward_modules_w_2_parameters_bias_,
        )
        dropout_10 = l_self_modules_up_encoders_modules_2_modules_feed_forward_modules_w_2_parameters_weight_ = l_self_modules_up_encoders_modules_2_modules_feed_forward_modules_w_2_parameters_bias_ = (None)
        dropout_11 = torch.nn.functional.dropout(linear_20, 0.1, False, False)
        linear_20 = None
        mul_2 = 1.0 * dropout_11
        dropout_11 = None
        x_20 = x_18 + mul_2
        x_18 = mul_2 = None
        x_21 = torch.nn.functional.layer_norm(
            x_20,
            (512,),
            l_self_modules_up_encoders_modules_3_modules_norm_mha_parameters_weight_,
            l_self_modules_up_encoders_modules_3_modules_norm_mha_parameters_bias_,
            1e-12,
        )
        l_self_modules_up_encoders_modules_3_modules_norm_mha_parameters_weight_ = (
            l_self_modules_up_encoders_modules_3_modules_norm_mha_parameters_bias_
        ) = None
        linear_21 = torch._C._nn.linear(
            x_21,
            l_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_q_parameters_weight_,
            l_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_q_parameters_weight_ = l_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_9 = linear_21.view(1, -1, 8, 64)
        linear_21 = None
        linear_22 = torch._C._nn.linear(
            x_21,
            l_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_k_parameters_weight_,
            l_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_k_parameters_weight_ = l_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_6 = linear_22.view(1, -1, 8, 64)
        linear_22 = None
        linear_23 = torch._C._nn.linear(
            x_21,
            l_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_v_parameters_weight_,
            l_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_21 = l_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_v_parameters_weight_ = l_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_6 = linear_23.view(1, -1, 8, 64)
        linear_23 = None
        q_10 = q_9.transpose(1, 2)
        q_9 = None
        k_7 = k_6.transpose(1, 2)
        k_6 = None
        v_7 = v_6.transpose(1, 2)
        v_6 = None
        q_11 = q_10.transpose(1, 2)
        q_10 = None
        new_cache_3 = torch.cat((k_7, v_7), dim=-1)
        new_cache_3 = None
        linear_24 = torch._C._nn.linear(
            l_pos_emb_,
            l_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_pos_emb_ = l_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_pos_parameters_weight_ = (None)
        p_6 = linear_24.view(1, -1, 8, 64)
        linear_24 = None
        p_7 = p_6.transpose(1, 2)
        p_6 = None
        add_15 = (
            q_11
            + l_self_modules_up_encoders_modules_3_modules_self_attn_parameters_pos_bias_u_
        )
        l_self_modules_up_encoders_modules_3_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_3 = add_15.transpose(1, 2)
        add_15 = None
        add_16 = (
            q_11
            + l_self_modules_up_encoders_modules_3_modules_self_attn_parameters_pos_bias_v_
        )
        q_11 = l_self_modules_up_encoders_modules_3_modules_self_attn_parameters_pos_bias_v_ = (None)
        q_with_bias_v_3 = add_16.transpose(1, 2)
        add_16 = None
        transpose_37 = k_7.transpose(-2, -1)
        k_7 = None
        matrix_ac_3 = torch.matmul(q_with_bias_u_3, transpose_37)
        q_with_bias_u_3 = transpose_37 = None
        transpose_38 = p_7.transpose(-2, -1)
        p_7 = None
        matrix_bd_3 = torch.matmul(q_with_bias_v_3, transpose_38)
        q_with_bias_v_3 = transpose_38 = None
        zero_pad_3 = torch.zeros(
            (1, 8, 796, 1), device=device(type="cuda", index=0), dtype=torch.float32
        )
        x_padded_6 = torch.cat([zero_pad_3, matrix_bd_3], dim=-1)
        zero_pad_3 = None
        x_padded_7 = x_padded_6.view(1, 8, 1592, 796)
        x_padded_6 = None
        getitem_9 = x_padded_7[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_padded_7 = None
        view_as_3 = getitem_9.view_as(matrix_bd_3)
        getitem_9 = matrix_bd_3 = None
        x_22 = view_as_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 796, None),
            )
        ]
        view_as_3 = None
        add_17 = matrix_ac_3 + x_22
        matrix_ac_3 = x_22 = None
        scores_6 = add_17 / 8.0
        add_17 = None
        unsqueeze_3 = l_stack0_.unsqueeze(1)
        l_stack0_ = None
        mask_6 = unsqueeze_3.eq(0)
        unsqueeze_3 = None
        mask_7 = mask_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 796, None),
            )
        ]
        mask_6 = None
        scores_7 = scores_6.masked_fill(mask_7, -inf)
        scores_6 = None
        softmax_3 = torch.softmax(scores_7, dim=-1)
        scores_7 = None
        attn_3 = softmax_3.masked_fill(mask_7, 0.0)
        softmax_3 = mask_7 = None
        p_attn_3 = torch.nn.functional.dropout(attn_3, 0.1, False, False)
        attn_3 = None
        x_23 = torch.matmul(p_attn_3, v_7)
        p_attn_3 = v_7 = None
        transpose_39 = x_23.transpose(1, 2)
        x_23 = None
        contiguous_3 = transpose_39.contiguous()
        transpose_39 = None
        x_24 = contiguous_3.view(1, -1, 512)
        contiguous_3 = None
        x_att_3 = torch._C._nn.linear(
            x_24,
            l_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_out_parameters_weight_,
            l_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_24 = l_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_out_parameters_weight_ = l_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_13 = torch.nn.functional.dropout(x_att_3, 0.1, False, False)
        x_att_3 = None
        x_25 = x_20 + dropout_13
        x_20 = dropout_13 = None
        new_cnn_cache_3 = torch.zeros(
            (0, 0, 0), dtype=torch.float32, device=device(type="cuda", index=0)
        )
        new_cnn_cache_3 = None
        x_26 = torch.nn.functional.layer_norm(
            x_25,
            (512,),
            l_self_modules_up_encoders_modules_3_modules_norm_ff_parameters_weight_,
            l_self_modules_up_encoders_modules_3_modules_norm_ff_parameters_bias_,
            1e-12,
        )
        l_self_modules_up_encoders_modules_3_modules_norm_ff_parameters_weight_ = (
            l_self_modules_up_encoders_modules_3_modules_norm_ff_parameters_bias_
        ) = None
        linear_26 = torch._C._nn.linear(
            x_26,
            l_self_modules_up_encoders_modules_3_modules_feed_forward_modules_w_1_parameters_weight_,
            l_self_modules_up_encoders_modules_3_modules_feed_forward_modules_w_1_parameters_bias_,
        )
        x_26 = l_self_modules_up_encoders_modules_3_modules_feed_forward_modules_w_1_parameters_weight_ = l_self_modules_up_encoders_modules_3_modules_feed_forward_modules_w_1_parameters_bias_ = (None)
        silu_3 = torch.nn.functional.silu(linear_26, inplace=False)
        linear_26 = None
        dropout_14 = torch.nn.functional.dropout(silu_3, 0.1, False, False)
        silu_3 = None
        linear_27 = torch._C._nn.linear(
            dropout_14,
            l_self_modules_up_encoders_modules_3_modules_feed_forward_modules_w_2_parameters_weight_,
            l_self_modules_up_encoders_modules_3_modules_feed_forward_modules_w_2_parameters_bias_,
        )
        dropout_14 = l_self_modules_up_encoders_modules_3_modules_feed_forward_modules_w_2_parameters_weight_ = l_self_modules_up_encoders_modules_3_modules_feed_forward_modules_w_2_parameters_bias_ = (None)
        dropout_15 = torch.nn.functional.dropout(linear_27, 0.1, False, False)
        linear_27 = None
        mul_3 = 1.0 * dropout_15
        dropout_15 = None
        x_27 = x_25 + mul_3
        x_25 = mul_3 = None
        xs = torch.nn.functional.layer_norm(
            x_27,
            (512,),
            l_self_modules_after_norm_parameters_weight_,
            l_self_modules_after_norm_parameters_bias_,
            1e-05,
        )
        x_27 = (
            l_self_modules_after_norm_parameters_weight_
        ) = l_self_modules_after_norm_parameters_bias_ = None
        return (xs,)
