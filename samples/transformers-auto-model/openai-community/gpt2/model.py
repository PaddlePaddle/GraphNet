import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_wte_parameters_weight_: torch.nn.parameter.Parameter,
        L_input_ids_: torch.Tensor,
        L_self_modules_wpe_parameters_weight_: torch.nn.parameter.Parameter,
        L_attention_mask_: torch.Tensor,
        L_self_modules_h_modules_0_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_5_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_5_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_5_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_5_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_5_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_5_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_5_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_5_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_6_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_6_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_6_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_6_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_6_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_6_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_6_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_6_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_7_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_7_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_7_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_7_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_7_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_7_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_7_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_7_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_8_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_8_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_8_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_8_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_8_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_8_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_8_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_8_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_9_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_9_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_9_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_9_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_9_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_9_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_9_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_9_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_10_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_10_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_10_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_10_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_10_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_10_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_10_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_10_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_11_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_11_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_11_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_11_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_11_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_11_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_11_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_11_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ln_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ln_f_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_wte_parameters_weight_ = L_self_modules_wte_parameters_weight_
        l_input_ids_ = L_input_ids_
        l_self_modules_wpe_parameters_weight_ = L_self_modules_wpe_parameters_weight_
        l_attention_mask_ = L_attention_mask_
        l_self_modules_h_modules_0_modules_ln_1_parameters_weight_ = (
            L_self_modules_h_modules_0_modules_ln_1_parameters_weight_
        )
        l_self_modules_h_modules_0_modules_ln_1_parameters_bias_ = (
            L_self_modules_h_modules_0_modules_ln_1_parameters_bias_
        )
        l_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_ = (
            L_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_
        )
        l_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_ = (
            L_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_
        )
        l_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_0_modules_ln_2_parameters_weight_ = (
            L_self_modules_h_modules_0_modules_ln_2_parameters_weight_
        )
        l_self_modules_h_modules_0_modules_ln_2_parameters_bias_ = (
            L_self_modules_h_modules_0_modules_ln_2_parameters_bias_
        )
        l_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_ = (
            L_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_
        )
        l_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_ = (
            L_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_
        )
        l_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_1_modules_ln_1_parameters_weight_ = (
            L_self_modules_h_modules_1_modules_ln_1_parameters_weight_
        )
        l_self_modules_h_modules_1_modules_ln_1_parameters_bias_ = (
            L_self_modules_h_modules_1_modules_ln_1_parameters_bias_
        )
        l_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_ = (
            L_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_
        )
        l_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_ = (
            L_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_
        )
        l_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_1_modules_ln_2_parameters_weight_ = (
            L_self_modules_h_modules_1_modules_ln_2_parameters_weight_
        )
        l_self_modules_h_modules_1_modules_ln_2_parameters_bias_ = (
            L_self_modules_h_modules_1_modules_ln_2_parameters_bias_
        )
        l_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_ = (
            L_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_
        )
        l_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_ = (
            L_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_
        )
        l_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_2_modules_ln_1_parameters_weight_ = (
            L_self_modules_h_modules_2_modules_ln_1_parameters_weight_
        )
        l_self_modules_h_modules_2_modules_ln_1_parameters_bias_ = (
            L_self_modules_h_modules_2_modules_ln_1_parameters_bias_
        )
        l_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_ = (
            L_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_
        )
        l_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_ = (
            L_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_
        )
        l_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_2_modules_ln_2_parameters_weight_ = (
            L_self_modules_h_modules_2_modules_ln_2_parameters_weight_
        )
        l_self_modules_h_modules_2_modules_ln_2_parameters_bias_ = (
            L_self_modules_h_modules_2_modules_ln_2_parameters_bias_
        )
        l_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_ = (
            L_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_
        )
        l_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_ = (
            L_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_
        )
        l_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_3_modules_ln_1_parameters_weight_ = (
            L_self_modules_h_modules_3_modules_ln_1_parameters_weight_
        )
        l_self_modules_h_modules_3_modules_ln_1_parameters_bias_ = (
            L_self_modules_h_modules_3_modules_ln_1_parameters_bias_
        )
        l_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_ = (
            L_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_
        )
        l_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_ = (
            L_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_
        )
        l_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_3_modules_ln_2_parameters_weight_ = (
            L_self_modules_h_modules_3_modules_ln_2_parameters_weight_
        )
        l_self_modules_h_modules_3_modules_ln_2_parameters_bias_ = (
            L_self_modules_h_modules_3_modules_ln_2_parameters_bias_
        )
        l_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_ = (
            L_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_
        )
        l_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_ = (
            L_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_
        )
        l_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_4_modules_ln_1_parameters_weight_ = (
            L_self_modules_h_modules_4_modules_ln_1_parameters_weight_
        )
        l_self_modules_h_modules_4_modules_ln_1_parameters_bias_ = (
            L_self_modules_h_modules_4_modules_ln_1_parameters_bias_
        )
        l_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_ = (
            L_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_
        )
        l_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_ = (
            L_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_
        )
        l_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_4_modules_ln_2_parameters_weight_ = (
            L_self_modules_h_modules_4_modules_ln_2_parameters_weight_
        )
        l_self_modules_h_modules_4_modules_ln_2_parameters_bias_ = (
            L_self_modules_h_modules_4_modules_ln_2_parameters_bias_
        )
        l_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_ = (
            L_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_
        )
        l_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_ = (
            L_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_
        )
        l_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_5_modules_ln_1_parameters_weight_ = (
            L_self_modules_h_modules_5_modules_ln_1_parameters_weight_
        )
        l_self_modules_h_modules_5_modules_ln_1_parameters_bias_ = (
            L_self_modules_h_modules_5_modules_ln_1_parameters_bias_
        )
        l_self_modules_h_modules_5_modules_attn_modules_c_attn_parameters_bias_ = (
            L_self_modules_h_modules_5_modules_attn_modules_c_attn_parameters_bias_
        )
        l_self_modules_h_modules_5_modules_attn_modules_c_attn_parameters_weight_ = (
            L_self_modules_h_modules_5_modules_attn_modules_c_attn_parameters_weight_
        )
        l_self_modules_h_modules_5_modules_attn_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_5_modules_attn_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_5_modules_attn_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_5_modules_attn_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_5_modules_ln_2_parameters_weight_ = (
            L_self_modules_h_modules_5_modules_ln_2_parameters_weight_
        )
        l_self_modules_h_modules_5_modules_ln_2_parameters_bias_ = (
            L_self_modules_h_modules_5_modules_ln_2_parameters_bias_
        )
        l_self_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_bias_ = (
            L_self_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_bias_
        )
        l_self_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_weight_ = (
            L_self_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_weight_
        )
        l_self_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_6_modules_ln_1_parameters_weight_ = (
            L_self_modules_h_modules_6_modules_ln_1_parameters_weight_
        )
        l_self_modules_h_modules_6_modules_ln_1_parameters_bias_ = (
            L_self_modules_h_modules_6_modules_ln_1_parameters_bias_
        )
        l_self_modules_h_modules_6_modules_attn_modules_c_attn_parameters_bias_ = (
            L_self_modules_h_modules_6_modules_attn_modules_c_attn_parameters_bias_
        )
        l_self_modules_h_modules_6_modules_attn_modules_c_attn_parameters_weight_ = (
            L_self_modules_h_modules_6_modules_attn_modules_c_attn_parameters_weight_
        )
        l_self_modules_h_modules_6_modules_attn_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_6_modules_attn_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_6_modules_attn_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_6_modules_attn_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_6_modules_ln_2_parameters_weight_ = (
            L_self_modules_h_modules_6_modules_ln_2_parameters_weight_
        )
        l_self_modules_h_modules_6_modules_ln_2_parameters_bias_ = (
            L_self_modules_h_modules_6_modules_ln_2_parameters_bias_
        )
        l_self_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_bias_ = (
            L_self_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_bias_
        )
        l_self_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_weight_ = (
            L_self_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_weight_
        )
        l_self_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_7_modules_ln_1_parameters_weight_ = (
            L_self_modules_h_modules_7_modules_ln_1_parameters_weight_
        )
        l_self_modules_h_modules_7_modules_ln_1_parameters_bias_ = (
            L_self_modules_h_modules_7_modules_ln_1_parameters_bias_
        )
        l_self_modules_h_modules_7_modules_attn_modules_c_attn_parameters_bias_ = (
            L_self_modules_h_modules_7_modules_attn_modules_c_attn_parameters_bias_
        )
        l_self_modules_h_modules_7_modules_attn_modules_c_attn_parameters_weight_ = (
            L_self_modules_h_modules_7_modules_attn_modules_c_attn_parameters_weight_
        )
        l_self_modules_h_modules_7_modules_attn_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_7_modules_attn_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_7_modules_attn_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_7_modules_attn_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_7_modules_ln_2_parameters_weight_ = (
            L_self_modules_h_modules_7_modules_ln_2_parameters_weight_
        )
        l_self_modules_h_modules_7_modules_ln_2_parameters_bias_ = (
            L_self_modules_h_modules_7_modules_ln_2_parameters_bias_
        )
        l_self_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_bias_ = (
            L_self_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_bias_
        )
        l_self_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_weight_ = (
            L_self_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_weight_
        )
        l_self_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_8_modules_ln_1_parameters_weight_ = (
            L_self_modules_h_modules_8_modules_ln_1_parameters_weight_
        )
        l_self_modules_h_modules_8_modules_ln_1_parameters_bias_ = (
            L_self_modules_h_modules_8_modules_ln_1_parameters_bias_
        )
        l_self_modules_h_modules_8_modules_attn_modules_c_attn_parameters_bias_ = (
            L_self_modules_h_modules_8_modules_attn_modules_c_attn_parameters_bias_
        )
        l_self_modules_h_modules_8_modules_attn_modules_c_attn_parameters_weight_ = (
            L_self_modules_h_modules_8_modules_attn_modules_c_attn_parameters_weight_
        )
        l_self_modules_h_modules_8_modules_attn_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_8_modules_attn_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_8_modules_attn_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_8_modules_attn_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_8_modules_ln_2_parameters_weight_ = (
            L_self_modules_h_modules_8_modules_ln_2_parameters_weight_
        )
        l_self_modules_h_modules_8_modules_ln_2_parameters_bias_ = (
            L_self_modules_h_modules_8_modules_ln_2_parameters_bias_
        )
        l_self_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_bias_ = (
            L_self_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_bias_
        )
        l_self_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_weight_ = (
            L_self_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_weight_
        )
        l_self_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_9_modules_ln_1_parameters_weight_ = (
            L_self_modules_h_modules_9_modules_ln_1_parameters_weight_
        )
        l_self_modules_h_modules_9_modules_ln_1_parameters_bias_ = (
            L_self_modules_h_modules_9_modules_ln_1_parameters_bias_
        )
        l_self_modules_h_modules_9_modules_attn_modules_c_attn_parameters_bias_ = (
            L_self_modules_h_modules_9_modules_attn_modules_c_attn_parameters_bias_
        )
        l_self_modules_h_modules_9_modules_attn_modules_c_attn_parameters_weight_ = (
            L_self_modules_h_modules_9_modules_attn_modules_c_attn_parameters_weight_
        )
        l_self_modules_h_modules_9_modules_attn_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_9_modules_attn_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_9_modules_attn_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_9_modules_attn_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_9_modules_ln_2_parameters_weight_ = (
            L_self_modules_h_modules_9_modules_ln_2_parameters_weight_
        )
        l_self_modules_h_modules_9_modules_ln_2_parameters_bias_ = (
            L_self_modules_h_modules_9_modules_ln_2_parameters_bias_
        )
        l_self_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_bias_ = (
            L_self_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_bias_
        )
        l_self_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_weight_ = (
            L_self_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_weight_
        )
        l_self_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_10_modules_ln_1_parameters_weight_ = (
            L_self_modules_h_modules_10_modules_ln_1_parameters_weight_
        )
        l_self_modules_h_modules_10_modules_ln_1_parameters_bias_ = (
            L_self_modules_h_modules_10_modules_ln_1_parameters_bias_
        )
        l_self_modules_h_modules_10_modules_attn_modules_c_attn_parameters_bias_ = (
            L_self_modules_h_modules_10_modules_attn_modules_c_attn_parameters_bias_
        )
        l_self_modules_h_modules_10_modules_attn_modules_c_attn_parameters_weight_ = (
            L_self_modules_h_modules_10_modules_attn_modules_c_attn_parameters_weight_
        )
        l_self_modules_h_modules_10_modules_attn_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_10_modules_attn_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_10_modules_attn_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_10_modules_attn_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_10_modules_ln_2_parameters_weight_ = (
            L_self_modules_h_modules_10_modules_ln_2_parameters_weight_
        )
        l_self_modules_h_modules_10_modules_ln_2_parameters_bias_ = (
            L_self_modules_h_modules_10_modules_ln_2_parameters_bias_
        )
        l_self_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_bias_ = (
            L_self_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_bias_
        )
        l_self_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_weight_ = (
            L_self_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_weight_
        )
        l_self_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_11_modules_ln_1_parameters_weight_ = (
            L_self_modules_h_modules_11_modules_ln_1_parameters_weight_
        )
        l_self_modules_h_modules_11_modules_ln_1_parameters_bias_ = (
            L_self_modules_h_modules_11_modules_ln_1_parameters_bias_
        )
        l_self_modules_h_modules_11_modules_attn_modules_c_attn_parameters_bias_ = (
            L_self_modules_h_modules_11_modules_attn_modules_c_attn_parameters_bias_
        )
        l_self_modules_h_modules_11_modules_attn_modules_c_attn_parameters_weight_ = (
            L_self_modules_h_modules_11_modules_attn_modules_c_attn_parameters_weight_
        )
        l_self_modules_h_modules_11_modules_attn_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_11_modules_attn_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_11_modules_attn_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_11_modules_attn_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_11_modules_ln_2_parameters_weight_ = (
            L_self_modules_h_modules_11_modules_ln_2_parameters_weight_
        )
        l_self_modules_h_modules_11_modules_ln_2_parameters_bias_ = (
            L_self_modules_h_modules_11_modules_ln_2_parameters_bias_
        )
        l_self_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_bias_ = (
            L_self_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_bias_
        )
        l_self_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_weight_ = (
            L_self_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_weight_
        )
        l_self_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_weight_
        )
        l_self_modules_ln_f_parameters_weight_ = L_self_modules_ln_f_parameters_weight_
        l_self_modules_ln_f_parameters_bias_ = L_self_modules_ln_f_parameters_bias_
        inputs_embeds = torch.nn.functional.embedding(
            l_input_ids_,
            l_self_modules_wte_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        l_input_ids_ = l_self_modules_wte_parameters_weight_ = None
        cache_position = torch.arange(0, 2, device=device(type="cuda", index=0))
        position_ids = cache_position.unsqueeze(0)
        position_embeds = torch.nn.functional.embedding(
            position_ids,
            l_self_modules_wpe_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        position_ids = l_self_modules_wpe_parameters_weight_ = None
        to = position_embeds.to(device(type="cuda", index=0))
        position_embeds = None
        hidden_states = inputs_embeds + to
        inputs_embeds = to = None
        attention_mask = l_attention_mask_.view(1, -1)
        l_attention_mask_ = None
        attention_mask_1 = attention_mask.to(
            device=device(type="cuda", index=0), dtype=torch.bool
        )
        attention_mask = None
        mask_indices = torch.arange(2, device=device(type="cuda", index=0))
        mask_indices += 0
        mask_indices_1 = mask_indices
        mask_indices = None
        local_padding_mask = attention_mask_1[(slice(None, None, None), mask_indices_1)]
        attention_mask_1 = mask_indices_1 = None
        kv_arange = torch.arange(2, device=device(type="cuda", index=0))
        kv_arange += 0
        kv_arange_1 = kv_arange
        kv_arange = None
        reshaped_cache_position = cache_position.view(-1, 1)
        cache_position = None
        causal_mask = kv_arange_1 <= reshaped_cache_position
        kv_arange_1 = reshaped_cache_position = None
        getitem_1 = causal_mask[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        causal_mask = None
        causal_mask_1 = getitem_1.expand(1, -1, -1, -1)
        getitem_1 = None
        getitem_2 = local_padding_mask[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        local_padding_mask = None
        causal_mask_2 = causal_mask_1 * getitem_2
        causal_mask_1 = getitem_2 = None
        hidden_states_1 = torch.nn.functional.dropout(hidden_states, 0.1, False, False)
        hidden_states = None
        hidden_states_2 = torch.nn.functional.layer_norm(
            hidden_states_1,
            (768,),
            l_self_modules_h_modules_0_modules_ln_1_parameters_weight_,
            l_self_modules_h_modules_0_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_0_modules_ln_1_parameters_weight_ = (
            l_self_modules_h_modules_0_modules_ln_1_parameters_bias_
        ) = None
        view_2 = hidden_states_2.view(-1, 768)
        hidden_states_2 = None
        x = torch.addmm(
            l_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_,
            view_2,
            l_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_ = (
            view_2
        ) = (
            l_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_
        ) = None
        x_1 = x.view((1, 2, 2304))
        x = None
        split = x_1.split(768, dim=2)
        x_1 = None
        query_states = split[0]
        key_states = split[1]
        value_states = split[2]
        split = None
        view_4 = key_states.view((1, 2, -1, 64))
        key_states = None
        key_states_1 = view_4.transpose(1, 2)
        view_4 = None
        view_5 = value_states.view((1, 2, -1, 64))
        value_states = None
        value_states_1 = view_5.transpose(1, 2)
        view_5 = None
        view_6 = query_states.view((1, 2, -1, 64))
        query_states = None
        query_states_1 = view_6.transpose(1, 2)
        view_6 = None
        attention_mask_2 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query = query_states_1.contiguous()
        query_states_1 = None
        key = key_states_1.contiguous()
        value = value_states_1.contiguous()
        attn_output = torch._C._nn.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask_2,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query = key = value = attention_mask_2 = None
        transpose_3 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_3.contiguous()
        transpose_3 = None
        reshape = attn_output_1.reshape(1, 2, -1)
        attn_output_1 = None
        attn_output_2 = reshape.contiguous()
        reshape = None
        view_7 = attn_output_2.view(-1, 768)
        attn_output_2 = None
        x_2 = torch.addmm(
            l_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_,
            view_7,
            l_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_ = (
            view_7
        ) = (
            l_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_
        ) = None
        x_3 = x_2.view((1, 2, 768))
        x_2 = None
        attn_output_3 = torch.nn.functional.dropout(x_3, 0.1, False, False)
        x_3 = None
        hidden_states_3 = attn_output_3 + hidden_states_1
        attn_output_3 = hidden_states_1 = None
        hidden_states_4 = torch.nn.functional.layer_norm(
            hidden_states_3,
            (768,),
            l_self_modules_h_modules_0_modules_ln_2_parameters_weight_,
            l_self_modules_h_modules_0_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_0_modules_ln_2_parameters_weight_ = (
            l_self_modules_h_modules_0_modules_ln_2_parameters_bias_
        ) = None
        view_9 = hidden_states_4.view(-1, 768)
        hidden_states_4 = None
        x_4 = torch.addmm(
            l_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_,
            view_9,
            l_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_9
        ) = (
            l_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_
        ) = None
        x_5 = x_4.view((1, 2, 3072))
        x_4 = None
        mul_1 = 0.5 * x_5
        pow_1 = torch.pow(x_5, 3.0)
        mul_2 = 0.044715 * pow_1
        pow_1 = None
        add_2 = x_5 + mul_2
        x_5 = mul_2 = None
        mul_3 = 0.7978845608028654 * add_2
        add_2 = None
        tanh = torch.tanh(mul_3)
        mul_3 = None
        add_3 = 1.0 + tanh
        tanh = None
        hidden_states_5 = mul_1 * add_3
        mul_1 = add_3 = None
        view_11 = hidden_states_5.view(-1, 3072)
        hidden_states_5 = None
        x_6 = torch.addmm(
            l_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_,
            view_11,
            l_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_11
        ) = (
            l_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_
        ) = None
        x_7 = x_6.view((1, 2, 768))
        x_6 = None
        hidden_states_6 = torch.nn.functional.dropout(x_7, 0.1, False, False)
        x_7 = None
        hidden_states_7 = hidden_states_3 + hidden_states_6
        hidden_states_3 = hidden_states_6 = None
        hidden_states_8 = torch.nn.functional.layer_norm(
            hidden_states_7,
            (768,),
            l_self_modules_h_modules_1_modules_ln_1_parameters_weight_,
            l_self_modules_h_modules_1_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_1_modules_ln_1_parameters_weight_ = (
            l_self_modules_h_modules_1_modules_ln_1_parameters_bias_
        ) = None
        view_13 = hidden_states_8.view(-1, 768)
        hidden_states_8 = None
        x_8 = torch.addmm(
            l_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_,
            view_13,
            l_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_ = (
            view_13
        ) = (
            l_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_
        ) = None
        x_9 = x_8.view((1, 2, 2304))
        x_8 = None
        split_1 = x_9.split(768, dim=2)
        x_9 = None
        query_states_2 = split_1[0]
        key_states_2 = split_1[1]
        value_states_2 = split_1[2]
        split_1 = None
        view_15 = key_states_2.view((1, 2, -1, 64))
        key_states_2 = None
        key_states_3 = view_15.transpose(1, 2)
        view_15 = None
        view_16 = value_states_2.view((1, 2, -1, 64))
        value_states_2 = None
        value_states_3 = view_16.transpose(1, 2)
        view_16 = None
        view_17 = query_states_2.view((1, 2, -1, 64))
        query_states_2 = None
        query_states_3 = view_17.transpose(1, 2)
        view_17 = None
        attention_mask_3 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_1 = query_states_3.contiguous()
        query_states_3 = None
        key_1 = key_states_3.contiguous()
        value_1 = value_states_3.contiguous()
        attn_output_4 = torch._C._nn.scaled_dot_product_attention(
            query_1,
            key_1,
            value_1,
            attn_mask=attention_mask_3,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query_1 = key_1 = value_1 = attention_mask_3 = None
        transpose_7 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_7.contiguous()
        transpose_7 = None
        reshape_1 = attn_output_5.reshape(1, 2, -1)
        attn_output_5 = None
        attn_output_6 = reshape_1.contiguous()
        reshape_1 = None
        view_18 = attn_output_6.view(-1, 768)
        attn_output_6 = None
        x_10 = torch.addmm(
            l_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_,
            view_18,
            l_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_ = (
            view_18
        ) = (
            l_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_
        ) = None
        x_11 = x_10.view((1, 2, 768))
        x_10 = None
        attn_output_7 = torch.nn.functional.dropout(x_11, 0.1, False, False)
        x_11 = None
        hidden_states_9 = attn_output_7 + hidden_states_7
        attn_output_7 = hidden_states_7 = None
        hidden_states_10 = torch.nn.functional.layer_norm(
            hidden_states_9,
            (768,),
            l_self_modules_h_modules_1_modules_ln_2_parameters_weight_,
            l_self_modules_h_modules_1_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_1_modules_ln_2_parameters_weight_ = (
            l_self_modules_h_modules_1_modules_ln_2_parameters_bias_
        ) = None
        view_20 = hidden_states_10.view(-1, 768)
        hidden_states_10 = None
        x_12 = torch.addmm(
            l_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_,
            view_20,
            l_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_20
        ) = (
            l_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_
        ) = None
        x_13 = x_12.view((1, 2, 3072))
        x_12 = None
        mul_5 = 0.5 * x_13
        pow_2 = torch.pow(x_13, 3.0)
        mul_6 = 0.044715 * pow_2
        pow_2 = None
        add_6 = x_13 + mul_6
        x_13 = mul_6 = None
        mul_7 = 0.7978845608028654 * add_6
        add_6 = None
        tanh_1 = torch.tanh(mul_7)
        mul_7 = None
        add_7 = 1.0 + tanh_1
        tanh_1 = None
        hidden_states_11 = mul_5 * add_7
        mul_5 = add_7 = None
        view_22 = hidden_states_11.view(-1, 3072)
        hidden_states_11 = None
        x_14 = torch.addmm(
            l_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_,
            view_22,
            l_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_22
        ) = (
            l_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_
        ) = None
        x_15 = x_14.view((1, 2, 768))
        x_14 = None
        hidden_states_12 = torch.nn.functional.dropout(x_15, 0.1, False, False)
        x_15 = None
        hidden_states_13 = hidden_states_9 + hidden_states_12
        hidden_states_9 = hidden_states_12 = None
        hidden_states_14 = torch.nn.functional.layer_norm(
            hidden_states_13,
            (768,),
            l_self_modules_h_modules_2_modules_ln_1_parameters_weight_,
            l_self_modules_h_modules_2_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_2_modules_ln_1_parameters_weight_ = (
            l_self_modules_h_modules_2_modules_ln_1_parameters_bias_
        ) = None
        view_24 = hidden_states_14.view(-1, 768)
        hidden_states_14 = None
        x_16 = torch.addmm(
            l_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_,
            view_24,
            l_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_ = (
            view_24
        ) = (
            l_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_
        ) = None
        x_17 = x_16.view((1, 2, 2304))
        x_16 = None
        split_2 = x_17.split(768, dim=2)
        x_17 = None
        query_states_4 = split_2[0]
        key_states_4 = split_2[1]
        value_states_4 = split_2[2]
        split_2 = None
        view_26 = key_states_4.view((1, 2, -1, 64))
        key_states_4 = None
        key_states_5 = view_26.transpose(1, 2)
        view_26 = None
        view_27 = value_states_4.view((1, 2, -1, 64))
        value_states_4 = None
        value_states_5 = view_27.transpose(1, 2)
        view_27 = None
        view_28 = query_states_4.view((1, 2, -1, 64))
        query_states_4 = None
        query_states_5 = view_28.transpose(1, 2)
        view_28 = None
        attention_mask_4 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_2 = query_states_5.contiguous()
        query_states_5 = None
        key_2 = key_states_5.contiguous()
        value_2 = value_states_5.contiguous()
        attn_output_8 = torch._C._nn.scaled_dot_product_attention(
            query_2,
            key_2,
            value_2,
            attn_mask=attention_mask_4,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query_2 = key_2 = value_2 = attention_mask_4 = None
        transpose_11 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_11.contiguous()
        transpose_11 = None
        reshape_2 = attn_output_9.reshape(1, 2, -1)
        attn_output_9 = None
        attn_output_10 = reshape_2.contiguous()
        reshape_2 = None
        view_29 = attn_output_10.view(-1, 768)
        attn_output_10 = None
        x_18 = torch.addmm(
            l_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_,
            view_29,
            l_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_ = (
            view_29
        ) = (
            l_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_
        ) = None
        x_19 = x_18.view((1, 2, 768))
        x_18 = None
        attn_output_11 = torch.nn.functional.dropout(x_19, 0.1, False, False)
        x_19 = None
        hidden_states_15 = attn_output_11 + hidden_states_13
        attn_output_11 = hidden_states_13 = None
        hidden_states_16 = torch.nn.functional.layer_norm(
            hidden_states_15,
            (768,),
            l_self_modules_h_modules_2_modules_ln_2_parameters_weight_,
            l_self_modules_h_modules_2_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_2_modules_ln_2_parameters_weight_ = (
            l_self_modules_h_modules_2_modules_ln_2_parameters_bias_
        ) = None
        view_31 = hidden_states_16.view(-1, 768)
        hidden_states_16 = None
        x_20 = torch.addmm(
            l_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_,
            view_31,
            l_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_31
        ) = (
            l_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_
        ) = None
        x_21 = x_20.view((1, 2, 3072))
        x_20 = None
        mul_9 = 0.5 * x_21
        pow_3 = torch.pow(x_21, 3.0)
        mul_10 = 0.044715 * pow_3
        pow_3 = None
        add_10 = x_21 + mul_10
        x_21 = mul_10 = None
        mul_11 = 0.7978845608028654 * add_10
        add_10 = None
        tanh_2 = torch.tanh(mul_11)
        mul_11 = None
        add_11 = 1.0 + tanh_2
        tanh_2 = None
        hidden_states_17 = mul_9 * add_11
        mul_9 = add_11 = None
        view_33 = hidden_states_17.view(-1, 3072)
        hidden_states_17 = None
        x_22 = torch.addmm(
            l_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_,
            view_33,
            l_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_33
        ) = (
            l_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_
        ) = None
        x_23 = x_22.view((1, 2, 768))
        x_22 = None
        hidden_states_18 = torch.nn.functional.dropout(x_23, 0.1, False, False)
        x_23 = None
        hidden_states_19 = hidden_states_15 + hidden_states_18
        hidden_states_15 = hidden_states_18 = None
        hidden_states_20 = torch.nn.functional.layer_norm(
            hidden_states_19,
            (768,),
            l_self_modules_h_modules_3_modules_ln_1_parameters_weight_,
            l_self_modules_h_modules_3_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_3_modules_ln_1_parameters_weight_ = (
            l_self_modules_h_modules_3_modules_ln_1_parameters_bias_
        ) = None
        view_35 = hidden_states_20.view(-1, 768)
        hidden_states_20 = None
        x_24 = torch.addmm(
            l_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_,
            view_35,
            l_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_ = (
            view_35
        ) = (
            l_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_
        ) = None
        x_25 = x_24.view((1, 2, 2304))
        x_24 = None
        split_3 = x_25.split(768, dim=2)
        x_25 = None
        query_states_6 = split_3[0]
        key_states_6 = split_3[1]
        value_states_6 = split_3[2]
        split_3 = None
        view_37 = key_states_6.view((1, 2, -1, 64))
        key_states_6 = None
        key_states_7 = view_37.transpose(1, 2)
        view_37 = None
        view_38 = value_states_6.view((1, 2, -1, 64))
        value_states_6 = None
        value_states_7 = view_38.transpose(1, 2)
        view_38 = None
        view_39 = query_states_6.view((1, 2, -1, 64))
        query_states_6 = None
        query_states_7 = view_39.transpose(1, 2)
        view_39 = None
        attention_mask_5 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_3 = query_states_7.contiguous()
        query_states_7 = None
        key_3 = key_states_7.contiguous()
        value_3 = value_states_7.contiguous()
        attn_output_12 = torch._C._nn.scaled_dot_product_attention(
            query_3,
            key_3,
            value_3,
            attn_mask=attention_mask_5,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query_3 = key_3 = value_3 = attention_mask_5 = None
        transpose_15 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_15.contiguous()
        transpose_15 = None
        reshape_3 = attn_output_13.reshape(1, 2, -1)
        attn_output_13 = None
        attn_output_14 = reshape_3.contiguous()
        reshape_3 = None
        view_40 = attn_output_14.view(-1, 768)
        attn_output_14 = None
        x_26 = torch.addmm(
            l_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_,
            view_40,
            l_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_ = (
            view_40
        ) = (
            l_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_
        ) = None
        x_27 = x_26.view((1, 2, 768))
        x_26 = None
        attn_output_15 = torch.nn.functional.dropout(x_27, 0.1, False, False)
        x_27 = None
        hidden_states_21 = attn_output_15 + hidden_states_19
        attn_output_15 = hidden_states_19 = None
        hidden_states_22 = torch.nn.functional.layer_norm(
            hidden_states_21,
            (768,),
            l_self_modules_h_modules_3_modules_ln_2_parameters_weight_,
            l_self_modules_h_modules_3_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_3_modules_ln_2_parameters_weight_ = (
            l_self_modules_h_modules_3_modules_ln_2_parameters_bias_
        ) = None
        view_42 = hidden_states_22.view(-1, 768)
        hidden_states_22 = None
        x_28 = torch.addmm(
            l_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_,
            view_42,
            l_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_42
        ) = (
            l_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_
        ) = None
        x_29 = x_28.view((1, 2, 3072))
        x_28 = None
        mul_13 = 0.5 * x_29
        pow_4 = torch.pow(x_29, 3.0)
        mul_14 = 0.044715 * pow_4
        pow_4 = None
        add_14 = x_29 + mul_14
        x_29 = mul_14 = None
        mul_15 = 0.7978845608028654 * add_14
        add_14 = None
        tanh_3 = torch.tanh(mul_15)
        mul_15 = None
        add_15 = 1.0 + tanh_3
        tanh_3 = None
        hidden_states_23 = mul_13 * add_15
        mul_13 = add_15 = None
        view_44 = hidden_states_23.view(-1, 3072)
        hidden_states_23 = None
        x_30 = torch.addmm(
            l_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_,
            view_44,
            l_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_44
        ) = (
            l_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_
        ) = None
        x_31 = x_30.view((1, 2, 768))
        x_30 = None
        hidden_states_24 = torch.nn.functional.dropout(x_31, 0.1, False, False)
        x_31 = None
        hidden_states_25 = hidden_states_21 + hidden_states_24
        hidden_states_21 = hidden_states_24 = None
        hidden_states_26 = torch.nn.functional.layer_norm(
            hidden_states_25,
            (768,),
            l_self_modules_h_modules_4_modules_ln_1_parameters_weight_,
            l_self_modules_h_modules_4_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_4_modules_ln_1_parameters_weight_ = (
            l_self_modules_h_modules_4_modules_ln_1_parameters_bias_
        ) = None
        view_46 = hidden_states_26.view(-1, 768)
        hidden_states_26 = None
        x_32 = torch.addmm(
            l_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_,
            view_46,
            l_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_ = (
            view_46
        ) = (
            l_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_
        ) = None
        x_33 = x_32.view((1, 2, 2304))
        x_32 = None
        split_4 = x_33.split(768, dim=2)
        x_33 = None
        query_states_8 = split_4[0]
        key_states_8 = split_4[1]
        value_states_8 = split_4[2]
        split_4 = None
        view_48 = key_states_8.view((1, 2, -1, 64))
        key_states_8 = None
        key_states_9 = view_48.transpose(1, 2)
        view_48 = None
        view_49 = value_states_8.view((1, 2, -1, 64))
        value_states_8 = None
        value_states_9 = view_49.transpose(1, 2)
        view_49 = None
        view_50 = query_states_8.view((1, 2, -1, 64))
        query_states_8 = None
        query_states_9 = view_50.transpose(1, 2)
        view_50 = None
        attention_mask_6 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_4 = query_states_9.contiguous()
        query_states_9 = None
        key_4 = key_states_9.contiguous()
        value_4 = value_states_9.contiguous()
        attn_output_16 = torch._C._nn.scaled_dot_product_attention(
            query_4,
            key_4,
            value_4,
            attn_mask=attention_mask_6,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query_4 = key_4 = value_4 = attention_mask_6 = None
        transpose_19 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_17 = transpose_19.contiguous()
        transpose_19 = None
        reshape_4 = attn_output_17.reshape(1, 2, -1)
        attn_output_17 = None
        attn_output_18 = reshape_4.contiguous()
        reshape_4 = None
        view_51 = attn_output_18.view(-1, 768)
        attn_output_18 = None
        x_34 = torch.addmm(
            l_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_,
            view_51,
            l_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_ = (
            view_51
        ) = (
            l_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_
        ) = None
        x_35 = x_34.view((1, 2, 768))
        x_34 = None
        attn_output_19 = torch.nn.functional.dropout(x_35, 0.1, False, False)
        x_35 = None
        hidden_states_27 = attn_output_19 + hidden_states_25
        attn_output_19 = hidden_states_25 = None
        hidden_states_28 = torch.nn.functional.layer_norm(
            hidden_states_27,
            (768,),
            l_self_modules_h_modules_4_modules_ln_2_parameters_weight_,
            l_self_modules_h_modules_4_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_4_modules_ln_2_parameters_weight_ = (
            l_self_modules_h_modules_4_modules_ln_2_parameters_bias_
        ) = None
        view_53 = hidden_states_28.view(-1, 768)
        hidden_states_28 = None
        x_36 = torch.addmm(
            l_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_,
            view_53,
            l_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_53
        ) = (
            l_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_
        ) = None
        x_37 = x_36.view((1, 2, 3072))
        x_36 = None
        mul_17 = 0.5 * x_37
        pow_5 = torch.pow(x_37, 3.0)
        mul_18 = 0.044715 * pow_5
        pow_5 = None
        add_18 = x_37 + mul_18
        x_37 = mul_18 = None
        mul_19 = 0.7978845608028654 * add_18
        add_18 = None
        tanh_4 = torch.tanh(mul_19)
        mul_19 = None
        add_19 = 1.0 + tanh_4
        tanh_4 = None
        hidden_states_29 = mul_17 * add_19
        mul_17 = add_19 = None
        view_55 = hidden_states_29.view(-1, 3072)
        hidden_states_29 = None
        x_38 = torch.addmm(
            l_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_,
            view_55,
            l_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_55
        ) = (
            l_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_
        ) = None
        x_39 = x_38.view((1, 2, 768))
        x_38 = None
        hidden_states_30 = torch.nn.functional.dropout(x_39, 0.1, False, False)
        x_39 = None
        hidden_states_31 = hidden_states_27 + hidden_states_30
        hidden_states_27 = hidden_states_30 = None
        hidden_states_32 = torch.nn.functional.layer_norm(
            hidden_states_31,
            (768,),
            l_self_modules_h_modules_5_modules_ln_1_parameters_weight_,
            l_self_modules_h_modules_5_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_5_modules_ln_1_parameters_weight_ = (
            l_self_modules_h_modules_5_modules_ln_1_parameters_bias_
        ) = None
        view_57 = hidden_states_32.view(-1, 768)
        hidden_states_32 = None
        x_40 = torch.addmm(
            l_self_modules_h_modules_5_modules_attn_modules_c_attn_parameters_bias_,
            view_57,
            l_self_modules_h_modules_5_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_h_modules_5_modules_attn_modules_c_attn_parameters_bias_ = (
            view_57
        ) = (
            l_self_modules_h_modules_5_modules_attn_modules_c_attn_parameters_weight_
        ) = None
        x_41 = x_40.view((1, 2, 2304))
        x_40 = None
        split_5 = x_41.split(768, dim=2)
        x_41 = None
        query_states_10 = split_5[0]
        key_states_10 = split_5[1]
        value_states_10 = split_5[2]
        split_5 = None
        view_59 = key_states_10.view((1, 2, -1, 64))
        key_states_10 = None
        key_states_11 = view_59.transpose(1, 2)
        view_59 = None
        view_60 = value_states_10.view((1, 2, -1, 64))
        value_states_10 = None
        value_states_11 = view_60.transpose(1, 2)
        view_60 = None
        view_61 = query_states_10.view((1, 2, -1, 64))
        query_states_10 = None
        query_states_11 = view_61.transpose(1, 2)
        view_61 = None
        attention_mask_7 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_5 = query_states_11.contiguous()
        query_states_11 = None
        key_5 = key_states_11.contiguous()
        value_5 = value_states_11.contiguous()
        attn_output_20 = torch._C._nn.scaled_dot_product_attention(
            query_5,
            key_5,
            value_5,
            attn_mask=attention_mask_7,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query_5 = key_5 = value_5 = attention_mask_7 = None
        transpose_23 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_23.contiguous()
        transpose_23 = None
        reshape_5 = attn_output_21.reshape(1, 2, -1)
        attn_output_21 = None
        attn_output_22 = reshape_5.contiguous()
        reshape_5 = None
        view_62 = attn_output_22.view(-1, 768)
        attn_output_22 = None
        x_42 = torch.addmm(
            l_self_modules_h_modules_5_modules_attn_modules_c_proj_parameters_bias_,
            view_62,
            l_self_modules_h_modules_5_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_5_modules_attn_modules_c_proj_parameters_bias_ = (
            view_62
        ) = (
            l_self_modules_h_modules_5_modules_attn_modules_c_proj_parameters_weight_
        ) = None
        x_43 = x_42.view((1, 2, 768))
        x_42 = None
        attn_output_23 = torch.nn.functional.dropout(x_43, 0.1, False, False)
        x_43 = None
        hidden_states_33 = attn_output_23 + hidden_states_31
        attn_output_23 = hidden_states_31 = None
        hidden_states_34 = torch.nn.functional.layer_norm(
            hidden_states_33,
            (768,),
            l_self_modules_h_modules_5_modules_ln_2_parameters_weight_,
            l_self_modules_h_modules_5_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_5_modules_ln_2_parameters_weight_ = (
            l_self_modules_h_modules_5_modules_ln_2_parameters_bias_
        ) = None
        view_64 = hidden_states_34.view(-1, 768)
        hidden_states_34 = None
        x_44 = torch.addmm(
            l_self_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_bias_,
            view_64,
            l_self_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_64
        ) = (
            l_self_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_weight_
        ) = None
        x_45 = x_44.view((1, 2, 3072))
        x_44 = None
        mul_21 = 0.5 * x_45
        pow_6 = torch.pow(x_45, 3.0)
        mul_22 = 0.044715 * pow_6
        pow_6 = None
        add_22 = x_45 + mul_22
        x_45 = mul_22 = None
        mul_23 = 0.7978845608028654 * add_22
        add_22 = None
        tanh_5 = torch.tanh(mul_23)
        mul_23 = None
        add_23 = 1.0 + tanh_5
        tanh_5 = None
        hidden_states_35 = mul_21 * add_23
        mul_21 = add_23 = None
        view_66 = hidden_states_35.view(-1, 3072)
        hidden_states_35 = None
        x_46 = torch.addmm(
            l_self_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_bias_,
            view_66,
            l_self_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_66
        ) = (
            l_self_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_weight_
        ) = None
        x_47 = x_46.view((1, 2, 768))
        x_46 = None
        hidden_states_36 = torch.nn.functional.dropout(x_47, 0.1, False, False)
        x_47 = None
        hidden_states_37 = hidden_states_33 + hidden_states_36
        hidden_states_33 = hidden_states_36 = None
        hidden_states_38 = torch.nn.functional.layer_norm(
            hidden_states_37,
            (768,),
            l_self_modules_h_modules_6_modules_ln_1_parameters_weight_,
            l_self_modules_h_modules_6_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_6_modules_ln_1_parameters_weight_ = (
            l_self_modules_h_modules_6_modules_ln_1_parameters_bias_
        ) = None
        view_68 = hidden_states_38.view(-1, 768)
        hidden_states_38 = None
        x_48 = torch.addmm(
            l_self_modules_h_modules_6_modules_attn_modules_c_attn_parameters_bias_,
            view_68,
            l_self_modules_h_modules_6_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_h_modules_6_modules_attn_modules_c_attn_parameters_bias_ = (
            view_68
        ) = (
            l_self_modules_h_modules_6_modules_attn_modules_c_attn_parameters_weight_
        ) = None
        x_49 = x_48.view((1, 2, 2304))
        x_48 = None
        split_6 = x_49.split(768, dim=2)
        x_49 = None
        query_states_12 = split_6[0]
        key_states_12 = split_6[1]
        value_states_12 = split_6[2]
        split_6 = None
        view_70 = key_states_12.view((1, 2, -1, 64))
        key_states_12 = None
        key_states_13 = view_70.transpose(1, 2)
        view_70 = None
        view_71 = value_states_12.view((1, 2, -1, 64))
        value_states_12 = None
        value_states_13 = view_71.transpose(1, 2)
        view_71 = None
        view_72 = query_states_12.view((1, 2, -1, 64))
        query_states_12 = None
        query_states_13 = view_72.transpose(1, 2)
        view_72 = None
        attention_mask_8 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_6 = query_states_13.contiguous()
        query_states_13 = None
        key_6 = key_states_13.contiguous()
        value_6 = value_states_13.contiguous()
        attn_output_24 = torch._C._nn.scaled_dot_product_attention(
            query_6,
            key_6,
            value_6,
            attn_mask=attention_mask_8,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query_6 = key_6 = value_6 = attention_mask_8 = None
        transpose_27 = attn_output_24.transpose(1, 2)
        attn_output_24 = None
        attn_output_25 = transpose_27.contiguous()
        transpose_27 = None
        reshape_6 = attn_output_25.reshape(1, 2, -1)
        attn_output_25 = None
        attn_output_26 = reshape_6.contiguous()
        reshape_6 = None
        view_73 = attn_output_26.view(-1, 768)
        attn_output_26 = None
        x_50 = torch.addmm(
            l_self_modules_h_modules_6_modules_attn_modules_c_proj_parameters_bias_,
            view_73,
            l_self_modules_h_modules_6_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_6_modules_attn_modules_c_proj_parameters_bias_ = (
            view_73
        ) = (
            l_self_modules_h_modules_6_modules_attn_modules_c_proj_parameters_weight_
        ) = None
        x_51 = x_50.view((1, 2, 768))
        x_50 = None
        attn_output_27 = torch.nn.functional.dropout(x_51, 0.1, False, False)
        x_51 = None
        hidden_states_39 = attn_output_27 + hidden_states_37
        attn_output_27 = hidden_states_37 = None
        hidden_states_40 = torch.nn.functional.layer_norm(
            hidden_states_39,
            (768,),
            l_self_modules_h_modules_6_modules_ln_2_parameters_weight_,
            l_self_modules_h_modules_6_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_6_modules_ln_2_parameters_weight_ = (
            l_self_modules_h_modules_6_modules_ln_2_parameters_bias_
        ) = None
        view_75 = hidden_states_40.view(-1, 768)
        hidden_states_40 = None
        x_52 = torch.addmm(
            l_self_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_bias_,
            view_75,
            l_self_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_75
        ) = (
            l_self_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_weight_
        ) = None
        x_53 = x_52.view((1, 2, 3072))
        x_52 = None
        mul_25 = 0.5 * x_53
        pow_7 = torch.pow(x_53, 3.0)
        mul_26 = 0.044715 * pow_7
        pow_7 = None
        add_26 = x_53 + mul_26
        x_53 = mul_26 = None
        mul_27 = 0.7978845608028654 * add_26
        add_26 = None
        tanh_6 = torch.tanh(mul_27)
        mul_27 = None
        add_27 = 1.0 + tanh_6
        tanh_6 = None
        hidden_states_41 = mul_25 * add_27
        mul_25 = add_27 = None
        view_77 = hidden_states_41.view(-1, 3072)
        hidden_states_41 = None
        x_54 = torch.addmm(
            l_self_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_bias_,
            view_77,
            l_self_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_77
        ) = (
            l_self_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_weight_
        ) = None
        x_55 = x_54.view((1, 2, 768))
        x_54 = None
        hidden_states_42 = torch.nn.functional.dropout(x_55, 0.1, False, False)
        x_55 = None
        hidden_states_43 = hidden_states_39 + hidden_states_42
        hidden_states_39 = hidden_states_42 = None
        hidden_states_44 = torch.nn.functional.layer_norm(
            hidden_states_43,
            (768,),
            l_self_modules_h_modules_7_modules_ln_1_parameters_weight_,
            l_self_modules_h_modules_7_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_7_modules_ln_1_parameters_weight_ = (
            l_self_modules_h_modules_7_modules_ln_1_parameters_bias_
        ) = None
        view_79 = hidden_states_44.view(-1, 768)
        hidden_states_44 = None
        x_56 = torch.addmm(
            l_self_modules_h_modules_7_modules_attn_modules_c_attn_parameters_bias_,
            view_79,
            l_self_modules_h_modules_7_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_h_modules_7_modules_attn_modules_c_attn_parameters_bias_ = (
            view_79
        ) = (
            l_self_modules_h_modules_7_modules_attn_modules_c_attn_parameters_weight_
        ) = None
        x_57 = x_56.view((1, 2, 2304))
        x_56 = None
        split_7 = x_57.split(768, dim=2)
        x_57 = None
        query_states_14 = split_7[0]
        key_states_14 = split_7[1]
        value_states_14 = split_7[2]
        split_7 = None
        view_81 = key_states_14.view((1, 2, -1, 64))
        key_states_14 = None
        key_states_15 = view_81.transpose(1, 2)
        view_81 = None
        view_82 = value_states_14.view((1, 2, -1, 64))
        value_states_14 = None
        value_states_15 = view_82.transpose(1, 2)
        view_82 = None
        view_83 = query_states_14.view((1, 2, -1, 64))
        query_states_14 = None
        query_states_15 = view_83.transpose(1, 2)
        view_83 = None
        attention_mask_9 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_7 = query_states_15.contiguous()
        query_states_15 = None
        key_7 = key_states_15.contiguous()
        value_7 = value_states_15.contiguous()
        attn_output_28 = torch._C._nn.scaled_dot_product_attention(
            query_7,
            key_7,
            value_7,
            attn_mask=attention_mask_9,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query_7 = key_7 = value_7 = attention_mask_9 = None
        transpose_31 = attn_output_28.transpose(1, 2)
        attn_output_28 = None
        attn_output_29 = transpose_31.contiguous()
        transpose_31 = None
        reshape_7 = attn_output_29.reshape(1, 2, -1)
        attn_output_29 = None
        attn_output_30 = reshape_7.contiguous()
        reshape_7 = None
        view_84 = attn_output_30.view(-1, 768)
        attn_output_30 = None
        x_58 = torch.addmm(
            l_self_modules_h_modules_7_modules_attn_modules_c_proj_parameters_bias_,
            view_84,
            l_self_modules_h_modules_7_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_7_modules_attn_modules_c_proj_parameters_bias_ = (
            view_84
        ) = (
            l_self_modules_h_modules_7_modules_attn_modules_c_proj_parameters_weight_
        ) = None
        x_59 = x_58.view((1, 2, 768))
        x_58 = None
        attn_output_31 = torch.nn.functional.dropout(x_59, 0.1, False, False)
        x_59 = None
        hidden_states_45 = attn_output_31 + hidden_states_43
        attn_output_31 = hidden_states_43 = None
        hidden_states_46 = torch.nn.functional.layer_norm(
            hidden_states_45,
            (768,),
            l_self_modules_h_modules_7_modules_ln_2_parameters_weight_,
            l_self_modules_h_modules_7_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_7_modules_ln_2_parameters_weight_ = (
            l_self_modules_h_modules_7_modules_ln_2_parameters_bias_
        ) = None
        view_86 = hidden_states_46.view(-1, 768)
        hidden_states_46 = None
        x_60 = torch.addmm(
            l_self_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_bias_,
            view_86,
            l_self_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_86
        ) = (
            l_self_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_weight_
        ) = None
        x_61 = x_60.view((1, 2, 3072))
        x_60 = None
        mul_29 = 0.5 * x_61
        pow_8 = torch.pow(x_61, 3.0)
        mul_30 = 0.044715 * pow_8
        pow_8 = None
        add_30 = x_61 + mul_30
        x_61 = mul_30 = None
        mul_31 = 0.7978845608028654 * add_30
        add_30 = None
        tanh_7 = torch.tanh(mul_31)
        mul_31 = None
        add_31 = 1.0 + tanh_7
        tanh_7 = None
        hidden_states_47 = mul_29 * add_31
        mul_29 = add_31 = None
        view_88 = hidden_states_47.view(-1, 3072)
        hidden_states_47 = None
        x_62 = torch.addmm(
            l_self_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_bias_,
            view_88,
            l_self_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_88
        ) = (
            l_self_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_weight_
        ) = None
        x_63 = x_62.view((1, 2, 768))
        x_62 = None
        hidden_states_48 = torch.nn.functional.dropout(x_63, 0.1, False, False)
        x_63 = None
        hidden_states_49 = hidden_states_45 + hidden_states_48
        hidden_states_45 = hidden_states_48 = None
        hidden_states_50 = torch.nn.functional.layer_norm(
            hidden_states_49,
            (768,),
            l_self_modules_h_modules_8_modules_ln_1_parameters_weight_,
            l_self_modules_h_modules_8_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_8_modules_ln_1_parameters_weight_ = (
            l_self_modules_h_modules_8_modules_ln_1_parameters_bias_
        ) = None
        view_90 = hidden_states_50.view(-1, 768)
        hidden_states_50 = None
        x_64 = torch.addmm(
            l_self_modules_h_modules_8_modules_attn_modules_c_attn_parameters_bias_,
            view_90,
            l_self_modules_h_modules_8_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_h_modules_8_modules_attn_modules_c_attn_parameters_bias_ = (
            view_90
        ) = (
            l_self_modules_h_modules_8_modules_attn_modules_c_attn_parameters_weight_
        ) = None
        x_65 = x_64.view((1, 2, 2304))
        x_64 = None
        split_8 = x_65.split(768, dim=2)
        x_65 = None
        query_states_16 = split_8[0]
        key_states_16 = split_8[1]
        value_states_16 = split_8[2]
        split_8 = None
        view_92 = key_states_16.view((1, 2, -1, 64))
        key_states_16 = None
        key_states_17 = view_92.transpose(1, 2)
        view_92 = None
        view_93 = value_states_16.view((1, 2, -1, 64))
        value_states_16 = None
        value_states_17 = view_93.transpose(1, 2)
        view_93 = None
        view_94 = query_states_16.view((1, 2, -1, 64))
        query_states_16 = None
        query_states_17 = view_94.transpose(1, 2)
        view_94 = None
        attention_mask_10 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_8 = query_states_17.contiguous()
        query_states_17 = None
        key_8 = key_states_17.contiguous()
        value_8 = value_states_17.contiguous()
        attn_output_32 = torch._C._nn.scaled_dot_product_attention(
            query_8,
            key_8,
            value_8,
            attn_mask=attention_mask_10,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query_8 = key_8 = value_8 = attention_mask_10 = None
        transpose_35 = attn_output_32.transpose(1, 2)
        attn_output_32 = None
        attn_output_33 = transpose_35.contiguous()
        transpose_35 = None
        reshape_8 = attn_output_33.reshape(1, 2, -1)
        attn_output_33 = None
        attn_output_34 = reshape_8.contiguous()
        reshape_8 = None
        view_95 = attn_output_34.view(-1, 768)
        attn_output_34 = None
        x_66 = torch.addmm(
            l_self_modules_h_modules_8_modules_attn_modules_c_proj_parameters_bias_,
            view_95,
            l_self_modules_h_modules_8_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_8_modules_attn_modules_c_proj_parameters_bias_ = (
            view_95
        ) = (
            l_self_modules_h_modules_8_modules_attn_modules_c_proj_parameters_weight_
        ) = None
        x_67 = x_66.view((1, 2, 768))
        x_66 = None
        attn_output_35 = torch.nn.functional.dropout(x_67, 0.1, False, False)
        x_67 = None
        hidden_states_51 = attn_output_35 + hidden_states_49
        attn_output_35 = hidden_states_49 = None
        hidden_states_52 = torch.nn.functional.layer_norm(
            hidden_states_51,
            (768,),
            l_self_modules_h_modules_8_modules_ln_2_parameters_weight_,
            l_self_modules_h_modules_8_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_8_modules_ln_2_parameters_weight_ = (
            l_self_modules_h_modules_8_modules_ln_2_parameters_bias_
        ) = None
        view_97 = hidden_states_52.view(-1, 768)
        hidden_states_52 = None
        x_68 = torch.addmm(
            l_self_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_bias_,
            view_97,
            l_self_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_97
        ) = (
            l_self_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_weight_
        ) = None
        x_69 = x_68.view((1, 2, 3072))
        x_68 = None
        mul_33 = 0.5 * x_69
        pow_9 = torch.pow(x_69, 3.0)
        mul_34 = 0.044715 * pow_9
        pow_9 = None
        add_34 = x_69 + mul_34
        x_69 = mul_34 = None
        mul_35 = 0.7978845608028654 * add_34
        add_34 = None
        tanh_8 = torch.tanh(mul_35)
        mul_35 = None
        add_35 = 1.0 + tanh_8
        tanh_8 = None
        hidden_states_53 = mul_33 * add_35
        mul_33 = add_35 = None
        view_99 = hidden_states_53.view(-1, 3072)
        hidden_states_53 = None
        x_70 = torch.addmm(
            l_self_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_bias_,
            view_99,
            l_self_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_99
        ) = (
            l_self_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_weight_
        ) = None
        x_71 = x_70.view((1, 2, 768))
        x_70 = None
        hidden_states_54 = torch.nn.functional.dropout(x_71, 0.1, False, False)
        x_71 = None
        hidden_states_55 = hidden_states_51 + hidden_states_54
        hidden_states_51 = hidden_states_54 = None
        hidden_states_56 = torch.nn.functional.layer_norm(
            hidden_states_55,
            (768,),
            l_self_modules_h_modules_9_modules_ln_1_parameters_weight_,
            l_self_modules_h_modules_9_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_9_modules_ln_1_parameters_weight_ = (
            l_self_modules_h_modules_9_modules_ln_1_parameters_bias_
        ) = None
        view_101 = hidden_states_56.view(-1, 768)
        hidden_states_56 = None
        x_72 = torch.addmm(
            l_self_modules_h_modules_9_modules_attn_modules_c_attn_parameters_bias_,
            view_101,
            l_self_modules_h_modules_9_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_h_modules_9_modules_attn_modules_c_attn_parameters_bias_ = (
            view_101
        ) = (
            l_self_modules_h_modules_9_modules_attn_modules_c_attn_parameters_weight_
        ) = None
        x_73 = x_72.view((1, 2, 2304))
        x_72 = None
        split_9 = x_73.split(768, dim=2)
        x_73 = None
        query_states_18 = split_9[0]
        key_states_18 = split_9[1]
        value_states_18 = split_9[2]
        split_9 = None
        view_103 = key_states_18.view((1, 2, -1, 64))
        key_states_18 = None
        key_states_19 = view_103.transpose(1, 2)
        view_103 = None
        view_104 = value_states_18.view((1, 2, -1, 64))
        value_states_18 = None
        value_states_19 = view_104.transpose(1, 2)
        view_104 = None
        view_105 = query_states_18.view((1, 2, -1, 64))
        query_states_18 = None
        query_states_19 = view_105.transpose(1, 2)
        view_105 = None
        attention_mask_11 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_9 = query_states_19.contiguous()
        query_states_19 = None
        key_9 = key_states_19.contiguous()
        value_9 = value_states_19.contiguous()
        attn_output_36 = torch._C._nn.scaled_dot_product_attention(
            query_9,
            key_9,
            value_9,
            attn_mask=attention_mask_11,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query_9 = key_9 = value_9 = attention_mask_11 = None
        transpose_39 = attn_output_36.transpose(1, 2)
        attn_output_36 = None
        attn_output_37 = transpose_39.contiguous()
        transpose_39 = None
        reshape_9 = attn_output_37.reshape(1, 2, -1)
        attn_output_37 = None
        attn_output_38 = reshape_9.contiguous()
        reshape_9 = None
        view_106 = attn_output_38.view(-1, 768)
        attn_output_38 = None
        x_74 = torch.addmm(
            l_self_modules_h_modules_9_modules_attn_modules_c_proj_parameters_bias_,
            view_106,
            l_self_modules_h_modules_9_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_9_modules_attn_modules_c_proj_parameters_bias_ = (
            view_106
        ) = (
            l_self_modules_h_modules_9_modules_attn_modules_c_proj_parameters_weight_
        ) = None
        x_75 = x_74.view((1, 2, 768))
        x_74 = None
        attn_output_39 = torch.nn.functional.dropout(x_75, 0.1, False, False)
        x_75 = None
        hidden_states_57 = attn_output_39 + hidden_states_55
        attn_output_39 = hidden_states_55 = None
        hidden_states_58 = torch.nn.functional.layer_norm(
            hidden_states_57,
            (768,),
            l_self_modules_h_modules_9_modules_ln_2_parameters_weight_,
            l_self_modules_h_modules_9_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_9_modules_ln_2_parameters_weight_ = (
            l_self_modules_h_modules_9_modules_ln_2_parameters_bias_
        ) = None
        view_108 = hidden_states_58.view(-1, 768)
        hidden_states_58 = None
        x_76 = torch.addmm(
            l_self_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_bias_,
            view_108,
            l_self_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_108
        ) = (
            l_self_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_weight_
        ) = None
        x_77 = x_76.view((1, 2, 3072))
        x_76 = None
        mul_37 = 0.5 * x_77
        pow_10 = torch.pow(x_77, 3.0)
        mul_38 = 0.044715 * pow_10
        pow_10 = None
        add_38 = x_77 + mul_38
        x_77 = mul_38 = None
        mul_39 = 0.7978845608028654 * add_38
        add_38 = None
        tanh_9 = torch.tanh(mul_39)
        mul_39 = None
        add_39 = 1.0 + tanh_9
        tanh_9 = None
        hidden_states_59 = mul_37 * add_39
        mul_37 = add_39 = None
        view_110 = hidden_states_59.view(-1, 3072)
        hidden_states_59 = None
        x_78 = torch.addmm(
            l_self_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_bias_,
            view_110,
            l_self_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_110
        ) = (
            l_self_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_weight_
        ) = None
        x_79 = x_78.view((1, 2, 768))
        x_78 = None
        hidden_states_60 = torch.nn.functional.dropout(x_79, 0.1, False, False)
        x_79 = None
        hidden_states_61 = hidden_states_57 + hidden_states_60
        hidden_states_57 = hidden_states_60 = None
        hidden_states_62 = torch.nn.functional.layer_norm(
            hidden_states_61,
            (768,),
            l_self_modules_h_modules_10_modules_ln_1_parameters_weight_,
            l_self_modules_h_modules_10_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_10_modules_ln_1_parameters_weight_ = (
            l_self_modules_h_modules_10_modules_ln_1_parameters_bias_
        ) = None
        view_112 = hidden_states_62.view(-1, 768)
        hidden_states_62 = None
        x_80 = torch.addmm(
            l_self_modules_h_modules_10_modules_attn_modules_c_attn_parameters_bias_,
            view_112,
            l_self_modules_h_modules_10_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_h_modules_10_modules_attn_modules_c_attn_parameters_bias_ = (
            view_112
        ) = (
            l_self_modules_h_modules_10_modules_attn_modules_c_attn_parameters_weight_
        ) = None
        x_81 = x_80.view((1, 2, 2304))
        x_80 = None
        split_10 = x_81.split(768, dim=2)
        x_81 = None
        query_states_20 = split_10[0]
        key_states_20 = split_10[1]
        value_states_20 = split_10[2]
        split_10 = None
        view_114 = key_states_20.view((1, 2, -1, 64))
        key_states_20 = None
        key_states_21 = view_114.transpose(1, 2)
        view_114 = None
        view_115 = value_states_20.view((1, 2, -1, 64))
        value_states_20 = None
        value_states_21 = view_115.transpose(1, 2)
        view_115 = None
        view_116 = query_states_20.view((1, 2, -1, 64))
        query_states_20 = None
        query_states_21 = view_116.transpose(1, 2)
        view_116 = None
        attention_mask_12 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_10 = query_states_21.contiguous()
        query_states_21 = None
        key_10 = key_states_21.contiguous()
        value_10 = value_states_21.contiguous()
        attn_output_40 = torch._C._nn.scaled_dot_product_attention(
            query_10,
            key_10,
            value_10,
            attn_mask=attention_mask_12,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query_10 = key_10 = value_10 = attention_mask_12 = None
        transpose_43 = attn_output_40.transpose(1, 2)
        attn_output_40 = None
        attn_output_41 = transpose_43.contiguous()
        transpose_43 = None
        reshape_10 = attn_output_41.reshape(1, 2, -1)
        attn_output_41 = None
        attn_output_42 = reshape_10.contiguous()
        reshape_10 = None
        view_117 = attn_output_42.view(-1, 768)
        attn_output_42 = None
        x_82 = torch.addmm(
            l_self_modules_h_modules_10_modules_attn_modules_c_proj_parameters_bias_,
            view_117,
            l_self_modules_h_modules_10_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_10_modules_attn_modules_c_proj_parameters_bias_ = (
            view_117
        ) = (
            l_self_modules_h_modules_10_modules_attn_modules_c_proj_parameters_weight_
        ) = None
        x_83 = x_82.view((1, 2, 768))
        x_82 = None
        attn_output_43 = torch.nn.functional.dropout(x_83, 0.1, False, False)
        x_83 = None
        hidden_states_63 = attn_output_43 + hidden_states_61
        attn_output_43 = hidden_states_61 = None
        hidden_states_64 = torch.nn.functional.layer_norm(
            hidden_states_63,
            (768,),
            l_self_modules_h_modules_10_modules_ln_2_parameters_weight_,
            l_self_modules_h_modules_10_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_10_modules_ln_2_parameters_weight_ = (
            l_self_modules_h_modules_10_modules_ln_2_parameters_bias_
        ) = None
        view_119 = hidden_states_64.view(-1, 768)
        hidden_states_64 = None
        x_84 = torch.addmm(
            l_self_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_bias_,
            view_119,
            l_self_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_119
        ) = (
            l_self_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_weight_
        ) = None
        x_85 = x_84.view((1, 2, 3072))
        x_84 = None
        mul_41 = 0.5 * x_85
        pow_11 = torch.pow(x_85, 3.0)
        mul_42 = 0.044715 * pow_11
        pow_11 = None
        add_42 = x_85 + mul_42
        x_85 = mul_42 = None
        mul_43 = 0.7978845608028654 * add_42
        add_42 = None
        tanh_10 = torch.tanh(mul_43)
        mul_43 = None
        add_43 = 1.0 + tanh_10
        tanh_10 = None
        hidden_states_65 = mul_41 * add_43
        mul_41 = add_43 = None
        view_121 = hidden_states_65.view(-1, 3072)
        hidden_states_65 = None
        x_86 = torch.addmm(
            l_self_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_bias_,
            view_121,
            l_self_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_121
        ) = (
            l_self_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_weight_
        ) = None
        x_87 = x_86.view((1, 2, 768))
        x_86 = None
        hidden_states_66 = torch.nn.functional.dropout(x_87, 0.1, False, False)
        x_87 = None
        hidden_states_67 = hidden_states_63 + hidden_states_66
        hidden_states_63 = hidden_states_66 = None
        hidden_states_68 = torch.nn.functional.layer_norm(
            hidden_states_67,
            (768,),
            l_self_modules_h_modules_11_modules_ln_1_parameters_weight_,
            l_self_modules_h_modules_11_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_11_modules_ln_1_parameters_weight_ = (
            l_self_modules_h_modules_11_modules_ln_1_parameters_bias_
        ) = None
        view_123 = hidden_states_68.view(-1, 768)
        hidden_states_68 = None
        x_88 = torch.addmm(
            l_self_modules_h_modules_11_modules_attn_modules_c_attn_parameters_bias_,
            view_123,
            l_self_modules_h_modules_11_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_h_modules_11_modules_attn_modules_c_attn_parameters_bias_ = (
            view_123
        ) = (
            l_self_modules_h_modules_11_modules_attn_modules_c_attn_parameters_weight_
        ) = None
        x_89 = x_88.view((1, 2, 2304))
        x_88 = None
        split_11 = x_89.split(768, dim=2)
        x_89 = None
        query_states_22 = split_11[0]
        key_states_22 = split_11[1]
        value_states_22 = split_11[2]
        split_11 = None
        view_125 = key_states_22.view((1, 2, -1, 64))
        key_states_22 = None
        key_states_23 = view_125.transpose(1, 2)
        view_125 = None
        view_126 = value_states_22.view((1, 2, -1, 64))
        value_states_22 = None
        value_states_23 = view_126.transpose(1, 2)
        view_126 = None
        view_127 = query_states_22.view((1, 2, -1, 64))
        query_states_22 = None
        query_states_23 = view_127.transpose(1, 2)
        view_127 = None
        attention_mask_13 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        causal_mask_2 = None
        query_11 = query_states_23.contiguous()
        query_states_23 = None
        key_11 = key_states_23.contiguous()
        value_11 = value_states_23.contiguous()
        attn_output_44 = torch._C._nn.scaled_dot_product_attention(
            query_11,
            key_11,
            value_11,
            attn_mask=attention_mask_13,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query_11 = key_11 = value_11 = attention_mask_13 = None
        transpose_47 = attn_output_44.transpose(1, 2)
        attn_output_44 = None
        attn_output_45 = transpose_47.contiguous()
        transpose_47 = None
        reshape_11 = attn_output_45.reshape(1, 2, -1)
        attn_output_45 = None
        attn_output_46 = reshape_11.contiguous()
        reshape_11 = None
        view_128 = attn_output_46.view(-1, 768)
        attn_output_46 = None
        x_90 = torch.addmm(
            l_self_modules_h_modules_11_modules_attn_modules_c_proj_parameters_bias_,
            view_128,
            l_self_modules_h_modules_11_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_11_modules_attn_modules_c_proj_parameters_bias_ = (
            view_128
        ) = (
            l_self_modules_h_modules_11_modules_attn_modules_c_proj_parameters_weight_
        ) = None
        x_91 = x_90.view((1, 2, 768))
        x_90 = None
        attn_output_47 = torch.nn.functional.dropout(x_91, 0.1, False, False)
        x_91 = None
        hidden_states_69 = attn_output_47 + hidden_states_67
        attn_output_47 = hidden_states_67 = None
        hidden_states_70 = torch.nn.functional.layer_norm(
            hidden_states_69,
            (768,),
            l_self_modules_h_modules_11_modules_ln_2_parameters_weight_,
            l_self_modules_h_modules_11_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_11_modules_ln_2_parameters_weight_ = (
            l_self_modules_h_modules_11_modules_ln_2_parameters_bias_
        ) = None
        view_130 = hidden_states_70.view(-1, 768)
        hidden_states_70 = None
        x_92 = torch.addmm(
            l_self_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_bias_,
            view_130,
            l_self_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_130
        ) = (
            l_self_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_weight_
        ) = None
        x_93 = x_92.view((1, 2, 3072))
        x_92 = None
        mul_45 = 0.5 * x_93
        pow_12 = torch.pow(x_93, 3.0)
        mul_46 = 0.044715 * pow_12
        pow_12 = None
        add_46 = x_93 + mul_46
        x_93 = mul_46 = None
        mul_47 = 0.7978845608028654 * add_46
        add_46 = None
        tanh_11 = torch.tanh(mul_47)
        mul_47 = None
        add_47 = 1.0 + tanh_11
        tanh_11 = None
        hidden_states_71 = mul_45 * add_47
        mul_45 = add_47 = None
        view_132 = hidden_states_71.view(-1, 3072)
        hidden_states_71 = None
        x_94 = torch.addmm(
            l_self_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_bias_,
            view_132,
            l_self_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_132
        ) = (
            l_self_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_weight_
        ) = None
        x_95 = x_94.view((1, 2, 768))
        x_94 = None
        hidden_states_72 = torch.nn.functional.dropout(x_95, 0.1, False, False)
        x_95 = None
        hidden_states_73 = hidden_states_69 + hidden_states_72
        hidden_states_69 = hidden_states_72 = None
        hidden_states_74 = torch.nn.functional.layer_norm(
            hidden_states_73,
            (768,),
            l_self_modules_ln_f_parameters_weight_,
            l_self_modules_ln_f_parameters_bias_,
            1e-05,
        )
        hidden_states_73 = (
            l_self_modules_ln_f_parameters_weight_
        ) = l_self_modules_ln_f_parameters_bias_ = None
        hidden_states_75 = hidden_states_74.view((-1, 2, 768))
        hidden_states_74 = None
        return (
            value_states_1,
            key_states_1,
            value_states_3,
            key_states_3,
            value_states_5,
            key_states_5,
            value_states_7,
            key_states_7,
            value_states_9,
            key_states_9,
            value_states_11,
            key_states_11,
            value_states_13,
            key_states_13,
            value_states_15,
            key_states_15,
            value_states_17,
            key_states_17,
            value_states_19,
            key_states_19,
            value_states_21,
            key_states_21,
            value_states_23,
            key_states_23,
            hidden_states_75,
        )
