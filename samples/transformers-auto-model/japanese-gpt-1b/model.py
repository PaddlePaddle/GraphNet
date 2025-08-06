import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_self_modules_transformer_modules_wte_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_wpe_parameters_weight_: torch.nn.parameter.Parameter,
        L_attention_mask_: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_12_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_12_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_12_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_12_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_13_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_13_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_13_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_13_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_14_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_14_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_14_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_14_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_15_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_15_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_15_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_15_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_16_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_16_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_16_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_16_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_17_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_17_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_17_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_17_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_18_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_18_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_18_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_18_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_19_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_19_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_19_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_19_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_20_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_20_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_20_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_20_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_20_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_20_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_20_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_20_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_20_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_20_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_20_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_20_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_21_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_21_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_21_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_21_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_21_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_21_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_21_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_21_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_21_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_21_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_21_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_21_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_22_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_22_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_22_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_22_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_22_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_22_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_22_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_22_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_22_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_22_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_22_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_22_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_23_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_23_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_23_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_23_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_23_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_23_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_23_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_23_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_23_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_23_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_23_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_23_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_ln_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_ln_f_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_input_ids_ = L_input_ids_
        l_self_modules_transformer_modules_wte_parameters_weight_ = (
            L_self_modules_transformer_modules_wte_parameters_weight_
        )
        l_self_modules_transformer_modules_wpe_parameters_weight_ = (
            L_self_modules_transformer_modules_wpe_parameters_weight_
        )
        l_attention_mask_ = L_attention_mask_
        l_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_6_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_6_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_6_modules_ln_2_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_6_modules_ln_2_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_7_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_7_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_7_modules_ln_2_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_7_modules_ln_2_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_8_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_8_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_8_modules_ln_2_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_8_modules_ln_2_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_9_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_9_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_9_modules_ln_2_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_9_modules_ln_2_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_bias_ = L_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_bias_
        l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_10_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_10_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_10_modules_ln_2_parameters_bias_ = L_self_modules_transformer_modules_h_modules_10_modules_ln_2_parameters_bias_
        l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_bias_ = L_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_bias_
        l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_11_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_11_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_11_modules_ln_2_parameters_bias_ = L_self_modules_transformer_modules_h_modules_11_modules_ln_2_parameters_bias_
        l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_12_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_12_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_12_modules_ln_1_parameters_bias_ = L_self_modules_transformer_modules_h_modules_12_modules_ln_1_parameters_bias_
        l_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_12_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_12_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_12_modules_ln_2_parameters_bias_ = L_self_modules_transformer_modules_h_modules_12_modules_ln_2_parameters_bias_
        l_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_13_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_13_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_13_modules_ln_1_parameters_bias_ = L_self_modules_transformer_modules_h_modules_13_modules_ln_1_parameters_bias_
        l_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_13_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_13_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_13_modules_ln_2_parameters_bias_ = L_self_modules_transformer_modules_h_modules_13_modules_ln_2_parameters_bias_
        l_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_14_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_14_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_14_modules_ln_1_parameters_bias_ = L_self_modules_transformer_modules_h_modules_14_modules_ln_1_parameters_bias_
        l_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_14_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_14_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_14_modules_ln_2_parameters_bias_ = L_self_modules_transformer_modules_h_modules_14_modules_ln_2_parameters_bias_
        l_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_15_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_15_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_15_modules_ln_1_parameters_bias_ = L_self_modules_transformer_modules_h_modules_15_modules_ln_1_parameters_bias_
        l_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_15_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_15_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_15_modules_ln_2_parameters_bias_ = L_self_modules_transformer_modules_h_modules_15_modules_ln_2_parameters_bias_
        l_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_16_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_16_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_16_modules_ln_1_parameters_bias_ = L_self_modules_transformer_modules_h_modules_16_modules_ln_1_parameters_bias_
        l_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_16_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_16_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_16_modules_ln_2_parameters_bias_ = L_self_modules_transformer_modules_h_modules_16_modules_ln_2_parameters_bias_
        l_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_17_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_17_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_17_modules_ln_1_parameters_bias_ = L_self_modules_transformer_modules_h_modules_17_modules_ln_1_parameters_bias_
        l_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_17_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_17_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_17_modules_ln_2_parameters_bias_ = L_self_modules_transformer_modules_h_modules_17_modules_ln_2_parameters_bias_
        l_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_18_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_18_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_18_modules_ln_1_parameters_bias_ = L_self_modules_transformer_modules_h_modules_18_modules_ln_1_parameters_bias_
        l_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_18_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_18_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_18_modules_ln_2_parameters_bias_ = L_self_modules_transformer_modules_h_modules_18_modules_ln_2_parameters_bias_
        l_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_19_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_19_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_19_modules_ln_1_parameters_bias_ = L_self_modules_transformer_modules_h_modules_19_modules_ln_1_parameters_bias_
        l_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_19_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_19_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_19_modules_ln_2_parameters_bias_ = L_self_modules_transformer_modules_h_modules_19_modules_ln_2_parameters_bias_
        l_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_20_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_20_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_20_modules_ln_1_parameters_bias_ = L_self_modules_transformer_modules_h_modules_20_modules_ln_1_parameters_bias_
        l_self_modules_transformer_modules_h_modules_20_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_20_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_20_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_20_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_20_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_20_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_20_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_20_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_20_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_20_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_20_modules_ln_2_parameters_bias_ = L_self_modules_transformer_modules_h_modules_20_modules_ln_2_parameters_bias_
        l_self_modules_transformer_modules_h_modules_20_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_20_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_20_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_20_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_20_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_20_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_20_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_20_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_21_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_21_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_21_modules_ln_1_parameters_bias_ = L_self_modules_transformer_modules_h_modules_21_modules_ln_1_parameters_bias_
        l_self_modules_transformer_modules_h_modules_21_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_21_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_21_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_21_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_21_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_21_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_21_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_21_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_21_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_21_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_21_modules_ln_2_parameters_bias_ = L_self_modules_transformer_modules_h_modules_21_modules_ln_2_parameters_bias_
        l_self_modules_transformer_modules_h_modules_21_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_21_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_21_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_21_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_21_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_21_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_21_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_21_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_22_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_22_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_22_modules_ln_1_parameters_bias_ = L_self_modules_transformer_modules_h_modules_22_modules_ln_1_parameters_bias_
        l_self_modules_transformer_modules_h_modules_22_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_22_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_22_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_22_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_22_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_22_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_22_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_22_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_22_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_22_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_22_modules_ln_2_parameters_bias_ = L_self_modules_transformer_modules_h_modules_22_modules_ln_2_parameters_bias_
        l_self_modules_transformer_modules_h_modules_22_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_22_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_22_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_22_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_22_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_22_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_22_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_22_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_23_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_23_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_23_modules_ln_1_parameters_bias_ = L_self_modules_transformer_modules_h_modules_23_modules_ln_1_parameters_bias_
        l_self_modules_transformer_modules_h_modules_23_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_23_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_23_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_23_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_23_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_23_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_23_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_23_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_23_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_23_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_23_modules_ln_2_parameters_bias_ = L_self_modules_transformer_modules_h_modules_23_modules_ln_2_parameters_bias_
        l_self_modules_transformer_modules_h_modules_23_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_23_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_23_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_23_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_23_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_23_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_23_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_23_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_ln_f_parameters_weight_ = (
            L_self_modules_transformer_modules_ln_f_parameters_weight_
        )
        l_self_modules_transformer_modules_ln_f_parameters_bias_ = (
            L_self_modules_transformer_modules_ln_f_parameters_bias_
        )
        input_ids = l_input_ids_.view(-1, 39)
        l_input_ids_ = None
        inputs_embeds = torch.nn.functional.embedding(
            input_ids,
            l_self_modules_transformer_modules_wte_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        input_ids = None
        cache_position = torch.arange(0, 39, device=device(type="cpu"))
        position_ids = cache_position.unsqueeze(0)
        position_embeds = torch.nn.functional.embedding(
            position_ids,
            l_self_modules_transformer_modules_wpe_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        position_ids = l_self_modules_transformer_modules_wpe_parameters_weight_ = None
        to = position_embeds.to(device(type="cpu"))
        position_embeds = None
        hidden_states = inputs_embeds + to
        inputs_embeds = to = None
        attention_mask = l_attention_mask_.view(1, -1)
        l_attention_mask_ = None
        attention_mask_1 = attention_mask.to(
            device=device(type="cpu"), dtype=torch.bool
        )
        attention_mask = None
        kv_arange = torch.arange(39, device=device(type="cpu"))
        kv_arange += 0
        kv_arange_1 = kv_arange
        kv_arange = None
        batch_arange = torch.arange(1, device=device(type="cpu"))
        head_arange = torch.arange(1, device=device(type="cpu"))
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
            39, "error"
        )
        _vmap_increment_nesting_2 = None
        child_2 = torch._C._functorch._add_batch_dim(cache_position, 0, 3)
        cache_position = None
        lazy_load_decompositions_3 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_3 = None
        _vmap_increment_nesting_3 = torch._C._functorch._vmap_increment_nesting(
            39, "error"
        )
        _vmap_increment_nesting_3 = None
        child_3 = torch._C._functorch._add_batch_dim(kv_arange_1, 0, 4)
        kv_arange_1 = None
        result = child_2.new_ones((), dtype=torch.bool)
        le = child_3.le(child_2)
        child_2 = None
        to_2 = le.to(device(type="cpu"))
        le = None
        result_1 = result.__and__(to_2)
        result = to_2 = None
        function_ctx = torch.autograd.function.FunctionCtx()
        function_ctx = None
        index = torch.ops.aten.index(attention_mask_1, [child, child_3])
        attention_mask_1 = child = child_3 = None
        to_3 = index.to(device(type="cpu"))
        index = None
        result_2 = result_1.__and__(to_3)
        result_1 = to_3 = None
        batched_outputs = torch._C._functorch._remove_batch_dim(result_2, 4, 39, 0)
        result_2 = None
        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting = None
        batched_outputs_1 = torch._C._functorch._remove_batch_dim(
            batched_outputs, 3, 39, 0
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
        hidden_states_1 = torch.nn.functional.dropout(hidden_states, 0.1, False, False)
        hidden_states = None
        hidden_states_2 = torch.nn.functional.layer_norm(
            hidden_states_1,
            (2048,),
            l_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_bias_
        ) = None
        view_2 = hidden_states_2.view(-1, 2048)
        hidden_states_2 = None
        x = torch.addmm(
            l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_,
            view_2,
            l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_ = (
            view_2
        ) = l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_1 = x.view((1, 39, 6144))
        x = None
        split = x_1.split(2048, dim=2)
        x_1 = None
        query_states = split[0]
        key_states = split[1]
        value_states = split[2]
        split = None
        view_4 = key_states.view((1, 39, -1, 128))
        key_states = None
        key_states_1 = view_4.transpose(1, 2)
        view_4 = None
        view_5 = value_states.view((1, 39, -1, 128))
        value_states = None
        value_states_1 = view_5.transpose(1, 2)
        view_5 = None
        view_6 = query_states.view((1, 39, -1, 128))
        query_states = None
        query_states_1 = view_6.transpose(1, 2)
        view_6 = None
        attention_mask_2 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 39, None),
            )
        ]
        query = query_states_1.contiguous()
        query_states_1 = None
        key = key_states_1.contiguous()
        key_states_1 = None
        value = value_states_1.contiguous()
        value_states_1 = None
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
        reshape = attn_output_1.reshape(1, 39, -1)
        attn_output_1 = None
        attn_output_2 = reshape.contiguous()
        reshape = None
        view_7 = attn_output_2.view(-1, 2048)
        attn_output_2 = None
        x_2 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_,
            view_7,
            l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_ = (
            view_7
        ) = l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_3 = x_2.view((1, 39, 2048))
        x_2 = None
        attn_output_3 = torch.nn.functional.dropout(x_3, 0.1, False, False)
        x_3 = None
        hidden_states_3 = attn_output_3 + hidden_states_1
        attn_output_3 = hidden_states_1 = None
        hidden_states_4 = torch.nn.functional.layer_norm(
            hidden_states_3,
            (2048,),
            l_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_bias_
        ) = None
        view_9 = hidden_states_4.view(-1, 2048)
        hidden_states_4 = None
        x_4 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_,
            view_9,
            l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_9
        ) = l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_5 = x_4.view((1, 39, 8192))
        x_4 = None
        mul = 0.5 * x_5
        mul_1 = x_5 * 0.7978845608
        mul_2 = 0.044715 * x_5
        mul_3 = mul_2 * x_5
        mul_2 = x_5 = None
        add_2 = 1.0 + mul_3
        mul_3 = None
        mul_4 = mul_1 * add_2
        mul_1 = add_2 = None
        tanh = torch.tanh(mul_4)
        mul_4 = None
        add_3 = 1.0 + tanh
        tanh = None
        hidden_states_5 = mul * add_3
        mul = add_3 = None
        view_11 = hidden_states_5.view(-1, 8192)
        hidden_states_5 = None
        x_6 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_,
            view_11,
            l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_11
        ) = l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_7 = x_6.view((1, 39, 2048))
        x_6 = None
        hidden_states_6 = torch.nn.functional.dropout(x_7, 0.1, False, False)
        x_7 = None
        hidden_states_7 = hidden_states_3 + hidden_states_6
        hidden_states_3 = hidden_states_6 = None
        hidden_states_8 = torch.nn.functional.layer_norm(
            hidden_states_7,
            (2048,),
            l_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_bias_
        ) = None
        view_13 = hidden_states_8.view(-1, 2048)
        hidden_states_8 = None
        x_8 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_,
            view_13,
            l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_ = (
            view_13
        ) = l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_9 = x_8.view((1, 39, 6144))
        x_8 = None
        split_1 = x_9.split(2048, dim=2)
        x_9 = None
        query_states_2 = split_1[0]
        key_states_2 = split_1[1]
        value_states_2 = split_1[2]
        split_1 = None
        view_15 = key_states_2.view((1, 39, -1, 128))
        key_states_2 = None
        key_states_3 = view_15.transpose(1, 2)
        view_15 = None
        view_16 = value_states_2.view((1, 39, -1, 128))
        value_states_2 = None
        value_states_3 = view_16.transpose(1, 2)
        view_16 = None
        view_17 = query_states_2.view((1, 39, -1, 128))
        query_states_2 = None
        query_states_3 = view_17.transpose(1, 2)
        view_17 = None
        attention_mask_3 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 39, None),
            )
        ]
        query_1 = query_states_3.contiguous()
        query_states_3 = None
        key_1 = key_states_3.contiguous()
        key_states_3 = None
        value_1 = value_states_3.contiguous()
        value_states_3 = None
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
        reshape_1 = attn_output_5.reshape(1, 39, -1)
        attn_output_5 = None
        attn_output_6 = reshape_1.contiguous()
        reshape_1 = None
        view_18 = attn_output_6.view(-1, 2048)
        attn_output_6 = None
        x_10 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_,
            view_18,
            l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_ = (
            view_18
        ) = l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_11 = x_10.view((1, 39, 2048))
        x_10 = None
        attn_output_7 = torch.nn.functional.dropout(x_11, 0.1, False, False)
        x_11 = None
        hidden_states_9 = attn_output_7 + hidden_states_7
        attn_output_7 = hidden_states_7 = None
        hidden_states_10 = torch.nn.functional.layer_norm(
            hidden_states_9,
            (2048,),
            l_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_bias_
        ) = None
        view_20 = hidden_states_10.view(-1, 2048)
        hidden_states_10 = None
        x_12 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_,
            view_20,
            l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_20
        ) = l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_13 = x_12.view((1, 39, 8192))
        x_12 = None
        mul_6 = 0.5 * x_13
        mul_7 = x_13 * 0.7978845608
        mul_8 = 0.044715 * x_13
        mul_9 = mul_8 * x_13
        mul_8 = x_13 = None
        add_6 = 1.0 + mul_9
        mul_9 = None
        mul_10 = mul_7 * add_6
        mul_7 = add_6 = None
        tanh_1 = torch.tanh(mul_10)
        mul_10 = None
        add_7 = 1.0 + tanh_1
        tanh_1 = None
        hidden_states_11 = mul_6 * add_7
        mul_6 = add_7 = None
        view_22 = hidden_states_11.view(-1, 8192)
        hidden_states_11 = None
        x_14 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_,
            view_22,
            l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_22
        ) = l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_15 = x_14.view((1, 39, 2048))
        x_14 = None
        hidden_states_12 = torch.nn.functional.dropout(x_15, 0.1, False, False)
        x_15 = None
        hidden_states_13 = hidden_states_9 + hidden_states_12
        hidden_states_9 = hidden_states_12 = None
        hidden_states_14 = torch.nn.functional.layer_norm(
            hidden_states_13,
            (2048,),
            l_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_bias_
        ) = None
        view_24 = hidden_states_14.view(-1, 2048)
        hidden_states_14 = None
        x_16 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_,
            view_24,
            l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_ = (
            view_24
        ) = l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_17 = x_16.view((1, 39, 6144))
        x_16 = None
        split_2 = x_17.split(2048, dim=2)
        x_17 = None
        query_states_4 = split_2[0]
        key_states_4 = split_2[1]
        value_states_4 = split_2[2]
        split_2 = None
        view_26 = key_states_4.view((1, 39, -1, 128))
        key_states_4 = None
        key_states_5 = view_26.transpose(1, 2)
        view_26 = None
        view_27 = value_states_4.view((1, 39, -1, 128))
        value_states_4 = None
        value_states_5 = view_27.transpose(1, 2)
        view_27 = None
        view_28 = query_states_4.view((1, 39, -1, 128))
        query_states_4 = None
        query_states_5 = view_28.transpose(1, 2)
        view_28 = None
        attention_mask_4 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 39, None),
            )
        ]
        query_2 = query_states_5.contiguous()
        query_states_5 = None
        key_2 = key_states_5.contiguous()
        key_states_5 = None
        value_2 = value_states_5.contiguous()
        value_states_5 = None
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
        reshape_2 = attn_output_9.reshape(1, 39, -1)
        attn_output_9 = None
        attn_output_10 = reshape_2.contiguous()
        reshape_2 = None
        view_29 = attn_output_10.view(-1, 2048)
        attn_output_10 = None
        x_18 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_,
            view_29,
            l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_ = (
            view_29
        ) = l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_19 = x_18.view((1, 39, 2048))
        x_18 = None
        attn_output_11 = torch.nn.functional.dropout(x_19, 0.1, False, False)
        x_19 = None
        hidden_states_15 = attn_output_11 + hidden_states_13
        attn_output_11 = hidden_states_13 = None
        hidden_states_16 = torch.nn.functional.layer_norm(
            hidden_states_15,
            (2048,),
            l_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_bias_
        ) = None
        view_31 = hidden_states_16.view(-1, 2048)
        hidden_states_16 = None
        x_20 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_,
            view_31,
            l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_31
        ) = l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_21 = x_20.view((1, 39, 8192))
        x_20 = None
        mul_12 = 0.5 * x_21
        mul_13 = x_21 * 0.7978845608
        mul_14 = 0.044715 * x_21
        mul_15 = mul_14 * x_21
        mul_14 = x_21 = None
        add_10 = 1.0 + mul_15
        mul_15 = None
        mul_16 = mul_13 * add_10
        mul_13 = add_10 = None
        tanh_2 = torch.tanh(mul_16)
        mul_16 = None
        add_11 = 1.0 + tanh_2
        tanh_2 = None
        hidden_states_17 = mul_12 * add_11
        mul_12 = add_11 = None
        view_33 = hidden_states_17.view(-1, 8192)
        hidden_states_17 = None
        x_22 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_,
            view_33,
            l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_33
        ) = l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_23 = x_22.view((1, 39, 2048))
        x_22 = None
        hidden_states_18 = torch.nn.functional.dropout(x_23, 0.1, False, False)
        x_23 = None
        hidden_states_19 = hidden_states_15 + hidden_states_18
        hidden_states_15 = hidden_states_18 = None
        hidden_states_20 = torch.nn.functional.layer_norm(
            hidden_states_19,
            (2048,),
            l_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_bias_
        ) = None
        view_35 = hidden_states_20.view(-1, 2048)
        hidden_states_20 = None
        x_24 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_,
            view_35,
            l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_ = (
            view_35
        ) = l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_25 = x_24.view((1, 39, 6144))
        x_24 = None
        split_3 = x_25.split(2048, dim=2)
        x_25 = None
        query_states_6 = split_3[0]
        key_states_6 = split_3[1]
        value_states_6 = split_3[2]
        split_3 = None
        view_37 = key_states_6.view((1, 39, -1, 128))
        key_states_6 = None
        key_states_7 = view_37.transpose(1, 2)
        view_37 = None
        view_38 = value_states_6.view((1, 39, -1, 128))
        value_states_6 = None
        value_states_7 = view_38.transpose(1, 2)
        view_38 = None
        view_39 = query_states_6.view((1, 39, -1, 128))
        query_states_6 = None
        query_states_7 = view_39.transpose(1, 2)
        view_39 = None
        attention_mask_5 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 39, None),
            )
        ]
        query_3 = query_states_7.contiguous()
        query_states_7 = None
        key_3 = key_states_7.contiguous()
        key_states_7 = None
        value_3 = value_states_7.contiguous()
        value_states_7 = None
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
        reshape_3 = attn_output_13.reshape(1, 39, -1)
        attn_output_13 = None
        attn_output_14 = reshape_3.contiguous()
        reshape_3 = None
        view_40 = attn_output_14.view(-1, 2048)
        attn_output_14 = None
        x_26 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_,
            view_40,
            l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_ = (
            view_40
        ) = l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_27 = x_26.view((1, 39, 2048))
        x_26 = None
        attn_output_15 = torch.nn.functional.dropout(x_27, 0.1, False, False)
        x_27 = None
        hidden_states_21 = attn_output_15 + hidden_states_19
        attn_output_15 = hidden_states_19 = None
        hidden_states_22 = torch.nn.functional.layer_norm(
            hidden_states_21,
            (2048,),
            l_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_bias_
        ) = None
        view_42 = hidden_states_22.view(-1, 2048)
        hidden_states_22 = None
        x_28 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_,
            view_42,
            l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_42
        ) = l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_29 = x_28.view((1, 39, 8192))
        x_28 = None
        mul_18 = 0.5 * x_29
        mul_19 = x_29 * 0.7978845608
        mul_20 = 0.044715 * x_29
        mul_21 = mul_20 * x_29
        mul_20 = x_29 = None
        add_14 = 1.0 + mul_21
        mul_21 = None
        mul_22 = mul_19 * add_14
        mul_19 = add_14 = None
        tanh_3 = torch.tanh(mul_22)
        mul_22 = None
        add_15 = 1.0 + tanh_3
        tanh_3 = None
        hidden_states_23 = mul_18 * add_15
        mul_18 = add_15 = None
        view_44 = hidden_states_23.view(-1, 8192)
        hidden_states_23 = None
        x_30 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_,
            view_44,
            l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_44
        ) = l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_31 = x_30.view((1, 39, 2048))
        x_30 = None
        hidden_states_24 = torch.nn.functional.dropout(x_31, 0.1, False, False)
        x_31 = None
        hidden_states_25 = hidden_states_21 + hidden_states_24
        hidden_states_21 = hidden_states_24 = None
        hidden_states_26 = torch.nn.functional.layer_norm(
            hidden_states_25,
            (2048,),
            l_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_bias_
        ) = None
        view_46 = hidden_states_26.view(-1, 2048)
        hidden_states_26 = None
        x_32 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_,
            view_46,
            l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_ = (
            view_46
        ) = l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_33 = x_32.view((1, 39, 6144))
        x_32 = None
        split_4 = x_33.split(2048, dim=2)
        x_33 = None
        query_states_8 = split_4[0]
        key_states_8 = split_4[1]
        value_states_8 = split_4[2]
        split_4 = None
        view_48 = key_states_8.view((1, 39, -1, 128))
        key_states_8 = None
        key_states_9 = view_48.transpose(1, 2)
        view_48 = None
        view_49 = value_states_8.view((1, 39, -1, 128))
        value_states_8 = None
        value_states_9 = view_49.transpose(1, 2)
        view_49 = None
        view_50 = query_states_8.view((1, 39, -1, 128))
        query_states_8 = None
        query_states_9 = view_50.transpose(1, 2)
        view_50 = None
        attention_mask_6 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 39, None),
            )
        ]
        query_4 = query_states_9.contiguous()
        query_states_9 = None
        key_4 = key_states_9.contiguous()
        key_states_9 = None
        value_4 = value_states_9.contiguous()
        value_states_9 = None
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
        reshape_4 = attn_output_17.reshape(1, 39, -1)
        attn_output_17 = None
        attn_output_18 = reshape_4.contiguous()
        reshape_4 = None
        view_51 = attn_output_18.view(-1, 2048)
        attn_output_18 = None
        x_34 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_,
            view_51,
            l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_ = (
            view_51
        ) = l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_35 = x_34.view((1, 39, 2048))
        x_34 = None
        attn_output_19 = torch.nn.functional.dropout(x_35, 0.1, False, False)
        x_35 = None
        hidden_states_27 = attn_output_19 + hidden_states_25
        attn_output_19 = hidden_states_25 = None
        hidden_states_28 = torch.nn.functional.layer_norm(
            hidden_states_27,
            (2048,),
            l_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_bias_
        ) = None
        view_53 = hidden_states_28.view(-1, 2048)
        hidden_states_28 = None
        x_36 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_,
            view_53,
            l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_53
        ) = l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_37 = x_36.view((1, 39, 8192))
        x_36 = None
        mul_24 = 0.5 * x_37
        mul_25 = x_37 * 0.7978845608
        mul_26 = 0.044715 * x_37
        mul_27 = mul_26 * x_37
        mul_26 = x_37 = None
        add_18 = 1.0 + mul_27
        mul_27 = None
        mul_28 = mul_25 * add_18
        mul_25 = add_18 = None
        tanh_4 = torch.tanh(mul_28)
        mul_28 = None
        add_19 = 1.0 + tanh_4
        tanh_4 = None
        hidden_states_29 = mul_24 * add_19
        mul_24 = add_19 = None
        view_55 = hidden_states_29.view(-1, 8192)
        hidden_states_29 = None
        x_38 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_,
            view_55,
            l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_55
        ) = l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_39 = x_38.view((1, 39, 2048))
        x_38 = None
        hidden_states_30 = torch.nn.functional.dropout(x_39, 0.1, False, False)
        x_39 = None
        hidden_states_31 = hidden_states_27 + hidden_states_30
        hidden_states_27 = hidden_states_30 = None
        hidden_states_32 = torch.nn.functional.layer_norm(
            hidden_states_31,
            (2048,),
            l_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_bias_
        ) = None
        view_57 = hidden_states_32.view(-1, 2048)
        hidden_states_32 = None
        x_40 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_bias_,
            view_57,
            l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_bias_ = (
            view_57
        ) = l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_41 = x_40.view((1, 39, 6144))
        x_40 = None
        split_5 = x_41.split(2048, dim=2)
        x_41 = None
        query_states_10 = split_5[0]
        key_states_10 = split_5[1]
        value_states_10 = split_5[2]
        split_5 = None
        view_59 = key_states_10.view((1, 39, -1, 128))
        key_states_10 = None
        key_states_11 = view_59.transpose(1, 2)
        view_59 = None
        view_60 = value_states_10.view((1, 39, -1, 128))
        value_states_10 = None
        value_states_11 = view_60.transpose(1, 2)
        view_60 = None
        view_61 = query_states_10.view((1, 39, -1, 128))
        query_states_10 = None
        query_states_11 = view_61.transpose(1, 2)
        view_61 = None
        attention_mask_7 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 39, None),
            )
        ]
        query_5 = query_states_11.contiguous()
        query_states_11 = None
        key_5 = key_states_11.contiguous()
        key_states_11 = None
        value_5 = value_states_11.contiguous()
        value_states_11 = None
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
        reshape_5 = attn_output_21.reshape(1, 39, -1)
        attn_output_21 = None
        attn_output_22 = reshape_5.contiguous()
        reshape_5 = None
        view_62 = attn_output_22.view(-1, 2048)
        attn_output_22 = None
        x_42 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_bias_,
            view_62,
            l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_bias_ = (
            view_62
        ) = l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_43 = x_42.view((1, 39, 2048))
        x_42 = None
        attn_output_23 = torch.nn.functional.dropout(x_43, 0.1, False, False)
        x_43 = None
        hidden_states_33 = attn_output_23 + hidden_states_31
        attn_output_23 = hidden_states_31 = None
        hidden_states_34 = torch.nn.functional.layer_norm(
            hidden_states_33,
            (2048,),
            l_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_bias_
        ) = None
        view_64 = hidden_states_34.view(-1, 2048)
        hidden_states_34 = None
        x_44 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_bias_,
            view_64,
            l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_64
        ) = l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_45 = x_44.view((1, 39, 8192))
        x_44 = None
        mul_30 = 0.5 * x_45
        mul_31 = x_45 * 0.7978845608
        mul_32 = 0.044715 * x_45
        mul_33 = mul_32 * x_45
        mul_32 = x_45 = None
        add_22 = 1.0 + mul_33
        mul_33 = None
        mul_34 = mul_31 * add_22
        mul_31 = add_22 = None
        tanh_5 = torch.tanh(mul_34)
        mul_34 = None
        add_23 = 1.0 + tanh_5
        tanh_5 = None
        hidden_states_35 = mul_30 * add_23
        mul_30 = add_23 = None
        view_66 = hidden_states_35.view(-1, 8192)
        hidden_states_35 = None
        x_46 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_bias_,
            view_66,
            l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_66
        ) = l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_47 = x_46.view((1, 39, 2048))
        x_46 = None
        hidden_states_36 = torch.nn.functional.dropout(x_47, 0.1, False, False)
        x_47 = None
        hidden_states_37 = hidden_states_33 + hidden_states_36
        hidden_states_33 = hidden_states_36 = None
        hidden_states_38 = torch.nn.functional.layer_norm(
            hidden_states_37,
            (2048,),
            l_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_bias_
        ) = None
        view_68 = hidden_states_38.view(-1, 2048)
        hidden_states_38 = None
        x_48 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_attn_parameters_bias_,
            view_68,
            l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_attn_parameters_bias_ = (
            view_68
        ) = l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_49 = x_48.view((1, 39, 6144))
        x_48 = None
        split_6 = x_49.split(2048, dim=2)
        x_49 = None
        query_states_12 = split_6[0]
        key_states_12 = split_6[1]
        value_states_12 = split_6[2]
        split_6 = None
        view_70 = key_states_12.view((1, 39, -1, 128))
        key_states_12 = None
        key_states_13 = view_70.transpose(1, 2)
        view_70 = None
        view_71 = value_states_12.view((1, 39, -1, 128))
        value_states_12 = None
        value_states_13 = view_71.transpose(1, 2)
        view_71 = None
        view_72 = query_states_12.view((1, 39, -1, 128))
        query_states_12 = None
        query_states_13 = view_72.transpose(1, 2)
        view_72 = None
        attention_mask_8 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 39, None),
            )
        ]
        query_6 = query_states_13.contiguous()
        query_states_13 = None
        key_6 = key_states_13.contiguous()
        key_states_13 = None
        value_6 = value_states_13.contiguous()
        value_states_13 = None
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
        reshape_6 = attn_output_25.reshape(1, 39, -1)
        attn_output_25 = None
        attn_output_26 = reshape_6.contiguous()
        reshape_6 = None
        view_73 = attn_output_26.view(-1, 2048)
        attn_output_26 = None
        x_50 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_proj_parameters_bias_,
            view_73,
            l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_proj_parameters_bias_ = (
            view_73
        ) = l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_51 = x_50.view((1, 39, 2048))
        x_50 = None
        attn_output_27 = torch.nn.functional.dropout(x_51, 0.1, False, False)
        x_51 = None
        hidden_states_39 = attn_output_27 + hidden_states_37
        attn_output_27 = hidden_states_37 = None
        hidden_states_40 = torch.nn.functional.layer_norm(
            hidden_states_39,
            (2048,),
            l_self_modules_transformer_modules_h_modules_6_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_6_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_6_modules_ln_2_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_6_modules_ln_2_parameters_bias_
        ) = None
        view_75 = hidden_states_40.view(-1, 2048)
        hidden_states_40 = None
        x_52 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_bias_,
            view_75,
            l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_75
        ) = l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_53 = x_52.view((1, 39, 8192))
        x_52 = None
        mul_36 = 0.5 * x_53
        mul_37 = x_53 * 0.7978845608
        mul_38 = 0.044715 * x_53
        mul_39 = mul_38 * x_53
        mul_38 = x_53 = None
        add_26 = 1.0 + mul_39
        mul_39 = None
        mul_40 = mul_37 * add_26
        mul_37 = add_26 = None
        tanh_6 = torch.tanh(mul_40)
        mul_40 = None
        add_27 = 1.0 + tanh_6
        tanh_6 = None
        hidden_states_41 = mul_36 * add_27
        mul_36 = add_27 = None
        view_77 = hidden_states_41.view(-1, 8192)
        hidden_states_41 = None
        x_54 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_bias_,
            view_77,
            l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_77
        ) = l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_55 = x_54.view((1, 39, 2048))
        x_54 = None
        hidden_states_42 = torch.nn.functional.dropout(x_55, 0.1, False, False)
        x_55 = None
        hidden_states_43 = hidden_states_39 + hidden_states_42
        hidden_states_39 = hidden_states_42 = None
        hidden_states_44 = torch.nn.functional.layer_norm(
            hidden_states_43,
            (2048,),
            l_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_bias_
        ) = None
        view_79 = hidden_states_44.view(-1, 2048)
        hidden_states_44 = None
        x_56 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_attn_parameters_bias_,
            view_79,
            l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_attn_parameters_bias_ = (
            view_79
        ) = l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_57 = x_56.view((1, 39, 6144))
        x_56 = None
        split_7 = x_57.split(2048, dim=2)
        x_57 = None
        query_states_14 = split_7[0]
        key_states_14 = split_7[1]
        value_states_14 = split_7[2]
        split_7 = None
        view_81 = key_states_14.view((1, 39, -1, 128))
        key_states_14 = None
        key_states_15 = view_81.transpose(1, 2)
        view_81 = None
        view_82 = value_states_14.view((1, 39, -1, 128))
        value_states_14 = None
        value_states_15 = view_82.transpose(1, 2)
        view_82 = None
        view_83 = query_states_14.view((1, 39, -1, 128))
        query_states_14 = None
        query_states_15 = view_83.transpose(1, 2)
        view_83 = None
        attention_mask_9 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 39, None),
            )
        ]
        query_7 = query_states_15.contiguous()
        query_states_15 = None
        key_7 = key_states_15.contiguous()
        key_states_15 = None
        value_7 = value_states_15.contiguous()
        value_states_15 = None
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
        reshape_7 = attn_output_29.reshape(1, 39, -1)
        attn_output_29 = None
        attn_output_30 = reshape_7.contiguous()
        reshape_7 = None
        view_84 = attn_output_30.view(-1, 2048)
        attn_output_30 = None
        x_58 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_proj_parameters_bias_,
            view_84,
            l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_proj_parameters_bias_ = (
            view_84
        ) = l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_59 = x_58.view((1, 39, 2048))
        x_58 = None
        attn_output_31 = torch.nn.functional.dropout(x_59, 0.1, False, False)
        x_59 = None
        hidden_states_45 = attn_output_31 + hidden_states_43
        attn_output_31 = hidden_states_43 = None
        hidden_states_46 = torch.nn.functional.layer_norm(
            hidden_states_45,
            (2048,),
            l_self_modules_transformer_modules_h_modules_7_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_7_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_7_modules_ln_2_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_7_modules_ln_2_parameters_bias_
        ) = None
        view_86 = hidden_states_46.view(-1, 2048)
        hidden_states_46 = None
        x_60 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_bias_,
            view_86,
            l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_86
        ) = l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_61 = x_60.view((1, 39, 8192))
        x_60 = None
        mul_42 = 0.5 * x_61
        mul_43 = x_61 * 0.7978845608
        mul_44 = 0.044715 * x_61
        mul_45 = mul_44 * x_61
        mul_44 = x_61 = None
        add_30 = 1.0 + mul_45
        mul_45 = None
        mul_46 = mul_43 * add_30
        mul_43 = add_30 = None
        tanh_7 = torch.tanh(mul_46)
        mul_46 = None
        add_31 = 1.0 + tanh_7
        tanh_7 = None
        hidden_states_47 = mul_42 * add_31
        mul_42 = add_31 = None
        view_88 = hidden_states_47.view(-1, 8192)
        hidden_states_47 = None
        x_62 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_bias_,
            view_88,
            l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_88
        ) = l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_63 = x_62.view((1, 39, 2048))
        x_62 = None
        hidden_states_48 = torch.nn.functional.dropout(x_63, 0.1, False, False)
        x_63 = None
        hidden_states_49 = hidden_states_45 + hidden_states_48
        hidden_states_45 = hidden_states_48 = None
        hidden_states_50 = torch.nn.functional.layer_norm(
            hidden_states_49,
            (2048,),
            l_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_bias_
        ) = None
        view_90 = hidden_states_50.view(-1, 2048)
        hidden_states_50 = None
        x_64 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_attn_parameters_bias_,
            view_90,
            l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_attn_parameters_bias_ = (
            view_90
        ) = l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_65 = x_64.view((1, 39, 6144))
        x_64 = None
        split_8 = x_65.split(2048, dim=2)
        x_65 = None
        query_states_16 = split_8[0]
        key_states_16 = split_8[1]
        value_states_16 = split_8[2]
        split_8 = None
        view_92 = key_states_16.view((1, 39, -1, 128))
        key_states_16 = None
        key_states_17 = view_92.transpose(1, 2)
        view_92 = None
        view_93 = value_states_16.view((1, 39, -1, 128))
        value_states_16 = None
        value_states_17 = view_93.transpose(1, 2)
        view_93 = None
        view_94 = query_states_16.view((1, 39, -1, 128))
        query_states_16 = None
        query_states_17 = view_94.transpose(1, 2)
        view_94 = None
        attention_mask_10 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 39, None),
            )
        ]
        query_8 = query_states_17.contiguous()
        query_states_17 = None
        key_8 = key_states_17.contiguous()
        key_states_17 = None
        value_8 = value_states_17.contiguous()
        value_states_17 = None
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
        reshape_8 = attn_output_33.reshape(1, 39, -1)
        attn_output_33 = None
        attn_output_34 = reshape_8.contiguous()
        reshape_8 = None
        view_95 = attn_output_34.view(-1, 2048)
        attn_output_34 = None
        x_66 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_proj_parameters_bias_,
            view_95,
            l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_proj_parameters_bias_ = (
            view_95
        ) = l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_67 = x_66.view((1, 39, 2048))
        x_66 = None
        attn_output_35 = torch.nn.functional.dropout(x_67, 0.1, False, False)
        x_67 = None
        hidden_states_51 = attn_output_35 + hidden_states_49
        attn_output_35 = hidden_states_49 = None
        hidden_states_52 = torch.nn.functional.layer_norm(
            hidden_states_51,
            (2048,),
            l_self_modules_transformer_modules_h_modules_8_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_8_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_8_modules_ln_2_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_8_modules_ln_2_parameters_bias_
        ) = None
        view_97 = hidden_states_52.view(-1, 2048)
        hidden_states_52 = None
        x_68 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_bias_,
            view_97,
            l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_97
        ) = l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_69 = x_68.view((1, 39, 8192))
        x_68 = None
        mul_48 = 0.5 * x_69
        mul_49 = x_69 * 0.7978845608
        mul_50 = 0.044715 * x_69
        mul_51 = mul_50 * x_69
        mul_50 = x_69 = None
        add_34 = 1.0 + mul_51
        mul_51 = None
        mul_52 = mul_49 * add_34
        mul_49 = add_34 = None
        tanh_8 = torch.tanh(mul_52)
        mul_52 = None
        add_35 = 1.0 + tanh_8
        tanh_8 = None
        hidden_states_53 = mul_48 * add_35
        mul_48 = add_35 = None
        view_99 = hidden_states_53.view(-1, 8192)
        hidden_states_53 = None
        x_70 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_bias_,
            view_99,
            l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_99
        ) = l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_71 = x_70.view((1, 39, 2048))
        x_70 = None
        hidden_states_54 = torch.nn.functional.dropout(x_71, 0.1, False, False)
        x_71 = None
        hidden_states_55 = hidden_states_51 + hidden_states_54
        hidden_states_51 = hidden_states_54 = None
        hidden_states_56 = torch.nn.functional.layer_norm(
            hidden_states_55,
            (2048,),
            l_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_bias_
        ) = None
        view_101 = hidden_states_56.view(-1, 2048)
        hidden_states_56 = None
        x_72 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_attn_parameters_bias_,
            view_101,
            l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_attn_parameters_bias_ = (
            view_101
        ) = l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_73 = x_72.view((1, 39, 6144))
        x_72 = None
        split_9 = x_73.split(2048, dim=2)
        x_73 = None
        query_states_18 = split_9[0]
        key_states_18 = split_9[1]
        value_states_18 = split_9[2]
        split_9 = None
        view_103 = key_states_18.view((1, 39, -1, 128))
        key_states_18 = None
        key_states_19 = view_103.transpose(1, 2)
        view_103 = None
        view_104 = value_states_18.view((1, 39, -1, 128))
        value_states_18 = None
        value_states_19 = view_104.transpose(1, 2)
        view_104 = None
        view_105 = query_states_18.view((1, 39, -1, 128))
        query_states_18 = None
        query_states_19 = view_105.transpose(1, 2)
        view_105 = None
        attention_mask_11 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 39, None),
            )
        ]
        query_9 = query_states_19.contiguous()
        query_states_19 = None
        key_9 = key_states_19.contiguous()
        key_states_19 = None
        value_9 = value_states_19.contiguous()
        value_states_19 = None
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
        reshape_9 = attn_output_37.reshape(1, 39, -1)
        attn_output_37 = None
        attn_output_38 = reshape_9.contiguous()
        reshape_9 = None
        view_106 = attn_output_38.view(-1, 2048)
        attn_output_38 = None
        x_74 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_proj_parameters_bias_,
            view_106,
            l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_proj_parameters_bias_ = (
            view_106
        ) = l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_75 = x_74.view((1, 39, 2048))
        x_74 = None
        attn_output_39 = torch.nn.functional.dropout(x_75, 0.1, False, False)
        x_75 = None
        hidden_states_57 = attn_output_39 + hidden_states_55
        attn_output_39 = hidden_states_55 = None
        hidden_states_58 = torch.nn.functional.layer_norm(
            hidden_states_57,
            (2048,),
            l_self_modules_transformer_modules_h_modules_9_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_9_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_9_modules_ln_2_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_9_modules_ln_2_parameters_bias_
        ) = None
        view_108 = hidden_states_58.view(-1, 2048)
        hidden_states_58 = None
        x_76 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_bias_,
            view_108,
            l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_108
        ) = l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_77 = x_76.view((1, 39, 8192))
        x_76 = None
        mul_54 = 0.5 * x_77
        mul_55 = x_77 * 0.7978845608
        mul_56 = 0.044715 * x_77
        mul_57 = mul_56 * x_77
        mul_56 = x_77 = None
        add_38 = 1.0 + mul_57
        mul_57 = None
        mul_58 = mul_55 * add_38
        mul_55 = add_38 = None
        tanh_9 = torch.tanh(mul_58)
        mul_58 = None
        add_39 = 1.0 + tanh_9
        tanh_9 = None
        hidden_states_59 = mul_54 * add_39
        mul_54 = add_39 = None
        view_110 = hidden_states_59.view(-1, 8192)
        hidden_states_59 = None
        x_78 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_bias_,
            view_110,
            l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_110
        ) = l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_79 = x_78.view((1, 39, 2048))
        x_78 = None
        hidden_states_60 = torch.nn.functional.dropout(x_79, 0.1, False, False)
        x_79 = None
        hidden_states_61 = hidden_states_57 + hidden_states_60
        hidden_states_57 = hidden_states_60 = None
        hidden_states_62 = torch.nn.functional.layer_norm(
            hidden_states_61,
            (2048,),
            l_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_bias_ = (None)
        view_112 = hidden_states_62.view(-1, 2048)
        hidden_states_62 = None
        x_80 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_attn_parameters_bias_,
            view_112,
            l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_attn_parameters_bias_ = (
            view_112
        ) = l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_81 = x_80.view((1, 39, 6144))
        x_80 = None
        split_10 = x_81.split(2048, dim=2)
        x_81 = None
        query_states_20 = split_10[0]
        key_states_20 = split_10[1]
        value_states_20 = split_10[2]
        split_10 = None
        view_114 = key_states_20.view((1, 39, -1, 128))
        key_states_20 = None
        key_states_21 = view_114.transpose(1, 2)
        view_114 = None
        view_115 = value_states_20.view((1, 39, -1, 128))
        value_states_20 = None
        value_states_21 = view_115.transpose(1, 2)
        view_115 = None
        view_116 = query_states_20.view((1, 39, -1, 128))
        query_states_20 = None
        query_states_21 = view_116.transpose(1, 2)
        view_116 = None
        attention_mask_12 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 39, None),
            )
        ]
        query_10 = query_states_21.contiguous()
        query_states_21 = None
        key_10 = key_states_21.contiguous()
        key_states_21 = None
        value_10 = value_states_21.contiguous()
        value_states_21 = None
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
        reshape_10 = attn_output_41.reshape(1, 39, -1)
        attn_output_41 = None
        attn_output_42 = reshape_10.contiguous()
        reshape_10 = None
        view_117 = attn_output_42.view(-1, 2048)
        attn_output_42 = None
        x_82 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_proj_parameters_bias_,
            view_117,
            l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_proj_parameters_bias_ = (
            view_117
        ) = l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_83 = x_82.view((1, 39, 2048))
        x_82 = None
        attn_output_43 = torch.nn.functional.dropout(x_83, 0.1, False, False)
        x_83 = None
        hidden_states_63 = attn_output_43 + hidden_states_61
        attn_output_43 = hidden_states_61 = None
        hidden_states_64 = torch.nn.functional.layer_norm(
            hidden_states_63,
            (2048,),
            l_self_modules_transformer_modules_h_modules_10_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_10_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_10_modules_ln_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_10_modules_ln_2_parameters_bias_ = (None)
        view_119 = hidden_states_64.view(-1, 2048)
        hidden_states_64 = None
        x_84 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_bias_,
            view_119,
            l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_119
        ) = l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_85 = x_84.view((1, 39, 8192))
        x_84 = None
        mul_60 = 0.5 * x_85
        mul_61 = x_85 * 0.7978845608
        mul_62 = 0.044715 * x_85
        mul_63 = mul_62 * x_85
        mul_62 = x_85 = None
        add_42 = 1.0 + mul_63
        mul_63 = None
        mul_64 = mul_61 * add_42
        mul_61 = add_42 = None
        tanh_10 = torch.tanh(mul_64)
        mul_64 = None
        add_43 = 1.0 + tanh_10
        tanh_10 = None
        hidden_states_65 = mul_60 * add_43
        mul_60 = add_43 = None
        view_121 = hidden_states_65.view(-1, 8192)
        hidden_states_65 = None
        x_86 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_bias_,
            view_121,
            l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_121
        ) = l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_87 = x_86.view((1, 39, 2048))
        x_86 = None
        hidden_states_66 = torch.nn.functional.dropout(x_87, 0.1, False, False)
        x_87 = None
        hidden_states_67 = hidden_states_63 + hidden_states_66
        hidden_states_63 = hidden_states_66 = None
        hidden_states_68 = torch.nn.functional.layer_norm(
            hidden_states_67,
            (2048,),
            l_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_bias_ = (None)
        view_123 = hidden_states_68.view(-1, 2048)
        hidden_states_68 = None
        x_88 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_attn_parameters_bias_,
            view_123,
            l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_attn_parameters_bias_ = (
            view_123
        ) = l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_89 = x_88.view((1, 39, 6144))
        x_88 = None
        split_11 = x_89.split(2048, dim=2)
        x_89 = None
        query_states_22 = split_11[0]
        key_states_22 = split_11[1]
        value_states_22 = split_11[2]
        split_11 = None
        view_125 = key_states_22.view((1, 39, -1, 128))
        key_states_22 = None
        key_states_23 = view_125.transpose(1, 2)
        view_125 = None
        view_126 = value_states_22.view((1, 39, -1, 128))
        value_states_22 = None
        value_states_23 = view_126.transpose(1, 2)
        view_126 = None
        view_127 = query_states_22.view((1, 39, -1, 128))
        query_states_22 = None
        query_states_23 = view_127.transpose(1, 2)
        view_127 = None
        attention_mask_13 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 39, None),
            )
        ]
        query_11 = query_states_23.contiguous()
        query_states_23 = None
        key_11 = key_states_23.contiguous()
        key_states_23 = None
        value_11 = value_states_23.contiguous()
        value_states_23 = None
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
        reshape_11 = attn_output_45.reshape(1, 39, -1)
        attn_output_45 = None
        attn_output_46 = reshape_11.contiguous()
        reshape_11 = None
        view_128 = attn_output_46.view(-1, 2048)
        attn_output_46 = None
        x_90 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_proj_parameters_bias_,
            view_128,
            l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_proj_parameters_bias_ = (
            view_128
        ) = l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_91 = x_90.view((1, 39, 2048))
        x_90 = None
        attn_output_47 = torch.nn.functional.dropout(x_91, 0.1, False, False)
        x_91 = None
        hidden_states_69 = attn_output_47 + hidden_states_67
        attn_output_47 = hidden_states_67 = None
        hidden_states_70 = torch.nn.functional.layer_norm(
            hidden_states_69,
            (2048,),
            l_self_modules_transformer_modules_h_modules_11_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_11_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_11_modules_ln_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_11_modules_ln_2_parameters_bias_ = (None)
        view_130 = hidden_states_70.view(-1, 2048)
        hidden_states_70 = None
        x_92 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_bias_,
            view_130,
            l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_130
        ) = l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_93 = x_92.view((1, 39, 8192))
        x_92 = None
        mul_66 = 0.5 * x_93
        mul_67 = x_93 * 0.7978845608
        mul_68 = 0.044715 * x_93
        mul_69 = mul_68 * x_93
        mul_68 = x_93 = None
        add_46 = 1.0 + mul_69
        mul_69 = None
        mul_70 = mul_67 * add_46
        mul_67 = add_46 = None
        tanh_11 = torch.tanh(mul_70)
        mul_70 = None
        add_47 = 1.0 + tanh_11
        tanh_11 = None
        hidden_states_71 = mul_66 * add_47
        mul_66 = add_47 = None
        view_132 = hidden_states_71.view(-1, 8192)
        hidden_states_71 = None
        x_94 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_bias_,
            view_132,
            l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_132
        ) = l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_95 = x_94.view((1, 39, 2048))
        x_94 = None
        hidden_states_72 = torch.nn.functional.dropout(x_95, 0.1, False, False)
        x_95 = None
        hidden_states_73 = hidden_states_69 + hidden_states_72
        hidden_states_69 = hidden_states_72 = None
        hidden_states_74 = torch.nn.functional.layer_norm(
            hidden_states_73,
            (2048,),
            l_self_modules_transformer_modules_h_modules_12_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_12_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_12_modules_ln_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_12_modules_ln_1_parameters_bias_ = (None)
        view_134 = hidden_states_74.view(-1, 2048)
        hidden_states_74 = None
        x_96 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_attn_parameters_bias_,
            view_134,
            l_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_attn_parameters_bias_ = (
            view_134
        ) = l_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_97 = x_96.view((1, 39, 6144))
        x_96 = None
        split_12 = x_97.split(2048, dim=2)
        x_97 = None
        query_states_24 = split_12[0]
        key_states_24 = split_12[1]
        value_states_24 = split_12[2]
        split_12 = None
        view_136 = key_states_24.view((1, 39, -1, 128))
        key_states_24 = None
        key_states_25 = view_136.transpose(1, 2)
        view_136 = None
        view_137 = value_states_24.view((1, 39, -1, 128))
        value_states_24 = None
        value_states_25 = view_137.transpose(1, 2)
        view_137 = None
        view_138 = query_states_24.view((1, 39, -1, 128))
        query_states_24 = None
        query_states_25 = view_138.transpose(1, 2)
        view_138 = None
        attention_mask_14 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 39, None),
            )
        ]
        query_12 = query_states_25.contiguous()
        query_states_25 = None
        key_12 = key_states_25.contiguous()
        key_states_25 = None
        value_12 = value_states_25.contiguous()
        value_states_25 = None
        attn_output_48 = torch._C._nn.scaled_dot_product_attention(
            query_12,
            key_12,
            value_12,
            attn_mask=attention_mask_14,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query_12 = key_12 = value_12 = attention_mask_14 = None
        transpose_51 = attn_output_48.transpose(1, 2)
        attn_output_48 = None
        attn_output_49 = transpose_51.contiguous()
        transpose_51 = None
        reshape_12 = attn_output_49.reshape(1, 39, -1)
        attn_output_49 = None
        attn_output_50 = reshape_12.contiguous()
        reshape_12 = None
        view_139 = attn_output_50.view(-1, 2048)
        attn_output_50 = None
        x_98 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_proj_parameters_bias_,
            view_139,
            l_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_proj_parameters_bias_ = (
            view_139
        ) = l_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_99 = x_98.view((1, 39, 2048))
        x_98 = None
        attn_output_51 = torch.nn.functional.dropout(x_99, 0.1, False, False)
        x_99 = None
        hidden_states_75 = attn_output_51 + hidden_states_73
        attn_output_51 = hidden_states_73 = None
        hidden_states_76 = torch.nn.functional.layer_norm(
            hidden_states_75,
            (2048,),
            l_self_modules_transformer_modules_h_modules_12_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_12_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_12_modules_ln_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_12_modules_ln_2_parameters_bias_ = (None)
        view_141 = hidden_states_76.view(-1, 2048)
        hidden_states_76 = None
        x_100 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_fc_parameters_bias_,
            view_141,
            l_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_141
        ) = l_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_101 = x_100.view((1, 39, 8192))
        x_100 = None
        mul_72 = 0.5 * x_101
        mul_73 = x_101 * 0.7978845608
        mul_74 = 0.044715 * x_101
        mul_75 = mul_74 * x_101
        mul_74 = x_101 = None
        add_50 = 1.0 + mul_75
        mul_75 = None
        mul_76 = mul_73 * add_50
        mul_73 = add_50 = None
        tanh_12 = torch.tanh(mul_76)
        mul_76 = None
        add_51 = 1.0 + tanh_12
        tanh_12 = None
        hidden_states_77 = mul_72 * add_51
        mul_72 = add_51 = None
        view_143 = hidden_states_77.view(-1, 8192)
        hidden_states_77 = None
        x_102 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_proj_parameters_bias_,
            view_143,
            l_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_143
        ) = l_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_103 = x_102.view((1, 39, 2048))
        x_102 = None
        hidden_states_78 = torch.nn.functional.dropout(x_103, 0.1, False, False)
        x_103 = None
        hidden_states_79 = hidden_states_75 + hidden_states_78
        hidden_states_75 = hidden_states_78 = None
        hidden_states_80 = torch.nn.functional.layer_norm(
            hidden_states_79,
            (2048,),
            l_self_modules_transformer_modules_h_modules_13_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_13_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_13_modules_ln_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_13_modules_ln_1_parameters_bias_ = (None)
        view_145 = hidden_states_80.view(-1, 2048)
        hidden_states_80 = None
        x_104 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_attn_parameters_bias_,
            view_145,
            l_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_attn_parameters_bias_ = (
            view_145
        ) = l_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_105 = x_104.view((1, 39, 6144))
        x_104 = None
        split_13 = x_105.split(2048, dim=2)
        x_105 = None
        query_states_26 = split_13[0]
        key_states_26 = split_13[1]
        value_states_26 = split_13[2]
        split_13 = None
        view_147 = key_states_26.view((1, 39, -1, 128))
        key_states_26 = None
        key_states_27 = view_147.transpose(1, 2)
        view_147 = None
        view_148 = value_states_26.view((1, 39, -1, 128))
        value_states_26 = None
        value_states_27 = view_148.transpose(1, 2)
        view_148 = None
        view_149 = query_states_26.view((1, 39, -1, 128))
        query_states_26 = None
        query_states_27 = view_149.transpose(1, 2)
        view_149 = None
        attention_mask_15 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 39, None),
            )
        ]
        query_13 = query_states_27.contiguous()
        query_states_27 = None
        key_13 = key_states_27.contiguous()
        key_states_27 = None
        value_13 = value_states_27.contiguous()
        value_states_27 = None
        attn_output_52 = torch._C._nn.scaled_dot_product_attention(
            query_13,
            key_13,
            value_13,
            attn_mask=attention_mask_15,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query_13 = key_13 = value_13 = attention_mask_15 = None
        transpose_55 = attn_output_52.transpose(1, 2)
        attn_output_52 = None
        attn_output_53 = transpose_55.contiguous()
        transpose_55 = None
        reshape_13 = attn_output_53.reshape(1, 39, -1)
        attn_output_53 = None
        attn_output_54 = reshape_13.contiguous()
        reshape_13 = None
        view_150 = attn_output_54.view(-1, 2048)
        attn_output_54 = None
        x_106 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_proj_parameters_bias_,
            view_150,
            l_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_proj_parameters_bias_ = (
            view_150
        ) = l_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_107 = x_106.view((1, 39, 2048))
        x_106 = None
        attn_output_55 = torch.nn.functional.dropout(x_107, 0.1, False, False)
        x_107 = None
        hidden_states_81 = attn_output_55 + hidden_states_79
        attn_output_55 = hidden_states_79 = None
        hidden_states_82 = torch.nn.functional.layer_norm(
            hidden_states_81,
            (2048,),
            l_self_modules_transformer_modules_h_modules_13_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_13_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_13_modules_ln_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_13_modules_ln_2_parameters_bias_ = (None)
        view_152 = hidden_states_82.view(-1, 2048)
        hidden_states_82 = None
        x_108 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_fc_parameters_bias_,
            view_152,
            l_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_152
        ) = l_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_109 = x_108.view((1, 39, 8192))
        x_108 = None
        mul_78 = 0.5 * x_109
        mul_79 = x_109 * 0.7978845608
        mul_80 = 0.044715 * x_109
        mul_81 = mul_80 * x_109
        mul_80 = x_109 = None
        add_54 = 1.0 + mul_81
        mul_81 = None
        mul_82 = mul_79 * add_54
        mul_79 = add_54 = None
        tanh_13 = torch.tanh(mul_82)
        mul_82 = None
        add_55 = 1.0 + tanh_13
        tanh_13 = None
        hidden_states_83 = mul_78 * add_55
        mul_78 = add_55 = None
        view_154 = hidden_states_83.view(-1, 8192)
        hidden_states_83 = None
        x_110 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_proj_parameters_bias_,
            view_154,
            l_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_154
        ) = l_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_111 = x_110.view((1, 39, 2048))
        x_110 = None
        hidden_states_84 = torch.nn.functional.dropout(x_111, 0.1, False, False)
        x_111 = None
        hidden_states_85 = hidden_states_81 + hidden_states_84
        hidden_states_81 = hidden_states_84 = None
        hidden_states_86 = torch.nn.functional.layer_norm(
            hidden_states_85,
            (2048,),
            l_self_modules_transformer_modules_h_modules_14_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_14_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_14_modules_ln_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_14_modules_ln_1_parameters_bias_ = (None)
        view_156 = hidden_states_86.view(-1, 2048)
        hidden_states_86 = None
        x_112 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_attn_parameters_bias_,
            view_156,
            l_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_attn_parameters_bias_ = (
            view_156
        ) = l_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_113 = x_112.view((1, 39, 6144))
        x_112 = None
        split_14 = x_113.split(2048, dim=2)
        x_113 = None
        query_states_28 = split_14[0]
        key_states_28 = split_14[1]
        value_states_28 = split_14[2]
        split_14 = None
        view_158 = key_states_28.view((1, 39, -1, 128))
        key_states_28 = None
        key_states_29 = view_158.transpose(1, 2)
        view_158 = None
        view_159 = value_states_28.view((1, 39, -1, 128))
        value_states_28 = None
        value_states_29 = view_159.transpose(1, 2)
        view_159 = None
        view_160 = query_states_28.view((1, 39, -1, 128))
        query_states_28 = None
        query_states_29 = view_160.transpose(1, 2)
        view_160 = None
        attention_mask_16 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 39, None),
            )
        ]
        query_14 = query_states_29.contiguous()
        query_states_29 = None
        key_14 = key_states_29.contiguous()
        key_states_29 = None
        value_14 = value_states_29.contiguous()
        value_states_29 = None
        attn_output_56 = torch._C._nn.scaled_dot_product_attention(
            query_14,
            key_14,
            value_14,
            attn_mask=attention_mask_16,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query_14 = key_14 = value_14 = attention_mask_16 = None
        transpose_59 = attn_output_56.transpose(1, 2)
        attn_output_56 = None
        attn_output_57 = transpose_59.contiguous()
        transpose_59 = None
        reshape_14 = attn_output_57.reshape(1, 39, -1)
        attn_output_57 = None
        attn_output_58 = reshape_14.contiguous()
        reshape_14 = None
        view_161 = attn_output_58.view(-1, 2048)
        attn_output_58 = None
        x_114 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_proj_parameters_bias_,
            view_161,
            l_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_proj_parameters_bias_ = (
            view_161
        ) = l_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_115 = x_114.view((1, 39, 2048))
        x_114 = None
        attn_output_59 = torch.nn.functional.dropout(x_115, 0.1, False, False)
        x_115 = None
        hidden_states_87 = attn_output_59 + hidden_states_85
        attn_output_59 = hidden_states_85 = None
        hidden_states_88 = torch.nn.functional.layer_norm(
            hidden_states_87,
            (2048,),
            l_self_modules_transformer_modules_h_modules_14_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_14_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_14_modules_ln_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_14_modules_ln_2_parameters_bias_ = (None)
        view_163 = hidden_states_88.view(-1, 2048)
        hidden_states_88 = None
        x_116 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_fc_parameters_bias_,
            view_163,
            l_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_163
        ) = l_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_117 = x_116.view((1, 39, 8192))
        x_116 = None
        mul_84 = 0.5 * x_117
        mul_85 = x_117 * 0.7978845608
        mul_86 = 0.044715 * x_117
        mul_87 = mul_86 * x_117
        mul_86 = x_117 = None
        add_58 = 1.0 + mul_87
        mul_87 = None
        mul_88 = mul_85 * add_58
        mul_85 = add_58 = None
        tanh_14 = torch.tanh(mul_88)
        mul_88 = None
        add_59 = 1.0 + tanh_14
        tanh_14 = None
        hidden_states_89 = mul_84 * add_59
        mul_84 = add_59 = None
        view_165 = hidden_states_89.view(-1, 8192)
        hidden_states_89 = None
        x_118 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_proj_parameters_bias_,
            view_165,
            l_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_165
        ) = l_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_119 = x_118.view((1, 39, 2048))
        x_118 = None
        hidden_states_90 = torch.nn.functional.dropout(x_119, 0.1, False, False)
        x_119 = None
        hidden_states_91 = hidden_states_87 + hidden_states_90
        hidden_states_87 = hidden_states_90 = None
        hidden_states_92 = torch.nn.functional.layer_norm(
            hidden_states_91,
            (2048,),
            l_self_modules_transformer_modules_h_modules_15_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_15_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_15_modules_ln_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_15_modules_ln_1_parameters_bias_ = (None)
        view_167 = hidden_states_92.view(-1, 2048)
        hidden_states_92 = None
        x_120 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_attn_parameters_bias_,
            view_167,
            l_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_attn_parameters_bias_ = (
            view_167
        ) = l_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_121 = x_120.view((1, 39, 6144))
        x_120 = None
        split_15 = x_121.split(2048, dim=2)
        x_121 = None
        query_states_30 = split_15[0]
        key_states_30 = split_15[1]
        value_states_30 = split_15[2]
        split_15 = None
        view_169 = key_states_30.view((1, 39, -1, 128))
        key_states_30 = None
        key_states_31 = view_169.transpose(1, 2)
        view_169 = None
        view_170 = value_states_30.view((1, 39, -1, 128))
        value_states_30 = None
        value_states_31 = view_170.transpose(1, 2)
        view_170 = None
        view_171 = query_states_30.view((1, 39, -1, 128))
        query_states_30 = None
        query_states_31 = view_171.transpose(1, 2)
        view_171 = None
        attention_mask_17 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 39, None),
            )
        ]
        query_15 = query_states_31.contiguous()
        query_states_31 = None
        key_15 = key_states_31.contiguous()
        key_states_31 = None
        value_15 = value_states_31.contiguous()
        value_states_31 = None
        attn_output_60 = torch._C._nn.scaled_dot_product_attention(
            query_15,
            key_15,
            value_15,
            attn_mask=attention_mask_17,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query_15 = key_15 = value_15 = attention_mask_17 = None
        transpose_63 = attn_output_60.transpose(1, 2)
        attn_output_60 = None
        attn_output_61 = transpose_63.contiguous()
        transpose_63 = None
        reshape_15 = attn_output_61.reshape(1, 39, -1)
        attn_output_61 = None
        attn_output_62 = reshape_15.contiguous()
        reshape_15 = None
        view_172 = attn_output_62.view(-1, 2048)
        attn_output_62 = None
        x_122 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_proj_parameters_bias_,
            view_172,
            l_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_proj_parameters_bias_ = (
            view_172
        ) = l_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_123 = x_122.view((1, 39, 2048))
        x_122 = None
        attn_output_63 = torch.nn.functional.dropout(x_123, 0.1, False, False)
        x_123 = None
        hidden_states_93 = attn_output_63 + hidden_states_91
        attn_output_63 = hidden_states_91 = None
        hidden_states_94 = torch.nn.functional.layer_norm(
            hidden_states_93,
            (2048,),
            l_self_modules_transformer_modules_h_modules_15_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_15_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_15_modules_ln_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_15_modules_ln_2_parameters_bias_ = (None)
        view_174 = hidden_states_94.view(-1, 2048)
        hidden_states_94 = None
        x_124 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_fc_parameters_bias_,
            view_174,
            l_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_174
        ) = l_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_125 = x_124.view((1, 39, 8192))
        x_124 = None
        mul_90 = 0.5 * x_125
        mul_91 = x_125 * 0.7978845608
        mul_92 = 0.044715 * x_125
        mul_93 = mul_92 * x_125
        mul_92 = x_125 = None
        add_62 = 1.0 + mul_93
        mul_93 = None
        mul_94 = mul_91 * add_62
        mul_91 = add_62 = None
        tanh_15 = torch.tanh(mul_94)
        mul_94 = None
        add_63 = 1.0 + tanh_15
        tanh_15 = None
        hidden_states_95 = mul_90 * add_63
        mul_90 = add_63 = None
        view_176 = hidden_states_95.view(-1, 8192)
        hidden_states_95 = None
        x_126 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_proj_parameters_bias_,
            view_176,
            l_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_176
        ) = l_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_127 = x_126.view((1, 39, 2048))
        x_126 = None
        hidden_states_96 = torch.nn.functional.dropout(x_127, 0.1, False, False)
        x_127 = None
        hidden_states_97 = hidden_states_93 + hidden_states_96
        hidden_states_93 = hidden_states_96 = None
        hidden_states_98 = torch.nn.functional.layer_norm(
            hidden_states_97,
            (2048,),
            l_self_modules_transformer_modules_h_modules_16_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_16_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_16_modules_ln_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_16_modules_ln_1_parameters_bias_ = (None)
        view_178 = hidden_states_98.view(-1, 2048)
        hidden_states_98 = None
        x_128 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_attn_parameters_bias_,
            view_178,
            l_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_attn_parameters_bias_ = (
            view_178
        ) = l_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_129 = x_128.view((1, 39, 6144))
        x_128 = None
        split_16 = x_129.split(2048, dim=2)
        x_129 = None
        query_states_32 = split_16[0]
        key_states_32 = split_16[1]
        value_states_32 = split_16[2]
        split_16 = None
        view_180 = key_states_32.view((1, 39, -1, 128))
        key_states_32 = None
        key_states_33 = view_180.transpose(1, 2)
        view_180 = None
        view_181 = value_states_32.view((1, 39, -1, 128))
        value_states_32 = None
        value_states_33 = view_181.transpose(1, 2)
        view_181 = None
        view_182 = query_states_32.view((1, 39, -1, 128))
        query_states_32 = None
        query_states_33 = view_182.transpose(1, 2)
        view_182 = None
        attention_mask_18 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 39, None),
            )
        ]
        query_16 = query_states_33.contiguous()
        query_states_33 = None
        key_16 = key_states_33.contiguous()
        key_states_33 = None
        value_16 = value_states_33.contiguous()
        value_states_33 = None
        attn_output_64 = torch._C._nn.scaled_dot_product_attention(
            query_16,
            key_16,
            value_16,
            attn_mask=attention_mask_18,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query_16 = key_16 = value_16 = attention_mask_18 = None
        transpose_67 = attn_output_64.transpose(1, 2)
        attn_output_64 = None
        attn_output_65 = transpose_67.contiguous()
        transpose_67 = None
        reshape_16 = attn_output_65.reshape(1, 39, -1)
        attn_output_65 = None
        attn_output_66 = reshape_16.contiguous()
        reshape_16 = None
        view_183 = attn_output_66.view(-1, 2048)
        attn_output_66 = None
        x_130 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_proj_parameters_bias_,
            view_183,
            l_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_proj_parameters_bias_ = (
            view_183
        ) = l_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_131 = x_130.view((1, 39, 2048))
        x_130 = None
        attn_output_67 = torch.nn.functional.dropout(x_131, 0.1, False, False)
        x_131 = None
        hidden_states_99 = attn_output_67 + hidden_states_97
        attn_output_67 = hidden_states_97 = None
        hidden_states_100 = torch.nn.functional.layer_norm(
            hidden_states_99,
            (2048,),
            l_self_modules_transformer_modules_h_modules_16_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_16_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_16_modules_ln_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_16_modules_ln_2_parameters_bias_ = (None)
        view_185 = hidden_states_100.view(-1, 2048)
        hidden_states_100 = None
        x_132 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_fc_parameters_bias_,
            view_185,
            l_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_185
        ) = l_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_133 = x_132.view((1, 39, 8192))
        x_132 = None
        mul_96 = 0.5 * x_133
        mul_97 = x_133 * 0.7978845608
        mul_98 = 0.044715 * x_133
        mul_99 = mul_98 * x_133
        mul_98 = x_133 = None
        add_66 = 1.0 + mul_99
        mul_99 = None
        mul_100 = mul_97 * add_66
        mul_97 = add_66 = None
        tanh_16 = torch.tanh(mul_100)
        mul_100 = None
        add_67 = 1.0 + tanh_16
        tanh_16 = None
        hidden_states_101 = mul_96 * add_67
        mul_96 = add_67 = None
        view_187 = hidden_states_101.view(-1, 8192)
        hidden_states_101 = None
        x_134 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_proj_parameters_bias_,
            view_187,
            l_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_187
        ) = l_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_135 = x_134.view((1, 39, 2048))
        x_134 = None
        hidden_states_102 = torch.nn.functional.dropout(x_135, 0.1, False, False)
        x_135 = None
        hidden_states_103 = hidden_states_99 + hidden_states_102
        hidden_states_99 = hidden_states_102 = None
        hidden_states_104 = torch.nn.functional.layer_norm(
            hidden_states_103,
            (2048,),
            l_self_modules_transformer_modules_h_modules_17_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_17_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_17_modules_ln_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_17_modules_ln_1_parameters_bias_ = (None)
        view_189 = hidden_states_104.view(-1, 2048)
        hidden_states_104 = None
        x_136 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_attn_parameters_bias_,
            view_189,
            l_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_attn_parameters_bias_ = (
            view_189
        ) = l_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_137 = x_136.view((1, 39, 6144))
        x_136 = None
        split_17 = x_137.split(2048, dim=2)
        x_137 = None
        query_states_34 = split_17[0]
        key_states_34 = split_17[1]
        value_states_34 = split_17[2]
        split_17 = None
        view_191 = key_states_34.view((1, 39, -1, 128))
        key_states_34 = None
        key_states_35 = view_191.transpose(1, 2)
        view_191 = None
        view_192 = value_states_34.view((1, 39, -1, 128))
        value_states_34 = None
        value_states_35 = view_192.transpose(1, 2)
        view_192 = None
        view_193 = query_states_34.view((1, 39, -1, 128))
        query_states_34 = None
        query_states_35 = view_193.transpose(1, 2)
        view_193 = None
        attention_mask_19 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 39, None),
            )
        ]
        query_17 = query_states_35.contiguous()
        query_states_35 = None
        key_17 = key_states_35.contiguous()
        key_states_35 = None
        value_17 = value_states_35.contiguous()
        value_states_35 = None
        attn_output_68 = torch._C._nn.scaled_dot_product_attention(
            query_17,
            key_17,
            value_17,
            attn_mask=attention_mask_19,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query_17 = key_17 = value_17 = attention_mask_19 = None
        transpose_71 = attn_output_68.transpose(1, 2)
        attn_output_68 = None
        attn_output_69 = transpose_71.contiguous()
        transpose_71 = None
        reshape_17 = attn_output_69.reshape(1, 39, -1)
        attn_output_69 = None
        attn_output_70 = reshape_17.contiguous()
        reshape_17 = None
        view_194 = attn_output_70.view(-1, 2048)
        attn_output_70 = None
        x_138 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_proj_parameters_bias_,
            view_194,
            l_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_proj_parameters_bias_ = (
            view_194
        ) = l_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_139 = x_138.view((1, 39, 2048))
        x_138 = None
        attn_output_71 = torch.nn.functional.dropout(x_139, 0.1, False, False)
        x_139 = None
        hidden_states_105 = attn_output_71 + hidden_states_103
        attn_output_71 = hidden_states_103 = None
        hidden_states_106 = torch.nn.functional.layer_norm(
            hidden_states_105,
            (2048,),
            l_self_modules_transformer_modules_h_modules_17_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_17_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_17_modules_ln_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_17_modules_ln_2_parameters_bias_ = (None)
        view_196 = hidden_states_106.view(-1, 2048)
        hidden_states_106 = None
        x_140 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_fc_parameters_bias_,
            view_196,
            l_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_196
        ) = l_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_141 = x_140.view((1, 39, 8192))
        x_140 = None
        mul_102 = 0.5 * x_141
        mul_103 = x_141 * 0.7978845608
        mul_104 = 0.044715 * x_141
        mul_105 = mul_104 * x_141
        mul_104 = x_141 = None
        add_70 = 1.0 + mul_105
        mul_105 = None
        mul_106 = mul_103 * add_70
        mul_103 = add_70 = None
        tanh_17 = torch.tanh(mul_106)
        mul_106 = None
        add_71 = 1.0 + tanh_17
        tanh_17 = None
        hidden_states_107 = mul_102 * add_71
        mul_102 = add_71 = None
        view_198 = hidden_states_107.view(-1, 8192)
        hidden_states_107 = None
        x_142 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_proj_parameters_bias_,
            view_198,
            l_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_198
        ) = l_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_143 = x_142.view((1, 39, 2048))
        x_142 = None
        hidden_states_108 = torch.nn.functional.dropout(x_143, 0.1, False, False)
        x_143 = None
        hidden_states_109 = hidden_states_105 + hidden_states_108
        hidden_states_105 = hidden_states_108 = None
        hidden_states_110 = torch.nn.functional.layer_norm(
            hidden_states_109,
            (2048,),
            l_self_modules_transformer_modules_h_modules_18_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_18_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_18_modules_ln_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_18_modules_ln_1_parameters_bias_ = (None)
        view_200 = hidden_states_110.view(-1, 2048)
        hidden_states_110 = None
        x_144 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_attn_parameters_bias_,
            view_200,
            l_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_attn_parameters_bias_ = (
            view_200
        ) = l_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_145 = x_144.view((1, 39, 6144))
        x_144 = None
        split_18 = x_145.split(2048, dim=2)
        x_145 = None
        query_states_36 = split_18[0]
        key_states_36 = split_18[1]
        value_states_36 = split_18[2]
        split_18 = None
        view_202 = key_states_36.view((1, 39, -1, 128))
        key_states_36 = None
        key_states_37 = view_202.transpose(1, 2)
        view_202 = None
        view_203 = value_states_36.view((1, 39, -1, 128))
        value_states_36 = None
        value_states_37 = view_203.transpose(1, 2)
        view_203 = None
        view_204 = query_states_36.view((1, 39, -1, 128))
        query_states_36 = None
        query_states_37 = view_204.transpose(1, 2)
        view_204 = None
        attention_mask_20 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 39, None),
            )
        ]
        query_18 = query_states_37.contiguous()
        query_states_37 = None
        key_18 = key_states_37.contiguous()
        key_states_37 = None
        value_18 = value_states_37.contiguous()
        value_states_37 = None
        attn_output_72 = torch._C._nn.scaled_dot_product_attention(
            query_18,
            key_18,
            value_18,
            attn_mask=attention_mask_20,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query_18 = key_18 = value_18 = attention_mask_20 = None
        transpose_75 = attn_output_72.transpose(1, 2)
        attn_output_72 = None
        attn_output_73 = transpose_75.contiguous()
        transpose_75 = None
        reshape_18 = attn_output_73.reshape(1, 39, -1)
        attn_output_73 = None
        attn_output_74 = reshape_18.contiguous()
        reshape_18 = None
        view_205 = attn_output_74.view(-1, 2048)
        attn_output_74 = None
        x_146 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_proj_parameters_bias_,
            view_205,
            l_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_proj_parameters_bias_ = (
            view_205
        ) = l_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_147 = x_146.view((1, 39, 2048))
        x_146 = None
        attn_output_75 = torch.nn.functional.dropout(x_147, 0.1, False, False)
        x_147 = None
        hidden_states_111 = attn_output_75 + hidden_states_109
        attn_output_75 = hidden_states_109 = None
        hidden_states_112 = torch.nn.functional.layer_norm(
            hidden_states_111,
            (2048,),
            l_self_modules_transformer_modules_h_modules_18_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_18_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_18_modules_ln_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_18_modules_ln_2_parameters_bias_ = (None)
        view_207 = hidden_states_112.view(-1, 2048)
        hidden_states_112 = None
        x_148 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_fc_parameters_bias_,
            view_207,
            l_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_207
        ) = l_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_149 = x_148.view((1, 39, 8192))
        x_148 = None
        mul_108 = 0.5 * x_149
        mul_109 = x_149 * 0.7978845608
        mul_110 = 0.044715 * x_149
        mul_111 = mul_110 * x_149
        mul_110 = x_149 = None
        add_74 = 1.0 + mul_111
        mul_111 = None
        mul_112 = mul_109 * add_74
        mul_109 = add_74 = None
        tanh_18 = torch.tanh(mul_112)
        mul_112 = None
        add_75 = 1.0 + tanh_18
        tanh_18 = None
        hidden_states_113 = mul_108 * add_75
        mul_108 = add_75 = None
        view_209 = hidden_states_113.view(-1, 8192)
        hidden_states_113 = None
        x_150 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_proj_parameters_bias_,
            view_209,
            l_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_209
        ) = l_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_151 = x_150.view((1, 39, 2048))
        x_150 = None
        hidden_states_114 = torch.nn.functional.dropout(x_151, 0.1, False, False)
        x_151 = None
        hidden_states_115 = hidden_states_111 + hidden_states_114
        hidden_states_111 = hidden_states_114 = None
        hidden_states_116 = torch.nn.functional.layer_norm(
            hidden_states_115,
            (2048,),
            l_self_modules_transformer_modules_h_modules_19_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_19_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_19_modules_ln_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_19_modules_ln_1_parameters_bias_ = (None)
        view_211 = hidden_states_116.view(-1, 2048)
        hidden_states_116 = None
        x_152 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_attn_parameters_bias_,
            view_211,
            l_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_attn_parameters_bias_ = (
            view_211
        ) = l_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_153 = x_152.view((1, 39, 6144))
        x_152 = None
        split_19 = x_153.split(2048, dim=2)
        x_153 = None
        query_states_38 = split_19[0]
        key_states_38 = split_19[1]
        value_states_38 = split_19[2]
        split_19 = None
        view_213 = key_states_38.view((1, 39, -1, 128))
        key_states_38 = None
        key_states_39 = view_213.transpose(1, 2)
        view_213 = None
        view_214 = value_states_38.view((1, 39, -1, 128))
        value_states_38 = None
        value_states_39 = view_214.transpose(1, 2)
        view_214 = None
        view_215 = query_states_38.view((1, 39, -1, 128))
        query_states_38 = None
        query_states_39 = view_215.transpose(1, 2)
        view_215 = None
        attention_mask_21 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 39, None),
            )
        ]
        query_19 = query_states_39.contiguous()
        query_states_39 = None
        key_19 = key_states_39.contiguous()
        key_states_39 = None
        value_19 = value_states_39.contiguous()
        value_states_39 = None
        attn_output_76 = torch._C._nn.scaled_dot_product_attention(
            query_19,
            key_19,
            value_19,
            attn_mask=attention_mask_21,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query_19 = key_19 = value_19 = attention_mask_21 = None
        transpose_79 = attn_output_76.transpose(1, 2)
        attn_output_76 = None
        attn_output_77 = transpose_79.contiguous()
        transpose_79 = None
        reshape_19 = attn_output_77.reshape(1, 39, -1)
        attn_output_77 = None
        attn_output_78 = reshape_19.contiguous()
        reshape_19 = None
        view_216 = attn_output_78.view(-1, 2048)
        attn_output_78 = None
        x_154 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_proj_parameters_bias_,
            view_216,
            l_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_proj_parameters_bias_ = (
            view_216
        ) = l_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_155 = x_154.view((1, 39, 2048))
        x_154 = None
        attn_output_79 = torch.nn.functional.dropout(x_155, 0.1, False, False)
        x_155 = None
        hidden_states_117 = attn_output_79 + hidden_states_115
        attn_output_79 = hidden_states_115 = None
        hidden_states_118 = torch.nn.functional.layer_norm(
            hidden_states_117,
            (2048,),
            l_self_modules_transformer_modules_h_modules_19_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_19_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_19_modules_ln_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_19_modules_ln_2_parameters_bias_ = (None)
        view_218 = hidden_states_118.view(-1, 2048)
        hidden_states_118 = None
        x_156 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_fc_parameters_bias_,
            view_218,
            l_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_218
        ) = l_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_157 = x_156.view((1, 39, 8192))
        x_156 = None
        mul_114 = 0.5 * x_157
        mul_115 = x_157 * 0.7978845608
        mul_116 = 0.044715 * x_157
        mul_117 = mul_116 * x_157
        mul_116 = x_157 = None
        add_78 = 1.0 + mul_117
        mul_117 = None
        mul_118 = mul_115 * add_78
        mul_115 = add_78 = None
        tanh_19 = torch.tanh(mul_118)
        mul_118 = None
        add_79 = 1.0 + tanh_19
        tanh_19 = None
        hidden_states_119 = mul_114 * add_79
        mul_114 = add_79 = None
        view_220 = hidden_states_119.view(-1, 8192)
        hidden_states_119 = None
        x_158 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_proj_parameters_bias_,
            view_220,
            l_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_220
        ) = l_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_159 = x_158.view((1, 39, 2048))
        x_158 = None
        hidden_states_120 = torch.nn.functional.dropout(x_159, 0.1, False, False)
        x_159 = None
        hidden_states_121 = hidden_states_117 + hidden_states_120
        hidden_states_117 = hidden_states_120 = None
        hidden_states_122 = torch.nn.functional.layer_norm(
            hidden_states_121,
            (2048,),
            l_self_modules_transformer_modules_h_modules_20_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_20_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_20_modules_ln_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_20_modules_ln_1_parameters_bias_ = (None)
        view_222 = hidden_states_122.view(-1, 2048)
        hidden_states_122 = None
        x_160 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_20_modules_attn_modules_c_attn_parameters_bias_,
            view_222,
            l_self_modules_transformer_modules_h_modules_20_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_20_modules_attn_modules_c_attn_parameters_bias_ = (
            view_222
        ) = l_self_modules_transformer_modules_h_modules_20_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_161 = x_160.view((1, 39, 6144))
        x_160 = None
        split_20 = x_161.split(2048, dim=2)
        x_161 = None
        query_states_40 = split_20[0]
        key_states_40 = split_20[1]
        value_states_40 = split_20[2]
        split_20 = None
        view_224 = key_states_40.view((1, 39, -1, 128))
        key_states_40 = None
        key_states_41 = view_224.transpose(1, 2)
        view_224 = None
        view_225 = value_states_40.view((1, 39, -1, 128))
        value_states_40 = None
        value_states_41 = view_225.transpose(1, 2)
        view_225 = None
        view_226 = query_states_40.view((1, 39, -1, 128))
        query_states_40 = None
        query_states_41 = view_226.transpose(1, 2)
        view_226 = None
        attention_mask_22 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 39, None),
            )
        ]
        query_20 = query_states_41.contiguous()
        query_states_41 = None
        key_20 = key_states_41.contiguous()
        key_states_41 = None
        value_20 = value_states_41.contiguous()
        value_states_41 = None
        attn_output_80 = torch._C._nn.scaled_dot_product_attention(
            query_20,
            key_20,
            value_20,
            attn_mask=attention_mask_22,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query_20 = key_20 = value_20 = attention_mask_22 = None
        transpose_83 = attn_output_80.transpose(1, 2)
        attn_output_80 = None
        attn_output_81 = transpose_83.contiguous()
        transpose_83 = None
        reshape_20 = attn_output_81.reshape(1, 39, -1)
        attn_output_81 = None
        attn_output_82 = reshape_20.contiguous()
        reshape_20 = None
        view_227 = attn_output_82.view(-1, 2048)
        attn_output_82 = None
        x_162 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_20_modules_attn_modules_c_proj_parameters_bias_,
            view_227,
            l_self_modules_transformer_modules_h_modules_20_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_20_modules_attn_modules_c_proj_parameters_bias_ = (
            view_227
        ) = l_self_modules_transformer_modules_h_modules_20_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_163 = x_162.view((1, 39, 2048))
        x_162 = None
        attn_output_83 = torch.nn.functional.dropout(x_163, 0.1, False, False)
        x_163 = None
        hidden_states_123 = attn_output_83 + hidden_states_121
        attn_output_83 = hidden_states_121 = None
        hidden_states_124 = torch.nn.functional.layer_norm(
            hidden_states_123,
            (2048,),
            l_self_modules_transformer_modules_h_modules_20_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_20_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_20_modules_ln_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_20_modules_ln_2_parameters_bias_ = (None)
        view_229 = hidden_states_124.view(-1, 2048)
        hidden_states_124 = None
        x_164 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_20_modules_mlp_modules_c_fc_parameters_bias_,
            view_229,
            l_self_modules_transformer_modules_h_modules_20_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_20_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_229
        ) = l_self_modules_transformer_modules_h_modules_20_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_165 = x_164.view((1, 39, 8192))
        x_164 = None
        mul_120 = 0.5 * x_165
        mul_121 = x_165 * 0.7978845608
        mul_122 = 0.044715 * x_165
        mul_123 = mul_122 * x_165
        mul_122 = x_165 = None
        add_82 = 1.0 + mul_123
        mul_123 = None
        mul_124 = mul_121 * add_82
        mul_121 = add_82 = None
        tanh_20 = torch.tanh(mul_124)
        mul_124 = None
        add_83 = 1.0 + tanh_20
        tanh_20 = None
        hidden_states_125 = mul_120 * add_83
        mul_120 = add_83 = None
        view_231 = hidden_states_125.view(-1, 8192)
        hidden_states_125 = None
        x_166 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_20_modules_mlp_modules_c_proj_parameters_bias_,
            view_231,
            l_self_modules_transformer_modules_h_modules_20_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_20_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_231
        ) = l_self_modules_transformer_modules_h_modules_20_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_167 = x_166.view((1, 39, 2048))
        x_166 = None
        hidden_states_126 = torch.nn.functional.dropout(x_167, 0.1, False, False)
        x_167 = None
        hidden_states_127 = hidden_states_123 + hidden_states_126
        hidden_states_123 = hidden_states_126 = None
        hidden_states_128 = torch.nn.functional.layer_norm(
            hidden_states_127,
            (2048,),
            l_self_modules_transformer_modules_h_modules_21_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_21_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_21_modules_ln_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_21_modules_ln_1_parameters_bias_ = (None)
        view_233 = hidden_states_128.view(-1, 2048)
        hidden_states_128 = None
        x_168 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_21_modules_attn_modules_c_attn_parameters_bias_,
            view_233,
            l_self_modules_transformer_modules_h_modules_21_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_21_modules_attn_modules_c_attn_parameters_bias_ = (
            view_233
        ) = l_self_modules_transformer_modules_h_modules_21_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_169 = x_168.view((1, 39, 6144))
        x_168 = None
        split_21 = x_169.split(2048, dim=2)
        x_169 = None
        query_states_42 = split_21[0]
        key_states_42 = split_21[1]
        value_states_42 = split_21[2]
        split_21 = None
        view_235 = key_states_42.view((1, 39, -1, 128))
        key_states_42 = None
        key_states_43 = view_235.transpose(1, 2)
        view_235 = None
        view_236 = value_states_42.view((1, 39, -1, 128))
        value_states_42 = None
        value_states_43 = view_236.transpose(1, 2)
        view_236 = None
        view_237 = query_states_42.view((1, 39, -1, 128))
        query_states_42 = None
        query_states_43 = view_237.transpose(1, 2)
        view_237 = None
        attention_mask_23 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 39, None),
            )
        ]
        query_21 = query_states_43.contiguous()
        query_states_43 = None
        key_21 = key_states_43.contiguous()
        key_states_43 = None
        value_21 = value_states_43.contiguous()
        value_states_43 = None
        attn_output_84 = torch._C._nn.scaled_dot_product_attention(
            query_21,
            key_21,
            value_21,
            attn_mask=attention_mask_23,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query_21 = key_21 = value_21 = attention_mask_23 = None
        transpose_87 = attn_output_84.transpose(1, 2)
        attn_output_84 = None
        attn_output_85 = transpose_87.contiguous()
        transpose_87 = None
        reshape_21 = attn_output_85.reshape(1, 39, -1)
        attn_output_85 = None
        attn_output_86 = reshape_21.contiguous()
        reshape_21 = None
        view_238 = attn_output_86.view(-1, 2048)
        attn_output_86 = None
        x_170 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_21_modules_attn_modules_c_proj_parameters_bias_,
            view_238,
            l_self_modules_transformer_modules_h_modules_21_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_21_modules_attn_modules_c_proj_parameters_bias_ = (
            view_238
        ) = l_self_modules_transformer_modules_h_modules_21_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_171 = x_170.view((1, 39, 2048))
        x_170 = None
        attn_output_87 = torch.nn.functional.dropout(x_171, 0.1, False, False)
        x_171 = None
        hidden_states_129 = attn_output_87 + hidden_states_127
        attn_output_87 = hidden_states_127 = None
        hidden_states_130 = torch.nn.functional.layer_norm(
            hidden_states_129,
            (2048,),
            l_self_modules_transformer_modules_h_modules_21_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_21_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_21_modules_ln_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_21_modules_ln_2_parameters_bias_ = (None)
        view_240 = hidden_states_130.view(-1, 2048)
        hidden_states_130 = None
        x_172 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_21_modules_mlp_modules_c_fc_parameters_bias_,
            view_240,
            l_self_modules_transformer_modules_h_modules_21_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_21_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_240
        ) = l_self_modules_transformer_modules_h_modules_21_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_173 = x_172.view((1, 39, 8192))
        x_172 = None
        mul_126 = 0.5 * x_173
        mul_127 = x_173 * 0.7978845608
        mul_128 = 0.044715 * x_173
        mul_129 = mul_128 * x_173
        mul_128 = x_173 = None
        add_86 = 1.0 + mul_129
        mul_129 = None
        mul_130 = mul_127 * add_86
        mul_127 = add_86 = None
        tanh_21 = torch.tanh(mul_130)
        mul_130 = None
        add_87 = 1.0 + tanh_21
        tanh_21 = None
        hidden_states_131 = mul_126 * add_87
        mul_126 = add_87 = None
        view_242 = hidden_states_131.view(-1, 8192)
        hidden_states_131 = None
        x_174 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_21_modules_mlp_modules_c_proj_parameters_bias_,
            view_242,
            l_self_modules_transformer_modules_h_modules_21_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_21_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_242
        ) = l_self_modules_transformer_modules_h_modules_21_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_175 = x_174.view((1, 39, 2048))
        x_174 = None
        hidden_states_132 = torch.nn.functional.dropout(x_175, 0.1, False, False)
        x_175 = None
        hidden_states_133 = hidden_states_129 + hidden_states_132
        hidden_states_129 = hidden_states_132 = None
        hidden_states_134 = torch.nn.functional.layer_norm(
            hidden_states_133,
            (2048,),
            l_self_modules_transformer_modules_h_modules_22_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_22_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_22_modules_ln_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_22_modules_ln_1_parameters_bias_ = (None)
        view_244 = hidden_states_134.view(-1, 2048)
        hidden_states_134 = None
        x_176 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_22_modules_attn_modules_c_attn_parameters_bias_,
            view_244,
            l_self_modules_transformer_modules_h_modules_22_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_22_modules_attn_modules_c_attn_parameters_bias_ = (
            view_244
        ) = l_self_modules_transformer_modules_h_modules_22_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_177 = x_176.view((1, 39, 6144))
        x_176 = None
        split_22 = x_177.split(2048, dim=2)
        x_177 = None
        query_states_44 = split_22[0]
        key_states_44 = split_22[1]
        value_states_44 = split_22[2]
        split_22 = None
        view_246 = key_states_44.view((1, 39, -1, 128))
        key_states_44 = None
        key_states_45 = view_246.transpose(1, 2)
        view_246 = None
        view_247 = value_states_44.view((1, 39, -1, 128))
        value_states_44 = None
        value_states_45 = view_247.transpose(1, 2)
        view_247 = None
        view_248 = query_states_44.view((1, 39, -1, 128))
        query_states_44 = None
        query_states_45 = view_248.transpose(1, 2)
        view_248 = None
        attention_mask_24 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 39, None),
            )
        ]
        query_22 = query_states_45.contiguous()
        query_states_45 = None
        key_22 = key_states_45.contiguous()
        key_states_45 = None
        value_22 = value_states_45.contiguous()
        value_states_45 = None
        attn_output_88 = torch._C._nn.scaled_dot_product_attention(
            query_22,
            key_22,
            value_22,
            attn_mask=attention_mask_24,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query_22 = key_22 = value_22 = attention_mask_24 = None
        transpose_91 = attn_output_88.transpose(1, 2)
        attn_output_88 = None
        attn_output_89 = transpose_91.contiguous()
        transpose_91 = None
        reshape_22 = attn_output_89.reshape(1, 39, -1)
        attn_output_89 = None
        attn_output_90 = reshape_22.contiguous()
        reshape_22 = None
        view_249 = attn_output_90.view(-1, 2048)
        attn_output_90 = None
        x_178 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_22_modules_attn_modules_c_proj_parameters_bias_,
            view_249,
            l_self_modules_transformer_modules_h_modules_22_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_22_modules_attn_modules_c_proj_parameters_bias_ = (
            view_249
        ) = l_self_modules_transformer_modules_h_modules_22_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_179 = x_178.view((1, 39, 2048))
        x_178 = None
        attn_output_91 = torch.nn.functional.dropout(x_179, 0.1, False, False)
        x_179 = None
        hidden_states_135 = attn_output_91 + hidden_states_133
        attn_output_91 = hidden_states_133 = None
        hidden_states_136 = torch.nn.functional.layer_norm(
            hidden_states_135,
            (2048,),
            l_self_modules_transformer_modules_h_modules_22_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_22_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_22_modules_ln_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_22_modules_ln_2_parameters_bias_ = (None)
        view_251 = hidden_states_136.view(-1, 2048)
        hidden_states_136 = None
        x_180 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_22_modules_mlp_modules_c_fc_parameters_bias_,
            view_251,
            l_self_modules_transformer_modules_h_modules_22_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_22_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_251
        ) = l_self_modules_transformer_modules_h_modules_22_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_181 = x_180.view((1, 39, 8192))
        x_180 = None
        mul_132 = 0.5 * x_181
        mul_133 = x_181 * 0.7978845608
        mul_134 = 0.044715 * x_181
        mul_135 = mul_134 * x_181
        mul_134 = x_181 = None
        add_90 = 1.0 + mul_135
        mul_135 = None
        mul_136 = mul_133 * add_90
        mul_133 = add_90 = None
        tanh_22 = torch.tanh(mul_136)
        mul_136 = None
        add_91 = 1.0 + tanh_22
        tanh_22 = None
        hidden_states_137 = mul_132 * add_91
        mul_132 = add_91 = None
        view_253 = hidden_states_137.view(-1, 8192)
        hidden_states_137 = None
        x_182 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_22_modules_mlp_modules_c_proj_parameters_bias_,
            view_253,
            l_self_modules_transformer_modules_h_modules_22_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_22_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_253
        ) = l_self_modules_transformer_modules_h_modules_22_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_183 = x_182.view((1, 39, 2048))
        x_182 = None
        hidden_states_138 = torch.nn.functional.dropout(x_183, 0.1, False, False)
        x_183 = None
        hidden_states_139 = hidden_states_135 + hidden_states_138
        hidden_states_135 = hidden_states_138 = None
        hidden_states_140 = torch.nn.functional.layer_norm(
            hidden_states_139,
            (2048,),
            l_self_modules_transformer_modules_h_modules_23_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_23_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_23_modules_ln_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_23_modules_ln_1_parameters_bias_ = (None)
        view_255 = hidden_states_140.view(-1, 2048)
        hidden_states_140 = None
        x_184 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_23_modules_attn_modules_c_attn_parameters_bias_,
            view_255,
            l_self_modules_transformer_modules_h_modules_23_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_23_modules_attn_modules_c_attn_parameters_bias_ = (
            view_255
        ) = l_self_modules_transformer_modules_h_modules_23_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_185 = x_184.view((1, 39, 6144))
        x_184 = None
        split_23 = x_185.split(2048, dim=2)
        x_185 = None
        query_states_46 = split_23[0]
        key_states_46 = split_23[1]
        value_states_46 = split_23[2]
        split_23 = None
        view_257 = key_states_46.view((1, 39, -1, 128))
        key_states_46 = None
        key_states_47 = view_257.transpose(1, 2)
        view_257 = None
        view_258 = value_states_46.view((1, 39, -1, 128))
        value_states_46 = None
        value_states_47 = view_258.transpose(1, 2)
        view_258 = None
        view_259 = query_states_46.view((1, 39, -1, 128))
        query_states_46 = None
        query_states_47 = view_259.transpose(1, 2)
        view_259 = None
        attention_mask_25 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 39, None),
            )
        ]
        causal_mask = None
        query_23 = query_states_47.contiguous()
        query_states_47 = None
        key_23 = key_states_47.contiguous()
        key_states_47 = None
        value_23 = value_states_47.contiguous()
        value_states_47 = None
        attn_output_92 = torch._C._nn.scaled_dot_product_attention(
            query_23,
            key_23,
            value_23,
            attn_mask=attention_mask_25,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query_23 = key_23 = value_23 = attention_mask_25 = None
        transpose_95 = attn_output_92.transpose(1, 2)
        attn_output_92 = None
        attn_output_93 = transpose_95.contiguous()
        transpose_95 = None
        reshape_23 = attn_output_93.reshape(1, 39, -1)
        attn_output_93 = None
        attn_output_94 = reshape_23.contiguous()
        reshape_23 = None
        view_260 = attn_output_94.view(-1, 2048)
        attn_output_94 = None
        x_186 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_23_modules_attn_modules_c_proj_parameters_bias_,
            view_260,
            l_self_modules_transformer_modules_h_modules_23_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_23_modules_attn_modules_c_proj_parameters_bias_ = (
            view_260
        ) = l_self_modules_transformer_modules_h_modules_23_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_187 = x_186.view((1, 39, 2048))
        x_186 = None
        attn_output_95 = torch.nn.functional.dropout(x_187, 0.1, False, False)
        x_187 = None
        hidden_states_141 = attn_output_95 + hidden_states_139
        attn_output_95 = hidden_states_139 = None
        hidden_states_142 = torch.nn.functional.layer_norm(
            hidden_states_141,
            (2048,),
            l_self_modules_transformer_modules_h_modules_23_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_23_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_23_modules_ln_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_23_modules_ln_2_parameters_bias_ = (None)
        view_262 = hidden_states_142.view(-1, 2048)
        hidden_states_142 = None
        x_188 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_23_modules_mlp_modules_c_fc_parameters_bias_,
            view_262,
            l_self_modules_transformer_modules_h_modules_23_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_23_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_262
        ) = l_self_modules_transformer_modules_h_modules_23_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_189 = x_188.view((1, 39, 8192))
        x_188 = None
        mul_138 = 0.5 * x_189
        mul_139 = x_189 * 0.7978845608
        mul_140 = 0.044715 * x_189
        mul_141 = mul_140 * x_189
        mul_140 = x_189 = None
        add_94 = 1.0 + mul_141
        mul_141 = None
        mul_142 = mul_139 * add_94
        mul_139 = add_94 = None
        tanh_23 = torch.tanh(mul_142)
        mul_142 = None
        add_95 = 1.0 + tanh_23
        tanh_23 = None
        hidden_states_143 = mul_138 * add_95
        mul_138 = add_95 = None
        view_264 = hidden_states_143.view(-1, 8192)
        hidden_states_143 = None
        x_190 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_23_modules_mlp_modules_c_proj_parameters_bias_,
            view_264,
            l_self_modules_transformer_modules_h_modules_23_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_23_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_264
        ) = l_self_modules_transformer_modules_h_modules_23_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_191 = x_190.view((1, 39, 2048))
        x_190 = None
        hidden_states_144 = torch.nn.functional.dropout(x_191, 0.1, False, False)
        x_191 = None
        hidden_states_145 = hidden_states_141 + hidden_states_144
        hidden_states_141 = hidden_states_144 = None
        hidden_states_146 = torch.nn.functional.layer_norm(
            hidden_states_145,
            (2048,),
            l_self_modules_transformer_modules_ln_f_parameters_weight_,
            l_self_modules_transformer_modules_ln_f_parameters_bias_,
            1e-05,
        )
        hidden_states_145 = (
            l_self_modules_transformer_modules_ln_f_parameters_weight_
        ) = l_self_modules_transformer_modules_ln_f_parameters_bias_ = None
        hidden_states_147 = hidden_states_146.view((-1, 39, 2048))
        hidden_states_146 = None
        getitem_96 = hidden_states_147[
            (slice(None, None, None), slice(0, None, None), slice(None, None, None))
        ]
        hidden_states_147 = None
        logits = torch._C._nn.linear(
            getitem_96, l_self_modules_transformer_modules_wte_parameters_weight_, None
        )
        getitem_96 = l_self_modules_transformer_modules_wte_parameters_weight_ = None
        return (logits,)
