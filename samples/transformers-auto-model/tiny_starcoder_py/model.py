import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_self_modules_transformer_modules_wte_parameters_weight_: torch.nn.parameter.Parameter,
        L_attention_mask_: torch.Tensor,
        L_self_modules_transformer_modules_wpe_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_12_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_12_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_12_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_12_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_13_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_13_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_13_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_13_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_14_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_14_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_14_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_14_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_15_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_15_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_15_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_15_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_16_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_16_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_16_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_16_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_17_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_17_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_17_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_17_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_18_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_18_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_18_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_18_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_19_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_19_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_19_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_19_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_ln_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_ln_f_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_input_ids_ = L_input_ids_
        l_self_modules_transformer_modules_wte_parameters_weight_ = (
            L_self_modules_transformer_modules_wte_parameters_weight_
        )
        l_attention_mask_ = L_attention_mask_
        l_self_modules_transformer_modules_wpe_parameters_weight_ = (
            L_self_modules_transformer_modules_wpe_parameters_weight_
        )
        l_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_6_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_6_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_6_modules_ln_2_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_6_modules_ln_2_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_7_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_7_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_7_modules_ln_2_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_7_modules_ln_2_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_8_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_8_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_8_modules_ln_2_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_8_modules_ln_2_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_9_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_9_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_9_modules_ln_2_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_9_modules_ln_2_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_bias_ = L_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_bias_
        l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_10_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_10_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_10_modules_ln_2_parameters_bias_ = L_self_modules_transformer_modules_h_modules_10_modules_ln_2_parameters_bias_
        l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_bias_ = L_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_bias_
        l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_11_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_11_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_11_modules_ln_2_parameters_bias_ = L_self_modules_transformer_modules_h_modules_11_modules_ln_2_parameters_bias_
        l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_12_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_12_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_12_modules_ln_1_parameters_bias_ = L_self_modules_transformer_modules_h_modules_12_modules_ln_1_parameters_bias_
        l_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_12_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_12_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_12_modules_ln_2_parameters_bias_ = L_self_modules_transformer_modules_h_modules_12_modules_ln_2_parameters_bias_
        l_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_13_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_13_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_13_modules_ln_1_parameters_bias_ = L_self_modules_transformer_modules_h_modules_13_modules_ln_1_parameters_bias_
        l_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_13_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_13_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_13_modules_ln_2_parameters_bias_ = L_self_modules_transformer_modules_h_modules_13_modules_ln_2_parameters_bias_
        l_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_14_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_14_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_14_modules_ln_1_parameters_bias_ = L_self_modules_transformer_modules_h_modules_14_modules_ln_1_parameters_bias_
        l_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_14_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_14_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_14_modules_ln_2_parameters_bias_ = L_self_modules_transformer_modules_h_modules_14_modules_ln_2_parameters_bias_
        l_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_15_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_15_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_15_modules_ln_1_parameters_bias_ = L_self_modules_transformer_modules_h_modules_15_modules_ln_1_parameters_bias_
        l_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_15_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_15_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_15_modules_ln_2_parameters_bias_ = L_self_modules_transformer_modules_h_modules_15_modules_ln_2_parameters_bias_
        l_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_16_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_16_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_16_modules_ln_1_parameters_bias_ = L_self_modules_transformer_modules_h_modules_16_modules_ln_1_parameters_bias_
        l_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_16_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_16_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_16_modules_ln_2_parameters_bias_ = L_self_modules_transformer_modules_h_modules_16_modules_ln_2_parameters_bias_
        l_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_17_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_17_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_17_modules_ln_1_parameters_bias_ = L_self_modules_transformer_modules_h_modules_17_modules_ln_1_parameters_bias_
        l_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_17_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_17_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_17_modules_ln_2_parameters_bias_ = L_self_modules_transformer_modules_h_modules_17_modules_ln_2_parameters_bias_
        l_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_18_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_18_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_18_modules_ln_1_parameters_bias_ = L_self_modules_transformer_modules_h_modules_18_modules_ln_1_parameters_bias_
        l_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_18_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_18_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_18_modules_ln_2_parameters_bias_ = L_self_modules_transformer_modules_h_modules_18_modules_ln_2_parameters_bias_
        l_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_19_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_19_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_19_modules_ln_1_parameters_bias_ = L_self_modules_transformer_modules_h_modules_19_modules_ln_1_parameters_bias_
        l_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_19_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_19_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_19_modules_ln_2_parameters_bias_ = L_self_modules_transformer_modules_h_modules_19_modules_ln_2_parameters_bias_
        l_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_ln_f_parameters_weight_ = (
            L_self_modules_transformer_modules_ln_f_parameters_weight_
        )
        l_self_modules_transformer_modules_ln_f_parameters_bias_ = (
            L_self_modules_transformer_modules_ln_f_parameters_bias_
        )
        input_ids = l_input_ids_.view(-1, 19)
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
        cache_position = torch.arange(0, 19, device=device(type="cpu"))
        position_ids = cache_position.unsqueeze(0)
        attention_mask = l_attention_mask_.to(
            device=device(type="cpu"), dtype=torch.bool
        )
        l_attention_mask_ = None
        kv_arange = torch.arange(19, device=device(type="cpu"))
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
            19, "error"
        )
        _vmap_increment_nesting_2 = None
        child_2 = torch._C._functorch._add_batch_dim(cache_position, 0, 3)
        cache_position = None
        lazy_load_decompositions_3 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_3 = None
        _vmap_increment_nesting_3 = torch._C._functorch._vmap_increment_nesting(
            19, "error"
        )
        _vmap_increment_nesting_3 = None
        child_3 = torch._C._functorch._add_batch_dim(kv_arange_1, 0, 4)
        kv_arange_1 = None
        result = child_2.new_ones((), dtype=torch.bool)
        le = child_3.le(child_2)
        child_2 = None
        result_1 = result.__and__(le)
        result = le = None
        function_ctx = torch.autograd.function.FunctionCtx()
        function_ctx = None
        index = torch.ops.aten.index(attention_mask, [child, child_3])
        attention_mask = child = child_3 = None
        result_2 = result_1.__and__(index)
        result_1 = index = None
        batched_outputs = torch._C._functorch._remove_batch_dim(result_2, 4, 19, 0)
        result_2 = None
        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting = None
        batched_outputs_1 = torch._C._functorch._remove_batch_dim(
            batched_outputs, 3, 19, 0
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
        to_1 = position_embeds.to(device(type="cpu"))
        position_embeds = None
        hidden_states = inputs_embeds + to_1
        inputs_embeds = to_1 = None
        hidden_states_1 = torch.nn.functional.dropout(hidden_states, 0.1, False, False)
        hidden_states = None
        hidden_states_2 = torch.nn.functional.layer_norm(
            hidden_states_1,
            (768,),
            l_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_bias_
        ) = None
        linear = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_,
        )
        hidden_states_2 = l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_ = l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_ = (None)
        unsqueeze_1 = linear.unsqueeze(1)
        linear = None
        split = unsqueeze_1.split((768, 64, 64), dim=3)
        unsqueeze_1 = None
        query = split[0]
        key = split[1]
        value = split[2]
        split = None
        view_1 = query.view(1, 19, -1, 64)
        query = None
        query_1 = view_1.transpose(1, 2)
        view_1 = None
        getitem_3 = key[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        key = None
        hidden_states_3 = getitem_3.expand(1, 1, 12, 19, 64)
        getitem_3 = None
        key_1 = hidden_states_3.reshape(1, 12, 19, 64)
        hidden_states_3 = None
        getitem_4 = value[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value = None
        hidden_states_4 = getitem_4.expand(1, 1, 12, 19, 64)
        getitem_4 = None
        value_1 = hidden_states_4.reshape(1, 12, 19, 64)
        hidden_states_4 = None
        attention_mask_1 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_2 = query_1.contiguous()
        query_1 = None
        key_2 = key_1.contiguous()
        key_1 = None
        value_2 = value_1.contiguous()
        value_1 = None
        attn_output = torch._C._nn.scaled_dot_product_attention(
            query_2,
            key_2,
            value_2,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=8.0,
            is_causal=False,
        )
        query_2 = key_2 = value_2 = attention_mask_1 = None
        transpose_1 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_1.contiguous()
        transpose_1 = None
        reshape_2 = attn_output_1.reshape(1, 19, -1)
        attn_output_1 = None
        attn_output_2 = reshape_2.contiguous()
        reshape_2 = None
        attn_output_3 = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_,
        )
        attn_output_2 = l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_ = (None)
        attn_output_4 = torch.nn.functional.dropout(attn_output_3, 0.1, False, False)
        attn_output_3 = None
        hidden_states_5 = attn_output_4 + hidden_states_1
        attn_output_4 = hidden_states_1 = None
        hidden_states_6 = torch.nn.functional.layer_norm(
            hidden_states_5,
            (768,),
            l_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_bias_
        ) = None
        hidden_states_7 = torch._C._nn.linear(
            hidden_states_6,
            l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_,
        )
        hidden_states_6 = l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_ = l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_ = (None)
        hidden_states_8 = torch._C._nn.gelu(hidden_states_7, approximate="tanh")
        hidden_states_7 = None
        hidden_states_9 = torch._C._nn.linear(
            hidden_states_8,
            l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_,
        )
        hidden_states_8 = l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_ = (None)
        hidden_states_10 = torch.nn.functional.dropout(
            hidden_states_9, 0.1, False, False
        )
        hidden_states_9 = None
        hidden_states_11 = hidden_states_5 + hidden_states_10
        hidden_states_5 = hidden_states_10 = None
        hidden_states_12 = torch.nn.functional.layer_norm(
            hidden_states_11,
            (768,),
            l_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_bias_
        ) = None
        linear_4 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_,
        )
        hidden_states_12 = l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_ = l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_ = (None)
        unsqueeze_2 = linear_4.unsqueeze(1)
        linear_4 = None
        split_1 = unsqueeze_2.split((768, 64, 64), dim=3)
        unsqueeze_2 = None
        query_3 = split_1[0]
        key_3 = split_1[1]
        value_3 = split_1[2]
        split_1 = None
        view_2 = query_3.view(1, 19, -1, 64)
        query_3 = None
        query_4 = view_2.transpose(1, 2)
        view_2 = None
        getitem_9 = key_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        key_3 = None
        hidden_states_13 = getitem_9.expand(1, 1, 12, 19, 64)
        getitem_9 = None
        key_4 = hidden_states_13.reshape(1, 12, 19, 64)
        hidden_states_13 = None
        getitem_10 = value_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_3 = None
        hidden_states_14 = getitem_10.expand(1, 1, 12, 19, 64)
        getitem_10 = None
        value_4 = hidden_states_14.reshape(1, 12, 19, 64)
        hidden_states_14 = None
        attention_mask_2 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_5 = query_4.contiguous()
        query_4 = None
        key_5 = key_4.contiguous()
        key_4 = None
        value_5 = value_4.contiguous()
        value_4 = None
        attn_output_5 = torch._C._nn.scaled_dot_product_attention(
            query_5,
            key_5,
            value_5,
            attn_mask=attention_mask_2,
            dropout_p=0.0,
            scale=8.0,
            is_causal=False,
        )
        query_5 = key_5 = value_5 = attention_mask_2 = None
        transpose_3 = attn_output_5.transpose(1, 2)
        attn_output_5 = None
        attn_output_6 = transpose_3.contiguous()
        transpose_3 = None
        reshape_5 = attn_output_6.reshape(1, 19, -1)
        attn_output_6 = None
        attn_output_7 = reshape_5.contiguous()
        reshape_5 = None
        attn_output_8 = torch._C._nn.linear(
            attn_output_7,
            l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_,
        )
        attn_output_7 = l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_ = (None)
        attn_output_9 = torch.nn.functional.dropout(attn_output_8, 0.1, False, False)
        attn_output_8 = None
        hidden_states_15 = attn_output_9 + hidden_states_11
        attn_output_9 = hidden_states_11 = None
        hidden_states_16 = torch.nn.functional.layer_norm(
            hidden_states_15,
            (768,),
            l_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_bias_
        ) = None
        hidden_states_17 = torch._C._nn.linear(
            hidden_states_16,
            l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_,
        )
        hidden_states_16 = l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_ = l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_ = (None)
        hidden_states_18 = torch._C._nn.gelu(hidden_states_17, approximate="tanh")
        hidden_states_17 = None
        hidden_states_19 = torch._C._nn.linear(
            hidden_states_18,
            l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_,
        )
        hidden_states_18 = l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_ = (None)
        hidden_states_20 = torch.nn.functional.dropout(
            hidden_states_19, 0.1, False, False
        )
        hidden_states_19 = None
        hidden_states_21 = hidden_states_15 + hidden_states_20
        hidden_states_15 = hidden_states_20 = None
        hidden_states_22 = torch.nn.functional.layer_norm(
            hidden_states_21,
            (768,),
            l_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_bias_
        ) = None
        linear_8 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_,
        )
        hidden_states_22 = l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_ = l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_ = (None)
        unsqueeze_3 = linear_8.unsqueeze(1)
        linear_8 = None
        split_2 = unsqueeze_3.split((768, 64, 64), dim=3)
        unsqueeze_3 = None
        query_6 = split_2[0]
        key_6 = split_2[1]
        value_6 = split_2[2]
        split_2 = None
        view_3 = query_6.view(1, 19, -1, 64)
        query_6 = None
        query_7 = view_3.transpose(1, 2)
        view_3 = None
        getitem_15 = key_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        key_6 = None
        hidden_states_23 = getitem_15.expand(1, 1, 12, 19, 64)
        getitem_15 = None
        key_7 = hidden_states_23.reshape(1, 12, 19, 64)
        hidden_states_23 = None
        getitem_16 = value_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_6 = None
        hidden_states_24 = getitem_16.expand(1, 1, 12, 19, 64)
        getitem_16 = None
        value_7 = hidden_states_24.reshape(1, 12, 19, 64)
        hidden_states_24 = None
        attention_mask_3 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_8 = query_7.contiguous()
        query_7 = None
        key_8 = key_7.contiguous()
        key_7 = None
        value_8 = value_7.contiguous()
        value_7 = None
        attn_output_10 = torch._C._nn.scaled_dot_product_attention(
            query_8,
            key_8,
            value_8,
            attn_mask=attention_mask_3,
            dropout_p=0.0,
            scale=8.0,
            is_causal=False,
        )
        query_8 = key_8 = value_8 = attention_mask_3 = None
        transpose_5 = attn_output_10.transpose(1, 2)
        attn_output_10 = None
        attn_output_11 = transpose_5.contiguous()
        transpose_5 = None
        reshape_8 = attn_output_11.reshape(1, 19, -1)
        attn_output_11 = None
        attn_output_12 = reshape_8.contiguous()
        reshape_8 = None
        attn_output_13 = torch._C._nn.linear(
            attn_output_12,
            l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_,
        )
        attn_output_12 = l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_ = (None)
        attn_output_14 = torch.nn.functional.dropout(attn_output_13, 0.1, False, False)
        attn_output_13 = None
        hidden_states_25 = attn_output_14 + hidden_states_21
        attn_output_14 = hidden_states_21 = None
        hidden_states_26 = torch.nn.functional.layer_norm(
            hidden_states_25,
            (768,),
            l_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_bias_
        ) = None
        hidden_states_27 = torch._C._nn.linear(
            hidden_states_26,
            l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_,
        )
        hidden_states_26 = l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_ = l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_ = (None)
        hidden_states_28 = torch._C._nn.gelu(hidden_states_27, approximate="tanh")
        hidden_states_27 = None
        hidden_states_29 = torch._C._nn.linear(
            hidden_states_28,
            l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_,
        )
        hidden_states_28 = l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_ = (None)
        hidden_states_30 = torch.nn.functional.dropout(
            hidden_states_29, 0.1, False, False
        )
        hidden_states_29 = None
        hidden_states_31 = hidden_states_25 + hidden_states_30
        hidden_states_25 = hidden_states_30 = None
        hidden_states_32 = torch.nn.functional.layer_norm(
            hidden_states_31,
            (768,),
            l_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_bias_
        ) = None
        linear_12 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_,
        )
        hidden_states_32 = l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_ = l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_ = (None)
        unsqueeze_4 = linear_12.unsqueeze(1)
        linear_12 = None
        split_3 = unsqueeze_4.split((768, 64, 64), dim=3)
        unsqueeze_4 = None
        query_9 = split_3[0]
        key_9 = split_3[1]
        value_9 = split_3[2]
        split_3 = None
        view_4 = query_9.view(1, 19, -1, 64)
        query_9 = None
        query_10 = view_4.transpose(1, 2)
        view_4 = None
        getitem_21 = key_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        key_9 = None
        hidden_states_33 = getitem_21.expand(1, 1, 12, 19, 64)
        getitem_21 = None
        key_10 = hidden_states_33.reshape(1, 12, 19, 64)
        hidden_states_33 = None
        getitem_22 = value_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_9 = None
        hidden_states_34 = getitem_22.expand(1, 1, 12, 19, 64)
        getitem_22 = None
        value_10 = hidden_states_34.reshape(1, 12, 19, 64)
        hidden_states_34 = None
        attention_mask_4 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_11 = query_10.contiguous()
        query_10 = None
        key_11 = key_10.contiguous()
        key_10 = None
        value_11 = value_10.contiguous()
        value_10 = None
        attn_output_15 = torch._C._nn.scaled_dot_product_attention(
            query_11,
            key_11,
            value_11,
            attn_mask=attention_mask_4,
            dropout_p=0.0,
            scale=8.0,
            is_causal=False,
        )
        query_11 = key_11 = value_11 = attention_mask_4 = None
        transpose_7 = attn_output_15.transpose(1, 2)
        attn_output_15 = None
        attn_output_16 = transpose_7.contiguous()
        transpose_7 = None
        reshape_11 = attn_output_16.reshape(1, 19, -1)
        attn_output_16 = None
        attn_output_17 = reshape_11.contiguous()
        reshape_11 = None
        attn_output_18 = torch._C._nn.linear(
            attn_output_17,
            l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_,
        )
        attn_output_17 = l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_ = (None)
        attn_output_19 = torch.nn.functional.dropout(attn_output_18, 0.1, False, False)
        attn_output_18 = None
        hidden_states_35 = attn_output_19 + hidden_states_31
        attn_output_19 = hidden_states_31 = None
        hidden_states_36 = torch.nn.functional.layer_norm(
            hidden_states_35,
            (768,),
            l_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_bias_
        ) = None
        hidden_states_37 = torch._C._nn.linear(
            hidden_states_36,
            l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_,
        )
        hidden_states_36 = l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_ = l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_ = (None)
        hidden_states_38 = torch._C._nn.gelu(hidden_states_37, approximate="tanh")
        hidden_states_37 = None
        hidden_states_39 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_,
        )
        hidden_states_38 = l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_ = (None)
        hidden_states_40 = torch.nn.functional.dropout(
            hidden_states_39, 0.1, False, False
        )
        hidden_states_39 = None
        hidden_states_41 = hidden_states_35 + hidden_states_40
        hidden_states_35 = hidden_states_40 = None
        hidden_states_42 = torch.nn.functional.layer_norm(
            hidden_states_41,
            (768,),
            l_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_bias_
        ) = None
        linear_16 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_,
        )
        hidden_states_42 = l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_ = l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_ = (None)
        unsqueeze_5 = linear_16.unsqueeze(1)
        linear_16 = None
        split_4 = unsqueeze_5.split((768, 64, 64), dim=3)
        unsqueeze_5 = None
        query_12 = split_4[0]
        key_12 = split_4[1]
        value_12 = split_4[2]
        split_4 = None
        view_5 = query_12.view(1, 19, -1, 64)
        query_12 = None
        query_13 = view_5.transpose(1, 2)
        view_5 = None
        getitem_27 = key_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        key_12 = None
        hidden_states_43 = getitem_27.expand(1, 1, 12, 19, 64)
        getitem_27 = None
        key_13 = hidden_states_43.reshape(1, 12, 19, 64)
        hidden_states_43 = None
        getitem_28 = value_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_12 = None
        hidden_states_44 = getitem_28.expand(1, 1, 12, 19, 64)
        getitem_28 = None
        value_13 = hidden_states_44.reshape(1, 12, 19, 64)
        hidden_states_44 = None
        attention_mask_5 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_14 = query_13.contiguous()
        query_13 = None
        key_14 = key_13.contiguous()
        key_13 = None
        value_14 = value_13.contiguous()
        value_13 = None
        attn_output_20 = torch._C._nn.scaled_dot_product_attention(
            query_14,
            key_14,
            value_14,
            attn_mask=attention_mask_5,
            dropout_p=0.0,
            scale=8.0,
            is_causal=False,
        )
        query_14 = key_14 = value_14 = attention_mask_5 = None
        transpose_9 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_9.contiguous()
        transpose_9 = None
        reshape_14 = attn_output_21.reshape(1, 19, -1)
        attn_output_21 = None
        attn_output_22 = reshape_14.contiguous()
        reshape_14 = None
        attn_output_23 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_,
        )
        attn_output_22 = l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_ = (None)
        attn_output_24 = torch.nn.functional.dropout(attn_output_23, 0.1, False, False)
        attn_output_23 = None
        hidden_states_45 = attn_output_24 + hidden_states_41
        attn_output_24 = hidden_states_41 = None
        hidden_states_46 = torch.nn.functional.layer_norm(
            hidden_states_45,
            (768,),
            l_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_bias_
        ) = None
        hidden_states_47 = torch._C._nn.linear(
            hidden_states_46,
            l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_,
        )
        hidden_states_46 = l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_ = l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_ = (None)
        hidden_states_48 = torch._C._nn.gelu(hidden_states_47, approximate="tanh")
        hidden_states_47 = None
        hidden_states_49 = torch._C._nn.linear(
            hidden_states_48,
            l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_,
        )
        hidden_states_48 = l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_ = (None)
        hidden_states_50 = torch.nn.functional.dropout(
            hidden_states_49, 0.1, False, False
        )
        hidden_states_49 = None
        hidden_states_51 = hidden_states_45 + hidden_states_50
        hidden_states_45 = hidden_states_50 = None
        hidden_states_52 = torch.nn.functional.layer_norm(
            hidden_states_51,
            (768,),
            l_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_bias_
        ) = None
        linear_20 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_bias_,
        )
        hidden_states_52 = l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_weight_ = l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_bias_ = (None)
        unsqueeze_6 = linear_20.unsqueeze(1)
        linear_20 = None
        split_5 = unsqueeze_6.split((768, 64, 64), dim=3)
        unsqueeze_6 = None
        query_15 = split_5[0]
        key_15 = split_5[1]
        value_15 = split_5[2]
        split_5 = None
        view_6 = query_15.view(1, 19, -1, 64)
        query_15 = None
        query_16 = view_6.transpose(1, 2)
        view_6 = None
        getitem_33 = key_15[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        key_15 = None
        hidden_states_53 = getitem_33.expand(1, 1, 12, 19, 64)
        getitem_33 = None
        key_16 = hidden_states_53.reshape(1, 12, 19, 64)
        hidden_states_53 = None
        getitem_34 = value_15[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_15 = None
        hidden_states_54 = getitem_34.expand(1, 1, 12, 19, 64)
        getitem_34 = None
        value_16 = hidden_states_54.reshape(1, 12, 19, 64)
        hidden_states_54 = None
        attention_mask_6 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_17 = query_16.contiguous()
        query_16 = None
        key_17 = key_16.contiguous()
        key_16 = None
        value_17 = value_16.contiguous()
        value_16 = None
        attn_output_25 = torch._C._nn.scaled_dot_product_attention(
            query_17,
            key_17,
            value_17,
            attn_mask=attention_mask_6,
            dropout_p=0.0,
            scale=8.0,
            is_causal=False,
        )
        query_17 = key_17 = value_17 = attention_mask_6 = None
        transpose_11 = attn_output_25.transpose(1, 2)
        attn_output_25 = None
        attn_output_26 = transpose_11.contiguous()
        transpose_11 = None
        reshape_17 = attn_output_26.reshape(1, 19, -1)
        attn_output_26 = None
        attn_output_27 = reshape_17.contiguous()
        reshape_17 = None
        attn_output_28 = torch._C._nn.linear(
            attn_output_27,
            l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_bias_,
        )
        attn_output_27 = l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_bias_ = (None)
        attn_output_29 = torch.nn.functional.dropout(attn_output_28, 0.1, False, False)
        attn_output_28 = None
        hidden_states_55 = attn_output_29 + hidden_states_51
        attn_output_29 = hidden_states_51 = None
        hidden_states_56 = torch.nn.functional.layer_norm(
            hidden_states_55,
            (768,),
            l_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_bias_
        ) = None
        hidden_states_57 = torch._C._nn.linear(
            hidden_states_56,
            l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_bias_,
        )
        hidden_states_56 = l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_weight_ = l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_bias_ = (None)
        hidden_states_58 = torch._C._nn.gelu(hidden_states_57, approximate="tanh")
        hidden_states_57 = None
        hidden_states_59 = torch._C._nn.linear(
            hidden_states_58,
            l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_bias_,
        )
        hidden_states_58 = l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_bias_ = (None)
        hidden_states_60 = torch.nn.functional.dropout(
            hidden_states_59, 0.1, False, False
        )
        hidden_states_59 = None
        hidden_states_61 = hidden_states_55 + hidden_states_60
        hidden_states_55 = hidden_states_60 = None
        hidden_states_62 = torch.nn.functional.layer_norm(
            hidden_states_61,
            (768,),
            l_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_bias_
        ) = None
        linear_24 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_attn_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_attn_parameters_bias_,
        )
        hidden_states_62 = l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_attn_parameters_weight_ = l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_attn_parameters_bias_ = (None)
        unsqueeze_7 = linear_24.unsqueeze(1)
        linear_24 = None
        split_6 = unsqueeze_7.split((768, 64, 64), dim=3)
        unsqueeze_7 = None
        query_18 = split_6[0]
        key_18 = split_6[1]
        value_18 = split_6[2]
        split_6 = None
        view_7 = query_18.view(1, 19, -1, 64)
        query_18 = None
        query_19 = view_7.transpose(1, 2)
        view_7 = None
        getitem_39 = key_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        key_18 = None
        hidden_states_63 = getitem_39.expand(1, 1, 12, 19, 64)
        getitem_39 = None
        key_19 = hidden_states_63.reshape(1, 12, 19, 64)
        hidden_states_63 = None
        getitem_40 = value_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_18 = None
        hidden_states_64 = getitem_40.expand(1, 1, 12, 19, 64)
        getitem_40 = None
        value_19 = hidden_states_64.reshape(1, 12, 19, 64)
        hidden_states_64 = None
        attention_mask_7 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_20 = query_19.contiguous()
        query_19 = None
        key_20 = key_19.contiguous()
        key_19 = None
        value_20 = value_19.contiguous()
        value_19 = None
        attn_output_30 = torch._C._nn.scaled_dot_product_attention(
            query_20,
            key_20,
            value_20,
            attn_mask=attention_mask_7,
            dropout_p=0.0,
            scale=8.0,
            is_causal=False,
        )
        query_20 = key_20 = value_20 = attention_mask_7 = None
        transpose_13 = attn_output_30.transpose(1, 2)
        attn_output_30 = None
        attn_output_31 = transpose_13.contiguous()
        transpose_13 = None
        reshape_20 = attn_output_31.reshape(1, 19, -1)
        attn_output_31 = None
        attn_output_32 = reshape_20.contiguous()
        reshape_20 = None
        attn_output_33 = torch._C._nn.linear(
            attn_output_32,
            l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_proj_parameters_bias_,
        )
        attn_output_32 = l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_proj_parameters_bias_ = (None)
        attn_output_34 = torch.nn.functional.dropout(attn_output_33, 0.1, False, False)
        attn_output_33 = None
        hidden_states_65 = attn_output_34 + hidden_states_61
        attn_output_34 = hidden_states_61 = None
        hidden_states_66 = torch.nn.functional.layer_norm(
            hidden_states_65,
            (768,),
            l_self_modules_transformer_modules_h_modules_6_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_6_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_6_modules_ln_2_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_6_modules_ln_2_parameters_bias_
        ) = None
        hidden_states_67 = torch._C._nn.linear(
            hidden_states_66,
            l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_bias_,
        )
        hidden_states_66 = l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_weight_ = l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_bias_ = (None)
        hidden_states_68 = torch._C._nn.gelu(hidden_states_67, approximate="tanh")
        hidden_states_67 = None
        hidden_states_69 = torch._C._nn.linear(
            hidden_states_68,
            l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_bias_,
        )
        hidden_states_68 = l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_bias_ = (None)
        hidden_states_70 = torch.nn.functional.dropout(
            hidden_states_69, 0.1, False, False
        )
        hidden_states_69 = None
        hidden_states_71 = hidden_states_65 + hidden_states_70
        hidden_states_65 = hidden_states_70 = None
        hidden_states_72 = torch.nn.functional.layer_norm(
            hidden_states_71,
            (768,),
            l_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_bias_
        ) = None
        linear_28 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_attn_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_attn_parameters_bias_,
        )
        hidden_states_72 = l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_attn_parameters_weight_ = l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_attn_parameters_bias_ = (None)
        unsqueeze_8 = linear_28.unsqueeze(1)
        linear_28 = None
        split_7 = unsqueeze_8.split((768, 64, 64), dim=3)
        unsqueeze_8 = None
        query_21 = split_7[0]
        key_21 = split_7[1]
        value_21 = split_7[2]
        split_7 = None
        view_8 = query_21.view(1, 19, -1, 64)
        query_21 = None
        query_22 = view_8.transpose(1, 2)
        view_8 = None
        getitem_45 = key_21[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        key_21 = None
        hidden_states_73 = getitem_45.expand(1, 1, 12, 19, 64)
        getitem_45 = None
        key_22 = hidden_states_73.reshape(1, 12, 19, 64)
        hidden_states_73 = None
        getitem_46 = value_21[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_21 = None
        hidden_states_74 = getitem_46.expand(1, 1, 12, 19, 64)
        getitem_46 = None
        value_22 = hidden_states_74.reshape(1, 12, 19, 64)
        hidden_states_74 = None
        attention_mask_8 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_23 = query_22.contiguous()
        query_22 = None
        key_23 = key_22.contiguous()
        key_22 = None
        value_23 = value_22.contiguous()
        value_22 = None
        attn_output_35 = torch._C._nn.scaled_dot_product_attention(
            query_23,
            key_23,
            value_23,
            attn_mask=attention_mask_8,
            dropout_p=0.0,
            scale=8.0,
            is_causal=False,
        )
        query_23 = key_23 = value_23 = attention_mask_8 = None
        transpose_15 = attn_output_35.transpose(1, 2)
        attn_output_35 = None
        attn_output_36 = transpose_15.contiguous()
        transpose_15 = None
        reshape_23 = attn_output_36.reshape(1, 19, -1)
        attn_output_36 = None
        attn_output_37 = reshape_23.contiguous()
        reshape_23 = None
        attn_output_38 = torch._C._nn.linear(
            attn_output_37,
            l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_proj_parameters_bias_,
        )
        attn_output_37 = l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_proj_parameters_bias_ = (None)
        attn_output_39 = torch.nn.functional.dropout(attn_output_38, 0.1, False, False)
        attn_output_38 = None
        hidden_states_75 = attn_output_39 + hidden_states_71
        attn_output_39 = hidden_states_71 = None
        hidden_states_76 = torch.nn.functional.layer_norm(
            hidden_states_75,
            (768,),
            l_self_modules_transformer_modules_h_modules_7_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_7_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_7_modules_ln_2_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_7_modules_ln_2_parameters_bias_
        ) = None
        hidden_states_77 = torch._C._nn.linear(
            hidden_states_76,
            l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_bias_,
        )
        hidden_states_76 = l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_weight_ = l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_bias_ = (None)
        hidden_states_78 = torch._C._nn.gelu(hidden_states_77, approximate="tanh")
        hidden_states_77 = None
        hidden_states_79 = torch._C._nn.linear(
            hidden_states_78,
            l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_bias_,
        )
        hidden_states_78 = l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_bias_ = (None)
        hidden_states_80 = torch.nn.functional.dropout(
            hidden_states_79, 0.1, False, False
        )
        hidden_states_79 = None
        hidden_states_81 = hidden_states_75 + hidden_states_80
        hidden_states_75 = hidden_states_80 = None
        hidden_states_82 = torch.nn.functional.layer_norm(
            hidden_states_81,
            (768,),
            l_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_bias_
        ) = None
        linear_32 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_attn_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_attn_parameters_bias_,
        )
        hidden_states_82 = l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_attn_parameters_weight_ = l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_attn_parameters_bias_ = (None)
        unsqueeze_9 = linear_32.unsqueeze(1)
        linear_32 = None
        split_8 = unsqueeze_9.split((768, 64, 64), dim=3)
        unsqueeze_9 = None
        query_24 = split_8[0]
        key_24 = split_8[1]
        value_24 = split_8[2]
        split_8 = None
        view_9 = query_24.view(1, 19, -1, 64)
        query_24 = None
        query_25 = view_9.transpose(1, 2)
        view_9 = None
        getitem_51 = key_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        key_24 = None
        hidden_states_83 = getitem_51.expand(1, 1, 12, 19, 64)
        getitem_51 = None
        key_25 = hidden_states_83.reshape(1, 12, 19, 64)
        hidden_states_83 = None
        getitem_52 = value_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_24 = None
        hidden_states_84 = getitem_52.expand(1, 1, 12, 19, 64)
        getitem_52 = None
        value_25 = hidden_states_84.reshape(1, 12, 19, 64)
        hidden_states_84 = None
        attention_mask_9 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_26 = query_25.contiguous()
        query_25 = None
        key_26 = key_25.contiguous()
        key_25 = None
        value_26 = value_25.contiguous()
        value_25 = None
        attn_output_40 = torch._C._nn.scaled_dot_product_attention(
            query_26,
            key_26,
            value_26,
            attn_mask=attention_mask_9,
            dropout_p=0.0,
            scale=8.0,
            is_causal=False,
        )
        query_26 = key_26 = value_26 = attention_mask_9 = None
        transpose_17 = attn_output_40.transpose(1, 2)
        attn_output_40 = None
        attn_output_41 = transpose_17.contiguous()
        transpose_17 = None
        reshape_26 = attn_output_41.reshape(1, 19, -1)
        attn_output_41 = None
        attn_output_42 = reshape_26.contiguous()
        reshape_26 = None
        attn_output_43 = torch._C._nn.linear(
            attn_output_42,
            l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_proj_parameters_bias_,
        )
        attn_output_42 = l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_proj_parameters_bias_ = (None)
        attn_output_44 = torch.nn.functional.dropout(attn_output_43, 0.1, False, False)
        attn_output_43 = None
        hidden_states_85 = attn_output_44 + hidden_states_81
        attn_output_44 = hidden_states_81 = None
        hidden_states_86 = torch.nn.functional.layer_norm(
            hidden_states_85,
            (768,),
            l_self_modules_transformer_modules_h_modules_8_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_8_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_8_modules_ln_2_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_8_modules_ln_2_parameters_bias_
        ) = None
        hidden_states_87 = torch._C._nn.linear(
            hidden_states_86,
            l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_bias_,
        )
        hidden_states_86 = l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_weight_ = l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_bias_ = (None)
        hidden_states_88 = torch._C._nn.gelu(hidden_states_87, approximate="tanh")
        hidden_states_87 = None
        hidden_states_89 = torch._C._nn.linear(
            hidden_states_88,
            l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_bias_,
        )
        hidden_states_88 = l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_bias_ = (None)
        hidden_states_90 = torch.nn.functional.dropout(
            hidden_states_89, 0.1, False, False
        )
        hidden_states_89 = None
        hidden_states_91 = hidden_states_85 + hidden_states_90
        hidden_states_85 = hidden_states_90 = None
        hidden_states_92 = torch.nn.functional.layer_norm(
            hidden_states_91,
            (768,),
            l_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_bias_
        ) = None
        linear_36 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_attn_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_attn_parameters_bias_,
        )
        hidden_states_92 = l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_attn_parameters_weight_ = l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_attn_parameters_bias_ = (None)
        unsqueeze_10 = linear_36.unsqueeze(1)
        linear_36 = None
        split_9 = unsqueeze_10.split((768, 64, 64), dim=3)
        unsqueeze_10 = None
        query_27 = split_9[0]
        key_27 = split_9[1]
        value_27 = split_9[2]
        split_9 = None
        view_10 = query_27.view(1, 19, -1, 64)
        query_27 = None
        query_28 = view_10.transpose(1, 2)
        view_10 = None
        getitem_57 = key_27[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        key_27 = None
        hidden_states_93 = getitem_57.expand(1, 1, 12, 19, 64)
        getitem_57 = None
        key_28 = hidden_states_93.reshape(1, 12, 19, 64)
        hidden_states_93 = None
        getitem_58 = value_27[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_27 = None
        hidden_states_94 = getitem_58.expand(1, 1, 12, 19, 64)
        getitem_58 = None
        value_28 = hidden_states_94.reshape(1, 12, 19, 64)
        hidden_states_94 = None
        attention_mask_10 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_29 = query_28.contiguous()
        query_28 = None
        key_29 = key_28.contiguous()
        key_28 = None
        value_29 = value_28.contiguous()
        value_28 = None
        attn_output_45 = torch._C._nn.scaled_dot_product_attention(
            query_29,
            key_29,
            value_29,
            attn_mask=attention_mask_10,
            dropout_p=0.0,
            scale=8.0,
            is_causal=False,
        )
        query_29 = key_29 = value_29 = attention_mask_10 = None
        transpose_19 = attn_output_45.transpose(1, 2)
        attn_output_45 = None
        attn_output_46 = transpose_19.contiguous()
        transpose_19 = None
        reshape_29 = attn_output_46.reshape(1, 19, -1)
        attn_output_46 = None
        attn_output_47 = reshape_29.contiguous()
        reshape_29 = None
        attn_output_48 = torch._C._nn.linear(
            attn_output_47,
            l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_proj_parameters_bias_,
        )
        attn_output_47 = l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_proj_parameters_bias_ = (None)
        attn_output_49 = torch.nn.functional.dropout(attn_output_48, 0.1, False, False)
        attn_output_48 = None
        hidden_states_95 = attn_output_49 + hidden_states_91
        attn_output_49 = hidden_states_91 = None
        hidden_states_96 = torch.nn.functional.layer_norm(
            hidden_states_95,
            (768,),
            l_self_modules_transformer_modules_h_modules_9_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_9_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_9_modules_ln_2_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_9_modules_ln_2_parameters_bias_
        ) = None
        hidden_states_97 = torch._C._nn.linear(
            hidden_states_96,
            l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_bias_,
        )
        hidden_states_96 = l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_weight_ = l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_bias_ = (None)
        hidden_states_98 = torch._C._nn.gelu(hidden_states_97, approximate="tanh")
        hidden_states_97 = None
        hidden_states_99 = torch._C._nn.linear(
            hidden_states_98,
            l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_bias_,
        )
        hidden_states_98 = l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_bias_ = (None)
        hidden_states_100 = torch.nn.functional.dropout(
            hidden_states_99, 0.1, False, False
        )
        hidden_states_99 = None
        hidden_states_101 = hidden_states_95 + hidden_states_100
        hidden_states_95 = hidden_states_100 = None
        hidden_states_102 = torch.nn.functional.layer_norm(
            hidden_states_101,
            (768,),
            l_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_bias_ = (None)
        linear_40 = torch._C._nn.linear(
            hidden_states_102,
            l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_attn_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_attn_parameters_bias_,
        )
        hidden_states_102 = l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_attn_parameters_weight_ = l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_attn_parameters_bias_ = (None)
        unsqueeze_11 = linear_40.unsqueeze(1)
        linear_40 = None
        split_10 = unsqueeze_11.split((768, 64, 64), dim=3)
        unsqueeze_11 = None
        query_30 = split_10[0]
        key_30 = split_10[1]
        value_30 = split_10[2]
        split_10 = None
        view_11 = query_30.view(1, 19, -1, 64)
        query_30 = None
        query_31 = view_11.transpose(1, 2)
        view_11 = None
        getitem_63 = key_30[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        key_30 = None
        hidden_states_103 = getitem_63.expand(1, 1, 12, 19, 64)
        getitem_63 = None
        key_31 = hidden_states_103.reshape(1, 12, 19, 64)
        hidden_states_103 = None
        getitem_64 = value_30[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_30 = None
        hidden_states_104 = getitem_64.expand(1, 1, 12, 19, 64)
        getitem_64 = None
        value_31 = hidden_states_104.reshape(1, 12, 19, 64)
        hidden_states_104 = None
        attention_mask_11 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_32 = query_31.contiguous()
        query_31 = None
        key_32 = key_31.contiguous()
        key_31 = None
        value_32 = value_31.contiguous()
        value_31 = None
        attn_output_50 = torch._C._nn.scaled_dot_product_attention(
            query_32,
            key_32,
            value_32,
            attn_mask=attention_mask_11,
            dropout_p=0.0,
            scale=8.0,
            is_causal=False,
        )
        query_32 = key_32 = value_32 = attention_mask_11 = None
        transpose_21 = attn_output_50.transpose(1, 2)
        attn_output_50 = None
        attn_output_51 = transpose_21.contiguous()
        transpose_21 = None
        reshape_32 = attn_output_51.reshape(1, 19, -1)
        attn_output_51 = None
        attn_output_52 = reshape_32.contiguous()
        reshape_32 = None
        attn_output_53 = torch._C._nn.linear(
            attn_output_52,
            l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_proj_parameters_bias_,
        )
        attn_output_52 = l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_proj_parameters_bias_ = (None)
        attn_output_54 = torch.nn.functional.dropout(attn_output_53, 0.1, False, False)
        attn_output_53 = None
        hidden_states_105 = attn_output_54 + hidden_states_101
        attn_output_54 = hidden_states_101 = None
        hidden_states_106 = torch.nn.functional.layer_norm(
            hidden_states_105,
            (768,),
            l_self_modules_transformer_modules_h_modules_10_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_10_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_10_modules_ln_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_10_modules_ln_2_parameters_bias_ = (None)
        hidden_states_107 = torch._C._nn.linear(
            hidden_states_106,
            l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_bias_,
        )
        hidden_states_106 = l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_weight_ = l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_bias_ = (None)
        hidden_states_108 = torch._C._nn.gelu(hidden_states_107, approximate="tanh")
        hidden_states_107 = None
        hidden_states_109 = torch._C._nn.linear(
            hidden_states_108,
            l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_bias_,
        )
        hidden_states_108 = l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_bias_ = (None)
        hidden_states_110 = torch.nn.functional.dropout(
            hidden_states_109, 0.1, False, False
        )
        hidden_states_109 = None
        hidden_states_111 = hidden_states_105 + hidden_states_110
        hidden_states_105 = hidden_states_110 = None
        hidden_states_112 = torch.nn.functional.layer_norm(
            hidden_states_111,
            (768,),
            l_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_bias_ = (None)
        linear_44 = torch._C._nn.linear(
            hidden_states_112,
            l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_attn_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_attn_parameters_bias_,
        )
        hidden_states_112 = l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_attn_parameters_weight_ = l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_attn_parameters_bias_ = (None)
        unsqueeze_12 = linear_44.unsqueeze(1)
        linear_44 = None
        split_11 = unsqueeze_12.split((768, 64, 64), dim=3)
        unsqueeze_12 = None
        query_33 = split_11[0]
        key_33 = split_11[1]
        value_33 = split_11[2]
        split_11 = None
        view_12 = query_33.view(1, 19, -1, 64)
        query_33 = None
        query_34 = view_12.transpose(1, 2)
        view_12 = None
        getitem_69 = key_33[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        key_33 = None
        hidden_states_113 = getitem_69.expand(1, 1, 12, 19, 64)
        getitem_69 = None
        key_34 = hidden_states_113.reshape(1, 12, 19, 64)
        hidden_states_113 = None
        getitem_70 = value_33[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_33 = None
        hidden_states_114 = getitem_70.expand(1, 1, 12, 19, 64)
        getitem_70 = None
        value_34 = hidden_states_114.reshape(1, 12, 19, 64)
        hidden_states_114 = None
        attention_mask_12 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_35 = query_34.contiguous()
        query_34 = None
        key_35 = key_34.contiguous()
        key_34 = None
        value_35 = value_34.contiguous()
        value_34 = None
        attn_output_55 = torch._C._nn.scaled_dot_product_attention(
            query_35,
            key_35,
            value_35,
            attn_mask=attention_mask_12,
            dropout_p=0.0,
            scale=8.0,
            is_causal=False,
        )
        query_35 = key_35 = value_35 = attention_mask_12 = None
        transpose_23 = attn_output_55.transpose(1, 2)
        attn_output_55 = None
        attn_output_56 = transpose_23.contiguous()
        transpose_23 = None
        reshape_35 = attn_output_56.reshape(1, 19, -1)
        attn_output_56 = None
        attn_output_57 = reshape_35.contiguous()
        reshape_35 = None
        attn_output_58 = torch._C._nn.linear(
            attn_output_57,
            l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_proj_parameters_bias_,
        )
        attn_output_57 = l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_proj_parameters_bias_ = (None)
        attn_output_59 = torch.nn.functional.dropout(attn_output_58, 0.1, False, False)
        attn_output_58 = None
        hidden_states_115 = attn_output_59 + hidden_states_111
        attn_output_59 = hidden_states_111 = None
        hidden_states_116 = torch.nn.functional.layer_norm(
            hidden_states_115,
            (768,),
            l_self_modules_transformer_modules_h_modules_11_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_11_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_11_modules_ln_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_11_modules_ln_2_parameters_bias_ = (None)
        hidden_states_117 = torch._C._nn.linear(
            hidden_states_116,
            l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_bias_,
        )
        hidden_states_116 = l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_weight_ = l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_bias_ = (None)
        hidden_states_118 = torch._C._nn.gelu(hidden_states_117, approximate="tanh")
        hidden_states_117 = None
        hidden_states_119 = torch._C._nn.linear(
            hidden_states_118,
            l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_bias_,
        )
        hidden_states_118 = l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_bias_ = (None)
        hidden_states_120 = torch.nn.functional.dropout(
            hidden_states_119, 0.1, False, False
        )
        hidden_states_119 = None
        hidden_states_121 = hidden_states_115 + hidden_states_120
        hidden_states_115 = hidden_states_120 = None
        hidden_states_122 = torch.nn.functional.layer_norm(
            hidden_states_121,
            (768,),
            l_self_modules_transformer_modules_h_modules_12_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_12_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_12_modules_ln_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_12_modules_ln_1_parameters_bias_ = (None)
        linear_48 = torch._C._nn.linear(
            hidden_states_122,
            l_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_attn_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_attn_parameters_bias_,
        )
        hidden_states_122 = l_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_attn_parameters_weight_ = l_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_attn_parameters_bias_ = (None)
        unsqueeze_13 = linear_48.unsqueeze(1)
        linear_48 = None
        split_12 = unsqueeze_13.split((768, 64, 64), dim=3)
        unsqueeze_13 = None
        query_36 = split_12[0]
        key_36 = split_12[1]
        value_36 = split_12[2]
        split_12 = None
        view_13 = query_36.view(1, 19, -1, 64)
        query_36 = None
        query_37 = view_13.transpose(1, 2)
        view_13 = None
        getitem_75 = key_36[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        key_36 = None
        hidden_states_123 = getitem_75.expand(1, 1, 12, 19, 64)
        getitem_75 = None
        key_37 = hidden_states_123.reshape(1, 12, 19, 64)
        hidden_states_123 = None
        getitem_76 = value_36[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_36 = None
        hidden_states_124 = getitem_76.expand(1, 1, 12, 19, 64)
        getitem_76 = None
        value_37 = hidden_states_124.reshape(1, 12, 19, 64)
        hidden_states_124 = None
        attention_mask_13 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_38 = query_37.contiguous()
        query_37 = None
        key_38 = key_37.contiguous()
        key_37 = None
        value_38 = value_37.contiguous()
        value_37 = None
        attn_output_60 = torch._C._nn.scaled_dot_product_attention(
            query_38,
            key_38,
            value_38,
            attn_mask=attention_mask_13,
            dropout_p=0.0,
            scale=8.0,
            is_causal=False,
        )
        query_38 = key_38 = value_38 = attention_mask_13 = None
        transpose_25 = attn_output_60.transpose(1, 2)
        attn_output_60 = None
        attn_output_61 = transpose_25.contiguous()
        transpose_25 = None
        reshape_38 = attn_output_61.reshape(1, 19, -1)
        attn_output_61 = None
        attn_output_62 = reshape_38.contiguous()
        reshape_38 = None
        attn_output_63 = torch._C._nn.linear(
            attn_output_62,
            l_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_proj_parameters_bias_,
        )
        attn_output_62 = l_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_12_modules_attn_modules_c_proj_parameters_bias_ = (None)
        attn_output_64 = torch.nn.functional.dropout(attn_output_63, 0.1, False, False)
        attn_output_63 = None
        hidden_states_125 = attn_output_64 + hidden_states_121
        attn_output_64 = hidden_states_121 = None
        hidden_states_126 = torch.nn.functional.layer_norm(
            hidden_states_125,
            (768,),
            l_self_modules_transformer_modules_h_modules_12_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_12_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_12_modules_ln_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_12_modules_ln_2_parameters_bias_ = (None)
        hidden_states_127 = torch._C._nn.linear(
            hidden_states_126,
            l_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_fc_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_fc_parameters_bias_,
        )
        hidden_states_126 = l_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_fc_parameters_weight_ = l_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_fc_parameters_bias_ = (None)
        hidden_states_128 = torch._C._nn.gelu(hidden_states_127, approximate="tanh")
        hidden_states_127 = None
        hidden_states_129 = torch._C._nn.linear(
            hidden_states_128,
            l_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_proj_parameters_bias_,
        )
        hidden_states_128 = l_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_12_modules_mlp_modules_c_proj_parameters_bias_ = (None)
        hidden_states_130 = torch.nn.functional.dropout(
            hidden_states_129, 0.1, False, False
        )
        hidden_states_129 = None
        hidden_states_131 = hidden_states_125 + hidden_states_130
        hidden_states_125 = hidden_states_130 = None
        hidden_states_132 = torch.nn.functional.layer_norm(
            hidden_states_131,
            (768,),
            l_self_modules_transformer_modules_h_modules_13_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_13_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_13_modules_ln_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_13_modules_ln_1_parameters_bias_ = (None)
        linear_52 = torch._C._nn.linear(
            hidden_states_132,
            l_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_attn_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_attn_parameters_bias_,
        )
        hidden_states_132 = l_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_attn_parameters_weight_ = l_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_attn_parameters_bias_ = (None)
        unsqueeze_14 = linear_52.unsqueeze(1)
        linear_52 = None
        split_13 = unsqueeze_14.split((768, 64, 64), dim=3)
        unsqueeze_14 = None
        query_39 = split_13[0]
        key_39 = split_13[1]
        value_39 = split_13[2]
        split_13 = None
        view_14 = query_39.view(1, 19, -1, 64)
        query_39 = None
        query_40 = view_14.transpose(1, 2)
        view_14 = None
        getitem_81 = key_39[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        key_39 = None
        hidden_states_133 = getitem_81.expand(1, 1, 12, 19, 64)
        getitem_81 = None
        key_40 = hidden_states_133.reshape(1, 12, 19, 64)
        hidden_states_133 = None
        getitem_82 = value_39[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_39 = None
        hidden_states_134 = getitem_82.expand(1, 1, 12, 19, 64)
        getitem_82 = None
        value_40 = hidden_states_134.reshape(1, 12, 19, 64)
        hidden_states_134 = None
        attention_mask_14 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_41 = query_40.contiguous()
        query_40 = None
        key_41 = key_40.contiguous()
        key_40 = None
        value_41 = value_40.contiguous()
        value_40 = None
        attn_output_65 = torch._C._nn.scaled_dot_product_attention(
            query_41,
            key_41,
            value_41,
            attn_mask=attention_mask_14,
            dropout_p=0.0,
            scale=8.0,
            is_causal=False,
        )
        query_41 = key_41 = value_41 = attention_mask_14 = None
        transpose_27 = attn_output_65.transpose(1, 2)
        attn_output_65 = None
        attn_output_66 = transpose_27.contiguous()
        transpose_27 = None
        reshape_41 = attn_output_66.reshape(1, 19, -1)
        attn_output_66 = None
        attn_output_67 = reshape_41.contiguous()
        reshape_41 = None
        attn_output_68 = torch._C._nn.linear(
            attn_output_67,
            l_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_proj_parameters_bias_,
        )
        attn_output_67 = l_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_13_modules_attn_modules_c_proj_parameters_bias_ = (None)
        attn_output_69 = torch.nn.functional.dropout(attn_output_68, 0.1, False, False)
        attn_output_68 = None
        hidden_states_135 = attn_output_69 + hidden_states_131
        attn_output_69 = hidden_states_131 = None
        hidden_states_136 = torch.nn.functional.layer_norm(
            hidden_states_135,
            (768,),
            l_self_modules_transformer_modules_h_modules_13_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_13_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_13_modules_ln_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_13_modules_ln_2_parameters_bias_ = (None)
        hidden_states_137 = torch._C._nn.linear(
            hidden_states_136,
            l_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_fc_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_fc_parameters_bias_,
        )
        hidden_states_136 = l_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_fc_parameters_weight_ = l_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_fc_parameters_bias_ = (None)
        hidden_states_138 = torch._C._nn.gelu(hidden_states_137, approximate="tanh")
        hidden_states_137 = None
        hidden_states_139 = torch._C._nn.linear(
            hidden_states_138,
            l_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_proj_parameters_bias_,
        )
        hidden_states_138 = l_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_13_modules_mlp_modules_c_proj_parameters_bias_ = (None)
        hidden_states_140 = torch.nn.functional.dropout(
            hidden_states_139, 0.1, False, False
        )
        hidden_states_139 = None
        hidden_states_141 = hidden_states_135 + hidden_states_140
        hidden_states_135 = hidden_states_140 = None
        hidden_states_142 = torch.nn.functional.layer_norm(
            hidden_states_141,
            (768,),
            l_self_modules_transformer_modules_h_modules_14_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_14_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_14_modules_ln_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_14_modules_ln_1_parameters_bias_ = (None)
        linear_56 = torch._C._nn.linear(
            hidden_states_142,
            l_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_attn_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_attn_parameters_bias_,
        )
        hidden_states_142 = l_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_attn_parameters_weight_ = l_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_attn_parameters_bias_ = (None)
        unsqueeze_15 = linear_56.unsqueeze(1)
        linear_56 = None
        split_14 = unsqueeze_15.split((768, 64, 64), dim=3)
        unsqueeze_15 = None
        query_42 = split_14[0]
        key_42 = split_14[1]
        value_42 = split_14[2]
        split_14 = None
        view_15 = query_42.view(1, 19, -1, 64)
        query_42 = None
        query_43 = view_15.transpose(1, 2)
        view_15 = None
        getitem_87 = key_42[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        key_42 = None
        hidden_states_143 = getitem_87.expand(1, 1, 12, 19, 64)
        getitem_87 = None
        key_43 = hidden_states_143.reshape(1, 12, 19, 64)
        hidden_states_143 = None
        getitem_88 = value_42[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_42 = None
        hidden_states_144 = getitem_88.expand(1, 1, 12, 19, 64)
        getitem_88 = None
        value_43 = hidden_states_144.reshape(1, 12, 19, 64)
        hidden_states_144 = None
        attention_mask_15 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_44 = query_43.contiguous()
        query_43 = None
        key_44 = key_43.contiguous()
        key_43 = None
        value_44 = value_43.contiguous()
        value_43 = None
        attn_output_70 = torch._C._nn.scaled_dot_product_attention(
            query_44,
            key_44,
            value_44,
            attn_mask=attention_mask_15,
            dropout_p=0.0,
            scale=8.0,
            is_causal=False,
        )
        query_44 = key_44 = value_44 = attention_mask_15 = None
        transpose_29 = attn_output_70.transpose(1, 2)
        attn_output_70 = None
        attn_output_71 = transpose_29.contiguous()
        transpose_29 = None
        reshape_44 = attn_output_71.reshape(1, 19, -1)
        attn_output_71 = None
        attn_output_72 = reshape_44.contiguous()
        reshape_44 = None
        attn_output_73 = torch._C._nn.linear(
            attn_output_72,
            l_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_proj_parameters_bias_,
        )
        attn_output_72 = l_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_14_modules_attn_modules_c_proj_parameters_bias_ = (None)
        attn_output_74 = torch.nn.functional.dropout(attn_output_73, 0.1, False, False)
        attn_output_73 = None
        hidden_states_145 = attn_output_74 + hidden_states_141
        attn_output_74 = hidden_states_141 = None
        hidden_states_146 = torch.nn.functional.layer_norm(
            hidden_states_145,
            (768,),
            l_self_modules_transformer_modules_h_modules_14_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_14_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_14_modules_ln_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_14_modules_ln_2_parameters_bias_ = (None)
        hidden_states_147 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_fc_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_fc_parameters_bias_,
        )
        hidden_states_146 = l_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_fc_parameters_weight_ = l_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_fc_parameters_bias_ = (None)
        hidden_states_148 = torch._C._nn.gelu(hidden_states_147, approximate="tanh")
        hidden_states_147 = None
        hidden_states_149 = torch._C._nn.linear(
            hidden_states_148,
            l_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_proj_parameters_bias_,
        )
        hidden_states_148 = l_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_14_modules_mlp_modules_c_proj_parameters_bias_ = (None)
        hidden_states_150 = torch.nn.functional.dropout(
            hidden_states_149, 0.1, False, False
        )
        hidden_states_149 = None
        hidden_states_151 = hidden_states_145 + hidden_states_150
        hidden_states_145 = hidden_states_150 = None
        hidden_states_152 = torch.nn.functional.layer_norm(
            hidden_states_151,
            (768,),
            l_self_modules_transformer_modules_h_modules_15_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_15_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_15_modules_ln_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_15_modules_ln_1_parameters_bias_ = (None)
        linear_60 = torch._C._nn.linear(
            hidden_states_152,
            l_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_attn_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_attn_parameters_bias_,
        )
        hidden_states_152 = l_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_attn_parameters_weight_ = l_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_attn_parameters_bias_ = (None)
        unsqueeze_16 = linear_60.unsqueeze(1)
        linear_60 = None
        split_15 = unsqueeze_16.split((768, 64, 64), dim=3)
        unsqueeze_16 = None
        query_45 = split_15[0]
        key_45 = split_15[1]
        value_45 = split_15[2]
        split_15 = None
        view_16 = query_45.view(1, 19, -1, 64)
        query_45 = None
        query_46 = view_16.transpose(1, 2)
        view_16 = None
        getitem_93 = key_45[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        key_45 = None
        hidden_states_153 = getitem_93.expand(1, 1, 12, 19, 64)
        getitem_93 = None
        key_46 = hidden_states_153.reshape(1, 12, 19, 64)
        hidden_states_153 = None
        getitem_94 = value_45[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_45 = None
        hidden_states_154 = getitem_94.expand(1, 1, 12, 19, 64)
        getitem_94 = None
        value_46 = hidden_states_154.reshape(1, 12, 19, 64)
        hidden_states_154 = None
        attention_mask_16 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_47 = query_46.contiguous()
        query_46 = None
        key_47 = key_46.contiguous()
        key_46 = None
        value_47 = value_46.contiguous()
        value_46 = None
        attn_output_75 = torch._C._nn.scaled_dot_product_attention(
            query_47,
            key_47,
            value_47,
            attn_mask=attention_mask_16,
            dropout_p=0.0,
            scale=8.0,
            is_causal=False,
        )
        query_47 = key_47 = value_47 = attention_mask_16 = None
        transpose_31 = attn_output_75.transpose(1, 2)
        attn_output_75 = None
        attn_output_76 = transpose_31.contiguous()
        transpose_31 = None
        reshape_47 = attn_output_76.reshape(1, 19, -1)
        attn_output_76 = None
        attn_output_77 = reshape_47.contiguous()
        reshape_47 = None
        attn_output_78 = torch._C._nn.linear(
            attn_output_77,
            l_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_proj_parameters_bias_,
        )
        attn_output_77 = l_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_15_modules_attn_modules_c_proj_parameters_bias_ = (None)
        attn_output_79 = torch.nn.functional.dropout(attn_output_78, 0.1, False, False)
        attn_output_78 = None
        hidden_states_155 = attn_output_79 + hidden_states_151
        attn_output_79 = hidden_states_151 = None
        hidden_states_156 = torch.nn.functional.layer_norm(
            hidden_states_155,
            (768,),
            l_self_modules_transformer_modules_h_modules_15_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_15_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_15_modules_ln_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_15_modules_ln_2_parameters_bias_ = (None)
        hidden_states_157 = torch._C._nn.linear(
            hidden_states_156,
            l_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_fc_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_fc_parameters_bias_,
        )
        hidden_states_156 = l_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_fc_parameters_weight_ = l_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_fc_parameters_bias_ = (None)
        hidden_states_158 = torch._C._nn.gelu(hidden_states_157, approximate="tanh")
        hidden_states_157 = None
        hidden_states_159 = torch._C._nn.linear(
            hidden_states_158,
            l_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_proj_parameters_bias_,
        )
        hidden_states_158 = l_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_15_modules_mlp_modules_c_proj_parameters_bias_ = (None)
        hidden_states_160 = torch.nn.functional.dropout(
            hidden_states_159, 0.1, False, False
        )
        hidden_states_159 = None
        hidden_states_161 = hidden_states_155 + hidden_states_160
        hidden_states_155 = hidden_states_160 = None
        hidden_states_162 = torch.nn.functional.layer_norm(
            hidden_states_161,
            (768,),
            l_self_modules_transformer_modules_h_modules_16_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_16_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_16_modules_ln_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_16_modules_ln_1_parameters_bias_ = (None)
        linear_64 = torch._C._nn.linear(
            hidden_states_162,
            l_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_attn_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_attn_parameters_bias_,
        )
        hidden_states_162 = l_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_attn_parameters_weight_ = l_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_attn_parameters_bias_ = (None)
        unsqueeze_17 = linear_64.unsqueeze(1)
        linear_64 = None
        split_16 = unsqueeze_17.split((768, 64, 64), dim=3)
        unsqueeze_17 = None
        query_48 = split_16[0]
        key_48 = split_16[1]
        value_48 = split_16[2]
        split_16 = None
        view_17 = query_48.view(1, 19, -1, 64)
        query_48 = None
        query_49 = view_17.transpose(1, 2)
        view_17 = None
        getitem_99 = key_48[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        key_48 = None
        hidden_states_163 = getitem_99.expand(1, 1, 12, 19, 64)
        getitem_99 = None
        key_49 = hidden_states_163.reshape(1, 12, 19, 64)
        hidden_states_163 = None
        getitem_100 = value_48[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_48 = None
        hidden_states_164 = getitem_100.expand(1, 1, 12, 19, 64)
        getitem_100 = None
        value_49 = hidden_states_164.reshape(1, 12, 19, 64)
        hidden_states_164 = None
        attention_mask_17 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_50 = query_49.contiguous()
        query_49 = None
        key_50 = key_49.contiguous()
        key_49 = None
        value_50 = value_49.contiguous()
        value_49 = None
        attn_output_80 = torch._C._nn.scaled_dot_product_attention(
            query_50,
            key_50,
            value_50,
            attn_mask=attention_mask_17,
            dropout_p=0.0,
            scale=8.0,
            is_causal=False,
        )
        query_50 = key_50 = value_50 = attention_mask_17 = None
        transpose_33 = attn_output_80.transpose(1, 2)
        attn_output_80 = None
        attn_output_81 = transpose_33.contiguous()
        transpose_33 = None
        reshape_50 = attn_output_81.reshape(1, 19, -1)
        attn_output_81 = None
        attn_output_82 = reshape_50.contiguous()
        reshape_50 = None
        attn_output_83 = torch._C._nn.linear(
            attn_output_82,
            l_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_proj_parameters_bias_,
        )
        attn_output_82 = l_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_16_modules_attn_modules_c_proj_parameters_bias_ = (None)
        attn_output_84 = torch.nn.functional.dropout(attn_output_83, 0.1, False, False)
        attn_output_83 = None
        hidden_states_165 = attn_output_84 + hidden_states_161
        attn_output_84 = hidden_states_161 = None
        hidden_states_166 = torch.nn.functional.layer_norm(
            hidden_states_165,
            (768,),
            l_self_modules_transformer_modules_h_modules_16_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_16_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_16_modules_ln_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_16_modules_ln_2_parameters_bias_ = (None)
        hidden_states_167 = torch._C._nn.linear(
            hidden_states_166,
            l_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_fc_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_fc_parameters_bias_,
        )
        hidden_states_166 = l_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_fc_parameters_weight_ = l_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_fc_parameters_bias_ = (None)
        hidden_states_168 = torch._C._nn.gelu(hidden_states_167, approximate="tanh")
        hidden_states_167 = None
        hidden_states_169 = torch._C._nn.linear(
            hidden_states_168,
            l_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_proj_parameters_bias_,
        )
        hidden_states_168 = l_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_16_modules_mlp_modules_c_proj_parameters_bias_ = (None)
        hidden_states_170 = torch.nn.functional.dropout(
            hidden_states_169, 0.1, False, False
        )
        hidden_states_169 = None
        hidden_states_171 = hidden_states_165 + hidden_states_170
        hidden_states_165 = hidden_states_170 = None
        hidden_states_172 = torch.nn.functional.layer_norm(
            hidden_states_171,
            (768,),
            l_self_modules_transformer_modules_h_modules_17_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_17_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_17_modules_ln_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_17_modules_ln_1_parameters_bias_ = (None)
        linear_68 = torch._C._nn.linear(
            hidden_states_172,
            l_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_attn_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_attn_parameters_bias_,
        )
        hidden_states_172 = l_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_attn_parameters_weight_ = l_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_attn_parameters_bias_ = (None)
        unsqueeze_18 = linear_68.unsqueeze(1)
        linear_68 = None
        split_17 = unsqueeze_18.split((768, 64, 64), dim=3)
        unsqueeze_18 = None
        query_51 = split_17[0]
        key_51 = split_17[1]
        value_51 = split_17[2]
        split_17 = None
        view_18 = query_51.view(1, 19, -1, 64)
        query_51 = None
        query_52 = view_18.transpose(1, 2)
        view_18 = None
        getitem_105 = key_51[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        key_51 = None
        hidden_states_173 = getitem_105.expand(1, 1, 12, 19, 64)
        getitem_105 = None
        key_52 = hidden_states_173.reshape(1, 12, 19, 64)
        hidden_states_173 = None
        getitem_106 = value_51[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_51 = None
        hidden_states_174 = getitem_106.expand(1, 1, 12, 19, 64)
        getitem_106 = None
        value_52 = hidden_states_174.reshape(1, 12, 19, 64)
        hidden_states_174 = None
        attention_mask_18 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_53 = query_52.contiguous()
        query_52 = None
        key_53 = key_52.contiguous()
        key_52 = None
        value_53 = value_52.contiguous()
        value_52 = None
        attn_output_85 = torch._C._nn.scaled_dot_product_attention(
            query_53,
            key_53,
            value_53,
            attn_mask=attention_mask_18,
            dropout_p=0.0,
            scale=8.0,
            is_causal=False,
        )
        query_53 = key_53 = value_53 = attention_mask_18 = None
        transpose_35 = attn_output_85.transpose(1, 2)
        attn_output_85 = None
        attn_output_86 = transpose_35.contiguous()
        transpose_35 = None
        reshape_53 = attn_output_86.reshape(1, 19, -1)
        attn_output_86 = None
        attn_output_87 = reshape_53.contiguous()
        reshape_53 = None
        attn_output_88 = torch._C._nn.linear(
            attn_output_87,
            l_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_proj_parameters_bias_,
        )
        attn_output_87 = l_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_17_modules_attn_modules_c_proj_parameters_bias_ = (None)
        attn_output_89 = torch.nn.functional.dropout(attn_output_88, 0.1, False, False)
        attn_output_88 = None
        hidden_states_175 = attn_output_89 + hidden_states_171
        attn_output_89 = hidden_states_171 = None
        hidden_states_176 = torch.nn.functional.layer_norm(
            hidden_states_175,
            (768,),
            l_self_modules_transformer_modules_h_modules_17_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_17_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_17_modules_ln_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_17_modules_ln_2_parameters_bias_ = (None)
        hidden_states_177 = torch._C._nn.linear(
            hidden_states_176,
            l_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_fc_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_fc_parameters_bias_,
        )
        hidden_states_176 = l_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_fc_parameters_weight_ = l_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_fc_parameters_bias_ = (None)
        hidden_states_178 = torch._C._nn.gelu(hidden_states_177, approximate="tanh")
        hidden_states_177 = None
        hidden_states_179 = torch._C._nn.linear(
            hidden_states_178,
            l_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_proj_parameters_bias_,
        )
        hidden_states_178 = l_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_17_modules_mlp_modules_c_proj_parameters_bias_ = (None)
        hidden_states_180 = torch.nn.functional.dropout(
            hidden_states_179, 0.1, False, False
        )
        hidden_states_179 = None
        hidden_states_181 = hidden_states_175 + hidden_states_180
        hidden_states_175 = hidden_states_180 = None
        hidden_states_182 = torch.nn.functional.layer_norm(
            hidden_states_181,
            (768,),
            l_self_modules_transformer_modules_h_modules_18_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_18_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_18_modules_ln_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_18_modules_ln_1_parameters_bias_ = (None)
        linear_72 = torch._C._nn.linear(
            hidden_states_182,
            l_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_attn_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_attn_parameters_bias_,
        )
        hidden_states_182 = l_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_attn_parameters_weight_ = l_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_attn_parameters_bias_ = (None)
        unsqueeze_19 = linear_72.unsqueeze(1)
        linear_72 = None
        split_18 = unsqueeze_19.split((768, 64, 64), dim=3)
        unsqueeze_19 = None
        query_54 = split_18[0]
        key_54 = split_18[1]
        value_54 = split_18[2]
        split_18 = None
        view_19 = query_54.view(1, 19, -1, 64)
        query_54 = None
        query_55 = view_19.transpose(1, 2)
        view_19 = None
        getitem_111 = key_54[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        key_54 = None
        hidden_states_183 = getitem_111.expand(1, 1, 12, 19, 64)
        getitem_111 = None
        key_55 = hidden_states_183.reshape(1, 12, 19, 64)
        hidden_states_183 = None
        getitem_112 = value_54[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_54 = None
        hidden_states_184 = getitem_112.expand(1, 1, 12, 19, 64)
        getitem_112 = None
        value_55 = hidden_states_184.reshape(1, 12, 19, 64)
        hidden_states_184 = None
        attention_mask_19 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_56 = query_55.contiguous()
        query_55 = None
        key_56 = key_55.contiguous()
        key_55 = None
        value_56 = value_55.contiguous()
        value_55 = None
        attn_output_90 = torch._C._nn.scaled_dot_product_attention(
            query_56,
            key_56,
            value_56,
            attn_mask=attention_mask_19,
            dropout_p=0.0,
            scale=8.0,
            is_causal=False,
        )
        query_56 = key_56 = value_56 = attention_mask_19 = None
        transpose_37 = attn_output_90.transpose(1, 2)
        attn_output_90 = None
        attn_output_91 = transpose_37.contiguous()
        transpose_37 = None
        reshape_56 = attn_output_91.reshape(1, 19, -1)
        attn_output_91 = None
        attn_output_92 = reshape_56.contiguous()
        reshape_56 = None
        attn_output_93 = torch._C._nn.linear(
            attn_output_92,
            l_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_proj_parameters_bias_,
        )
        attn_output_92 = l_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_18_modules_attn_modules_c_proj_parameters_bias_ = (None)
        attn_output_94 = torch.nn.functional.dropout(attn_output_93, 0.1, False, False)
        attn_output_93 = None
        hidden_states_185 = attn_output_94 + hidden_states_181
        attn_output_94 = hidden_states_181 = None
        hidden_states_186 = torch.nn.functional.layer_norm(
            hidden_states_185,
            (768,),
            l_self_modules_transformer_modules_h_modules_18_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_18_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_18_modules_ln_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_18_modules_ln_2_parameters_bias_ = (None)
        hidden_states_187 = torch._C._nn.linear(
            hidden_states_186,
            l_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_fc_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_fc_parameters_bias_,
        )
        hidden_states_186 = l_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_fc_parameters_weight_ = l_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_fc_parameters_bias_ = (None)
        hidden_states_188 = torch._C._nn.gelu(hidden_states_187, approximate="tanh")
        hidden_states_187 = None
        hidden_states_189 = torch._C._nn.linear(
            hidden_states_188,
            l_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_proj_parameters_bias_,
        )
        hidden_states_188 = l_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_18_modules_mlp_modules_c_proj_parameters_bias_ = (None)
        hidden_states_190 = torch.nn.functional.dropout(
            hidden_states_189, 0.1, False, False
        )
        hidden_states_189 = None
        hidden_states_191 = hidden_states_185 + hidden_states_190
        hidden_states_185 = hidden_states_190 = None
        hidden_states_192 = torch.nn.functional.layer_norm(
            hidden_states_191,
            (768,),
            l_self_modules_transformer_modules_h_modules_19_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_19_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_19_modules_ln_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_19_modules_ln_1_parameters_bias_ = (None)
        linear_76 = torch._C._nn.linear(
            hidden_states_192,
            l_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_attn_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_attn_parameters_bias_,
        )
        hidden_states_192 = l_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_attn_parameters_weight_ = l_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_attn_parameters_bias_ = (None)
        unsqueeze_20 = linear_76.unsqueeze(1)
        linear_76 = None
        split_19 = unsqueeze_20.split((768, 64, 64), dim=3)
        unsqueeze_20 = None
        query_57 = split_19[0]
        key_57 = split_19[1]
        value_57 = split_19[2]
        split_19 = None
        view_20 = query_57.view(1, 19, -1, 64)
        query_57 = None
        query_58 = view_20.transpose(1, 2)
        view_20 = None
        getitem_117 = key_57[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        key_57 = None
        hidden_states_193 = getitem_117.expand(1, 1, 12, 19, 64)
        getitem_117 = None
        key_58 = hidden_states_193.reshape(1, 12, 19, 64)
        hidden_states_193 = None
        getitem_118 = value_57[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_57 = None
        hidden_states_194 = getitem_118.expand(1, 1, 12, 19, 64)
        getitem_118 = None
        value_58 = hidden_states_194.reshape(1, 12, 19, 64)
        hidden_states_194 = None
        attention_mask_20 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        causal_mask = None
        query_59 = query_58.contiguous()
        query_58 = None
        key_59 = key_58.contiguous()
        key_58 = None
        value_59 = value_58.contiguous()
        value_58 = None
        attn_output_95 = torch._C._nn.scaled_dot_product_attention(
            query_59,
            key_59,
            value_59,
            attn_mask=attention_mask_20,
            dropout_p=0.0,
            scale=8.0,
            is_causal=False,
        )
        query_59 = key_59 = value_59 = attention_mask_20 = None
        transpose_39 = attn_output_95.transpose(1, 2)
        attn_output_95 = None
        attn_output_96 = transpose_39.contiguous()
        transpose_39 = None
        reshape_59 = attn_output_96.reshape(1, 19, -1)
        attn_output_96 = None
        attn_output_97 = reshape_59.contiguous()
        reshape_59 = None
        attn_output_98 = torch._C._nn.linear(
            attn_output_97,
            l_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_proj_parameters_bias_,
        )
        attn_output_97 = l_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_19_modules_attn_modules_c_proj_parameters_bias_ = (None)
        attn_output_99 = torch.nn.functional.dropout(attn_output_98, 0.1, False, False)
        attn_output_98 = None
        hidden_states_195 = attn_output_99 + hidden_states_191
        attn_output_99 = hidden_states_191 = None
        hidden_states_196 = torch.nn.functional.layer_norm(
            hidden_states_195,
            (768,),
            l_self_modules_transformer_modules_h_modules_19_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_19_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_19_modules_ln_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_19_modules_ln_2_parameters_bias_ = (None)
        hidden_states_197 = torch._C._nn.linear(
            hidden_states_196,
            l_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_fc_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_fc_parameters_bias_,
        )
        hidden_states_196 = l_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_fc_parameters_weight_ = l_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_fc_parameters_bias_ = (None)
        hidden_states_198 = torch._C._nn.gelu(hidden_states_197, approximate="tanh")
        hidden_states_197 = None
        hidden_states_199 = torch._C._nn.linear(
            hidden_states_198,
            l_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_proj_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_proj_parameters_bias_,
        )
        hidden_states_198 = l_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_19_modules_mlp_modules_c_proj_parameters_bias_ = (None)
        hidden_states_200 = torch.nn.functional.dropout(
            hidden_states_199, 0.1, False, False
        )
        hidden_states_199 = None
        hidden_states_201 = hidden_states_195 + hidden_states_200
        hidden_states_195 = hidden_states_200 = None
        hidden_states_202 = torch.nn.functional.layer_norm(
            hidden_states_201,
            (768,),
            l_self_modules_transformer_modules_ln_f_parameters_weight_,
            l_self_modules_transformer_modules_ln_f_parameters_bias_,
            1e-05,
        )
        hidden_states_201 = (
            l_self_modules_transformer_modules_ln_f_parameters_weight_
        ) = l_self_modules_transformer_modules_ln_f_parameters_bias_ = None
        hidden_states_203 = hidden_states_202.view((1, 19, 768))
        hidden_states_202 = None
        lm_logits = torch._C._nn.linear(
            hidden_states_203,
            l_self_modules_transformer_modules_wte_parameters_weight_,
            None,
        )
        hidden_states_203 = (
            l_self_modules_transformer_modules_wte_parameters_weight_
        ) = None
        return (lm_logits,)
