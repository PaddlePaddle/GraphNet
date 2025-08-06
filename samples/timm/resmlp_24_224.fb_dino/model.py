import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_parameters_ls1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm1_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm1_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_linear_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_linear_tokens_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_parameters_ls2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm2_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm2_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_parameters_ls1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm1_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm1_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_linear_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_linear_tokens_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_parameters_ls2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm2_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm2_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_parameters_ls1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm1_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm1_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_linear_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_linear_tokens_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_parameters_ls2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm2_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm2_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_parameters_ls1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm1_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm1_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_linear_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_linear_tokens_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_parameters_ls2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm2_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm2_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_parameters_ls1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm1_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm1_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_linear_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_linear_tokens_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_parameters_ls2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm2_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm2_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_parameters_ls1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm1_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm1_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_linear_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_linear_tokens_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_parameters_ls2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm2_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm2_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_parameters_ls1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm1_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm1_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_linear_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_linear_tokens_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_parameters_ls2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm2_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm2_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_parameters_ls1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm1_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm1_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_linear_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_linear_tokens_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_parameters_ls2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm2_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm2_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_parameters_ls1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm1_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm1_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_linear_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_linear_tokens_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_parameters_ls2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm2_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm2_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_parameters_ls1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm1_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm1_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_linear_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_linear_tokens_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_parameters_ls2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm2_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm2_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_parameters_ls1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm1_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm1_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_linear_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_linear_tokens_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_parameters_ls2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm2_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm2_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_parameters_ls1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm1_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm1_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_linear_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_linear_tokens_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_parameters_ls2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm2_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm2_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_parameters_ls1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm1_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm1_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_linear_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_linear_tokens_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_parameters_ls2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm2_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm2_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_parameters_ls1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm1_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm1_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_linear_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_linear_tokens_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_parameters_ls2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm2_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm2_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_parameters_ls1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm1_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm1_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_linear_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_linear_tokens_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_parameters_ls2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm2_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm2_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_parameters_ls1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm1_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm1_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_linear_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_linear_tokens_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_parameters_ls2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm2_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm2_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_parameters_ls1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm1_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm1_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_linear_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_linear_tokens_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_parameters_ls2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm2_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm2_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_parameters_ls1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm1_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm1_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_linear_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_linear_tokens_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_parameters_ls2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm2_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm2_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_parameters_ls1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm1_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm1_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_linear_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_linear_tokens_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_parameters_ls2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm2_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm2_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_parameters_ls1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm1_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm1_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_linear_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_linear_tokens_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_parameters_ls2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm2_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm2_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_parameters_ls1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm1_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm1_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_linear_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_linear_tokens_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_parameters_ls2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm2_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm2_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_parameters_ls1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm1_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm1_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_linear_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_linear_tokens_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_parameters_ls2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm2_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm2_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_parameters_ls1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm1_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm1_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_linear_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_linear_tokens_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_parameters_ls2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm2_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm2_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_parameters_ls1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm1_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm1_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_linear_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_linear_tokens_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_parameters_ls2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm2_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm2_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_norm_parameters_beta_: torch.nn.parameter.Parameter,
        L_self_modules_norm_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_head_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_x_ = L_x_
        l_self_modules_stem_modules_proj_parameters_weight_ = (
            L_self_modules_stem_modules_proj_parameters_weight_
        )
        l_self_modules_stem_modules_proj_parameters_bias_ = (
            L_self_modules_stem_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_0_parameters_ls1_ = (
            L_self_modules_blocks_modules_0_parameters_ls1_
        )
        l_self_modules_blocks_modules_0_modules_norm1_parameters_beta_ = (
            L_self_modules_blocks_modules_0_modules_norm1_parameters_beta_
        )
        l_self_modules_blocks_modules_0_modules_norm1_parameters_alpha_ = (
            L_self_modules_blocks_modules_0_modules_norm1_parameters_alpha_
        )
        l_self_modules_blocks_modules_0_modules_linear_tokens_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_linear_tokens_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_linear_tokens_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_linear_tokens_parameters_bias_
        )
        l_self_modules_blocks_modules_0_parameters_ls2_ = (
            L_self_modules_blocks_modules_0_parameters_ls2_
        )
        l_self_modules_blocks_modules_0_modules_norm2_parameters_beta_ = (
            L_self_modules_blocks_modules_0_modules_norm2_parameters_beta_
        )
        l_self_modules_blocks_modules_0_modules_norm2_parameters_alpha_ = (
            L_self_modules_blocks_modules_0_modules_norm2_parameters_alpha_
        )
        l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_1_parameters_ls1_ = (
            L_self_modules_blocks_modules_1_parameters_ls1_
        )
        l_self_modules_blocks_modules_1_modules_norm1_parameters_beta_ = (
            L_self_modules_blocks_modules_1_modules_norm1_parameters_beta_
        )
        l_self_modules_blocks_modules_1_modules_norm1_parameters_alpha_ = (
            L_self_modules_blocks_modules_1_modules_norm1_parameters_alpha_
        )
        l_self_modules_blocks_modules_1_modules_linear_tokens_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_linear_tokens_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_linear_tokens_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_linear_tokens_parameters_bias_
        )
        l_self_modules_blocks_modules_1_parameters_ls2_ = (
            L_self_modules_blocks_modules_1_parameters_ls2_
        )
        l_self_modules_blocks_modules_1_modules_norm2_parameters_beta_ = (
            L_self_modules_blocks_modules_1_modules_norm2_parameters_beta_
        )
        l_self_modules_blocks_modules_1_modules_norm2_parameters_alpha_ = (
            L_self_modules_blocks_modules_1_modules_norm2_parameters_alpha_
        )
        l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_2_parameters_ls1_ = (
            L_self_modules_blocks_modules_2_parameters_ls1_
        )
        l_self_modules_blocks_modules_2_modules_norm1_parameters_beta_ = (
            L_self_modules_blocks_modules_2_modules_norm1_parameters_beta_
        )
        l_self_modules_blocks_modules_2_modules_norm1_parameters_alpha_ = (
            L_self_modules_blocks_modules_2_modules_norm1_parameters_alpha_
        )
        l_self_modules_blocks_modules_2_modules_linear_tokens_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_linear_tokens_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_linear_tokens_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_linear_tokens_parameters_bias_
        )
        l_self_modules_blocks_modules_2_parameters_ls2_ = (
            L_self_modules_blocks_modules_2_parameters_ls2_
        )
        l_self_modules_blocks_modules_2_modules_norm2_parameters_beta_ = (
            L_self_modules_blocks_modules_2_modules_norm2_parameters_beta_
        )
        l_self_modules_blocks_modules_2_modules_norm2_parameters_alpha_ = (
            L_self_modules_blocks_modules_2_modules_norm2_parameters_alpha_
        )
        l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_3_parameters_ls1_ = (
            L_self_modules_blocks_modules_3_parameters_ls1_
        )
        l_self_modules_blocks_modules_3_modules_norm1_parameters_beta_ = (
            L_self_modules_blocks_modules_3_modules_norm1_parameters_beta_
        )
        l_self_modules_blocks_modules_3_modules_norm1_parameters_alpha_ = (
            L_self_modules_blocks_modules_3_modules_norm1_parameters_alpha_
        )
        l_self_modules_blocks_modules_3_modules_linear_tokens_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_linear_tokens_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_linear_tokens_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_linear_tokens_parameters_bias_
        )
        l_self_modules_blocks_modules_3_parameters_ls2_ = (
            L_self_modules_blocks_modules_3_parameters_ls2_
        )
        l_self_modules_blocks_modules_3_modules_norm2_parameters_beta_ = (
            L_self_modules_blocks_modules_3_modules_norm2_parameters_beta_
        )
        l_self_modules_blocks_modules_3_modules_norm2_parameters_alpha_ = (
            L_self_modules_blocks_modules_3_modules_norm2_parameters_alpha_
        )
        l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_4_parameters_ls1_ = (
            L_self_modules_blocks_modules_4_parameters_ls1_
        )
        l_self_modules_blocks_modules_4_modules_norm1_parameters_beta_ = (
            L_self_modules_blocks_modules_4_modules_norm1_parameters_beta_
        )
        l_self_modules_blocks_modules_4_modules_norm1_parameters_alpha_ = (
            L_self_modules_blocks_modules_4_modules_norm1_parameters_alpha_
        )
        l_self_modules_blocks_modules_4_modules_linear_tokens_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_linear_tokens_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_linear_tokens_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_linear_tokens_parameters_bias_
        )
        l_self_modules_blocks_modules_4_parameters_ls2_ = (
            L_self_modules_blocks_modules_4_parameters_ls2_
        )
        l_self_modules_blocks_modules_4_modules_norm2_parameters_beta_ = (
            L_self_modules_blocks_modules_4_modules_norm2_parameters_beta_
        )
        l_self_modules_blocks_modules_4_modules_norm2_parameters_alpha_ = (
            L_self_modules_blocks_modules_4_modules_norm2_parameters_alpha_
        )
        l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_5_parameters_ls1_ = (
            L_self_modules_blocks_modules_5_parameters_ls1_
        )
        l_self_modules_blocks_modules_5_modules_norm1_parameters_beta_ = (
            L_self_modules_blocks_modules_5_modules_norm1_parameters_beta_
        )
        l_self_modules_blocks_modules_5_modules_norm1_parameters_alpha_ = (
            L_self_modules_blocks_modules_5_modules_norm1_parameters_alpha_
        )
        l_self_modules_blocks_modules_5_modules_linear_tokens_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_linear_tokens_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_linear_tokens_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_linear_tokens_parameters_bias_
        )
        l_self_modules_blocks_modules_5_parameters_ls2_ = (
            L_self_modules_blocks_modules_5_parameters_ls2_
        )
        l_self_modules_blocks_modules_5_modules_norm2_parameters_beta_ = (
            L_self_modules_blocks_modules_5_modules_norm2_parameters_beta_
        )
        l_self_modules_blocks_modules_5_modules_norm2_parameters_alpha_ = (
            L_self_modules_blocks_modules_5_modules_norm2_parameters_alpha_
        )
        l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_6_parameters_ls1_ = (
            L_self_modules_blocks_modules_6_parameters_ls1_
        )
        l_self_modules_blocks_modules_6_modules_norm1_parameters_beta_ = (
            L_self_modules_blocks_modules_6_modules_norm1_parameters_beta_
        )
        l_self_modules_blocks_modules_6_modules_norm1_parameters_alpha_ = (
            L_self_modules_blocks_modules_6_modules_norm1_parameters_alpha_
        )
        l_self_modules_blocks_modules_6_modules_linear_tokens_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_linear_tokens_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_linear_tokens_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_linear_tokens_parameters_bias_
        )
        l_self_modules_blocks_modules_6_parameters_ls2_ = (
            L_self_modules_blocks_modules_6_parameters_ls2_
        )
        l_self_modules_blocks_modules_6_modules_norm2_parameters_beta_ = (
            L_self_modules_blocks_modules_6_modules_norm2_parameters_beta_
        )
        l_self_modules_blocks_modules_6_modules_norm2_parameters_alpha_ = (
            L_self_modules_blocks_modules_6_modules_norm2_parameters_alpha_
        )
        l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_7_parameters_ls1_ = (
            L_self_modules_blocks_modules_7_parameters_ls1_
        )
        l_self_modules_blocks_modules_7_modules_norm1_parameters_beta_ = (
            L_self_modules_blocks_modules_7_modules_norm1_parameters_beta_
        )
        l_self_modules_blocks_modules_7_modules_norm1_parameters_alpha_ = (
            L_self_modules_blocks_modules_7_modules_norm1_parameters_alpha_
        )
        l_self_modules_blocks_modules_7_modules_linear_tokens_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_linear_tokens_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_linear_tokens_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_linear_tokens_parameters_bias_
        )
        l_self_modules_blocks_modules_7_parameters_ls2_ = (
            L_self_modules_blocks_modules_7_parameters_ls2_
        )
        l_self_modules_blocks_modules_7_modules_norm2_parameters_beta_ = (
            L_self_modules_blocks_modules_7_modules_norm2_parameters_beta_
        )
        l_self_modules_blocks_modules_7_modules_norm2_parameters_alpha_ = (
            L_self_modules_blocks_modules_7_modules_norm2_parameters_alpha_
        )
        l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_8_parameters_ls1_ = (
            L_self_modules_blocks_modules_8_parameters_ls1_
        )
        l_self_modules_blocks_modules_8_modules_norm1_parameters_beta_ = (
            L_self_modules_blocks_modules_8_modules_norm1_parameters_beta_
        )
        l_self_modules_blocks_modules_8_modules_norm1_parameters_alpha_ = (
            L_self_modules_blocks_modules_8_modules_norm1_parameters_alpha_
        )
        l_self_modules_blocks_modules_8_modules_linear_tokens_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_linear_tokens_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_linear_tokens_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_linear_tokens_parameters_bias_
        )
        l_self_modules_blocks_modules_8_parameters_ls2_ = (
            L_self_modules_blocks_modules_8_parameters_ls2_
        )
        l_self_modules_blocks_modules_8_modules_norm2_parameters_beta_ = (
            L_self_modules_blocks_modules_8_modules_norm2_parameters_beta_
        )
        l_self_modules_blocks_modules_8_modules_norm2_parameters_alpha_ = (
            L_self_modules_blocks_modules_8_modules_norm2_parameters_alpha_
        )
        l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_9_parameters_ls1_ = (
            L_self_modules_blocks_modules_9_parameters_ls1_
        )
        l_self_modules_blocks_modules_9_modules_norm1_parameters_beta_ = (
            L_self_modules_blocks_modules_9_modules_norm1_parameters_beta_
        )
        l_self_modules_blocks_modules_9_modules_norm1_parameters_alpha_ = (
            L_self_modules_blocks_modules_9_modules_norm1_parameters_alpha_
        )
        l_self_modules_blocks_modules_9_modules_linear_tokens_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_linear_tokens_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_linear_tokens_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_linear_tokens_parameters_bias_
        )
        l_self_modules_blocks_modules_9_parameters_ls2_ = (
            L_self_modules_blocks_modules_9_parameters_ls2_
        )
        l_self_modules_blocks_modules_9_modules_norm2_parameters_beta_ = (
            L_self_modules_blocks_modules_9_modules_norm2_parameters_beta_
        )
        l_self_modules_blocks_modules_9_modules_norm2_parameters_alpha_ = (
            L_self_modules_blocks_modules_9_modules_norm2_parameters_alpha_
        )
        l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_10_parameters_ls1_ = (
            L_self_modules_blocks_modules_10_parameters_ls1_
        )
        l_self_modules_blocks_modules_10_modules_norm1_parameters_beta_ = (
            L_self_modules_blocks_modules_10_modules_norm1_parameters_beta_
        )
        l_self_modules_blocks_modules_10_modules_norm1_parameters_alpha_ = (
            L_self_modules_blocks_modules_10_modules_norm1_parameters_alpha_
        )
        l_self_modules_blocks_modules_10_modules_linear_tokens_parameters_weight_ = (
            L_self_modules_blocks_modules_10_modules_linear_tokens_parameters_weight_
        )
        l_self_modules_blocks_modules_10_modules_linear_tokens_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_linear_tokens_parameters_bias_
        )
        l_self_modules_blocks_modules_10_parameters_ls2_ = (
            L_self_modules_blocks_modules_10_parameters_ls2_
        )
        l_self_modules_blocks_modules_10_modules_norm2_parameters_beta_ = (
            L_self_modules_blocks_modules_10_modules_norm2_parameters_beta_
        )
        l_self_modules_blocks_modules_10_modules_norm2_parameters_alpha_ = (
            L_self_modules_blocks_modules_10_modules_norm2_parameters_alpha_
        )
        l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_11_parameters_ls1_ = (
            L_self_modules_blocks_modules_11_parameters_ls1_
        )
        l_self_modules_blocks_modules_11_modules_norm1_parameters_beta_ = (
            L_self_modules_blocks_modules_11_modules_norm1_parameters_beta_
        )
        l_self_modules_blocks_modules_11_modules_norm1_parameters_alpha_ = (
            L_self_modules_blocks_modules_11_modules_norm1_parameters_alpha_
        )
        l_self_modules_blocks_modules_11_modules_linear_tokens_parameters_weight_ = (
            L_self_modules_blocks_modules_11_modules_linear_tokens_parameters_weight_
        )
        l_self_modules_blocks_modules_11_modules_linear_tokens_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_linear_tokens_parameters_bias_
        )
        l_self_modules_blocks_modules_11_parameters_ls2_ = (
            L_self_modules_blocks_modules_11_parameters_ls2_
        )
        l_self_modules_blocks_modules_11_modules_norm2_parameters_beta_ = (
            L_self_modules_blocks_modules_11_modules_norm2_parameters_beta_
        )
        l_self_modules_blocks_modules_11_modules_norm2_parameters_alpha_ = (
            L_self_modules_blocks_modules_11_modules_norm2_parameters_alpha_
        )
        l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_12_parameters_ls1_ = (
            L_self_modules_blocks_modules_12_parameters_ls1_
        )
        l_self_modules_blocks_modules_12_modules_norm1_parameters_beta_ = (
            L_self_modules_blocks_modules_12_modules_norm1_parameters_beta_
        )
        l_self_modules_blocks_modules_12_modules_norm1_parameters_alpha_ = (
            L_self_modules_blocks_modules_12_modules_norm1_parameters_alpha_
        )
        l_self_modules_blocks_modules_12_modules_linear_tokens_parameters_weight_ = (
            L_self_modules_blocks_modules_12_modules_linear_tokens_parameters_weight_
        )
        l_self_modules_blocks_modules_12_modules_linear_tokens_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_linear_tokens_parameters_bias_
        )
        l_self_modules_blocks_modules_12_parameters_ls2_ = (
            L_self_modules_blocks_modules_12_parameters_ls2_
        )
        l_self_modules_blocks_modules_12_modules_norm2_parameters_beta_ = (
            L_self_modules_blocks_modules_12_modules_norm2_parameters_beta_
        )
        l_self_modules_blocks_modules_12_modules_norm2_parameters_alpha_ = (
            L_self_modules_blocks_modules_12_modules_norm2_parameters_alpha_
        )
        l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_13_parameters_ls1_ = (
            L_self_modules_blocks_modules_13_parameters_ls1_
        )
        l_self_modules_blocks_modules_13_modules_norm1_parameters_beta_ = (
            L_self_modules_blocks_modules_13_modules_norm1_parameters_beta_
        )
        l_self_modules_blocks_modules_13_modules_norm1_parameters_alpha_ = (
            L_self_modules_blocks_modules_13_modules_norm1_parameters_alpha_
        )
        l_self_modules_blocks_modules_13_modules_linear_tokens_parameters_weight_ = (
            L_self_modules_blocks_modules_13_modules_linear_tokens_parameters_weight_
        )
        l_self_modules_blocks_modules_13_modules_linear_tokens_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_linear_tokens_parameters_bias_
        )
        l_self_modules_blocks_modules_13_parameters_ls2_ = (
            L_self_modules_blocks_modules_13_parameters_ls2_
        )
        l_self_modules_blocks_modules_13_modules_norm2_parameters_beta_ = (
            L_self_modules_blocks_modules_13_modules_norm2_parameters_beta_
        )
        l_self_modules_blocks_modules_13_modules_norm2_parameters_alpha_ = (
            L_self_modules_blocks_modules_13_modules_norm2_parameters_alpha_
        )
        l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_14_parameters_ls1_ = (
            L_self_modules_blocks_modules_14_parameters_ls1_
        )
        l_self_modules_blocks_modules_14_modules_norm1_parameters_beta_ = (
            L_self_modules_blocks_modules_14_modules_norm1_parameters_beta_
        )
        l_self_modules_blocks_modules_14_modules_norm1_parameters_alpha_ = (
            L_self_modules_blocks_modules_14_modules_norm1_parameters_alpha_
        )
        l_self_modules_blocks_modules_14_modules_linear_tokens_parameters_weight_ = (
            L_self_modules_blocks_modules_14_modules_linear_tokens_parameters_weight_
        )
        l_self_modules_blocks_modules_14_modules_linear_tokens_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_linear_tokens_parameters_bias_
        )
        l_self_modules_blocks_modules_14_parameters_ls2_ = (
            L_self_modules_blocks_modules_14_parameters_ls2_
        )
        l_self_modules_blocks_modules_14_modules_norm2_parameters_beta_ = (
            L_self_modules_blocks_modules_14_modules_norm2_parameters_beta_
        )
        l_self_modules_blocks_modules_14_modules_norm2_parameters_alpha_ = (
            L_self_modules_blocks_modules_14_modules_norm2_parameters_alpha_
        )
        l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_15_parameters_ls1_ = (
            L_self_modules_blocks_modules_15_parameters_ls1_
        )
        l_self_modules_blocks_modules_15_modules_norm1_parameters_beta_ = (
            L_self_modules_blocks_modules_15_modules_norm1_parameters_beta_
        )
        l_self_modules_blocks_modules_15_modules_norm1_parameters_alpha_ = (
            L_self_modules_blocks_modules_15_modules_norm1_parameters_alpha_
        )
        l_self_modules_blocks_modules_15_modules_linear_tokens_parameters_weight_ = (
            L_self_modules_blocks_modules_15_modules_linear_tokens_parameters_weight_
        )
        l_self_modules_blocks_modules_15_modules_linear_tokens_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_linear_tokens_parameters_bias_
        )
        l_self_modules_blocks_modules_15_parameters_ls2_ = (
            L_self_modules_blocks_modules_15_parameters_ls2_
        )
        l_self_modules_blocks_modules_15_modules_norm2_parameters_beta_ = (
            L_self_modules_blocks_modules_15_modules_norm2_parameters_beta_
        )
        l_self_modules_blocks_modules_15_modules_norm2_parameters_alpha_ = (
            L_self_modules_blocks_modules_15_modules_norm2_parameters_alpha_
        )
        l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_16_parameters_ls1_ = (
            L_self_modules_blocks_modules_16_parameters_ls1_
        )
        l_self_modules_blocks_modules_16_modules_norm1_parameters_beta_ = (
            L_self_modules_blocks_modules_16_modules_norm1_parameters_beta_
        )
        l_self_modules_blocks_modules_16_modules_norm1_parameters_alpha_ = (
            L_self_modules_blocks_modules_16_modules_norm1_parameters_alpha_
        )
        l_self_modules_blocks_modules_16_modules_linear_tokens_parameters_weight_ = (
            L_self_modules_blocks_modules_16_modules_linear_tokens_parameters_weight_
        )
        l_self_modules_blocks_modules_16_modules_linear_tokens_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_linear_tokens_parameters_bias_
        )
        l_self_modules_blocks_modules_16_parameters_ls2_ = (
            L_self_modules_blocks_modules_16_parameters_ls2_
        )
        l_self_modules_blocks_modules_16_modules_norm2_parameters_beta_ = (
            L_self_modules_blocks_modules_16_modules_norm2_parameters_beta_
        )
        l_self_modules_blocks_modules_16_modules_norm2_parameters_alpha_ = (
            L_self_modules_blocks_modules_16_modules_norm2_parameters_alpha_
        )
        l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_17_parameters_ls1_ = (
            L_self_modules_blocks_modules_17_parameters_ls1_
        )
        l_self_modules_blocks_modules_17_modules_norm1_parameters_beta_ = (
            L_self_modules_blocks_modules_17_modules_norm1_parameters_beta_
        )
        l_self_modules_blocks_modules_17_modules_norm1_parameters_alpha_ = (
            L_self_modules_blocks_modules_17_modules_norm1_parameters_alpha_
        )
        l_self_modules_blocks_modules_17_modules_linear_tokens_parameters_weight_ = (
            L_self_modules_blocks_modules_17_modules_linear_tokens_parameters_weight_
        )
        l_self_modules_blocks_modules_17_modules_linear_tokens_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_linear_tokens_parameters_bias_
        )
        l_self_modules_blocks_modules_17_parameters_ls2_ = (
            L_self_modules_blocks_modules_17_parameters_ls2_
        )
        l_self_modules_blocks_modules_17_modules_norm2_parameters_beta_ = (
            L_self_modules_blocks_modules_17_modules_norm2_parameters_beta_
        )
        l_self_modules_blocks_modules_17_modules_norm2_parameters_alpha_ = (
            L_self_modules_blocks_modules_17_modules_norm2_parameters_alpha_
        )
        l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_18_parameters_ls1_ = (
            L_self_modules_blocks_modules_18_parameters_ls1_
        )
        l_self_modules_blocks_modules_18_modules_norm1_parameters_beta_ = (
            L_self_modules_blocks_modules_18_modules_norm1_parameters_beta_
        )
        l_self_modules_blocks_modules_18_modules_norm1_parameters_alpha_ = (
            L_self_modules_blocks_modules_18_modules_norm1_parameters_alpha_
        )
        l_self_modules_blocks_modules_18_modules_linear_tokens_parameters_weight_ = (
            L_self_modules_blocks_modules_18_modules_linear_tokens_parameters_weight_
        )
        l_self_modules_blocks_modules_18_modules_linear_tokens_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_linear_tokens_parameters_bias_
        )
        l_self_modules_blocks_modules_18_parameters_ls2_ = (
            L_self_modules_blocks_modules_18_parameters_ls2_
        )
        l_self_modules_blocks_modules_18_modules_norm2_parameters_beta_ = (
            L_self_modules_blocks_modules_18_modules_norm2_parameters_beta_
        )
        l_self_modules_blocks_modules_18_modules_norm2_parameters_alpha_ = (
            L_self_modules_blocks_modules_18_modules_norm2_parameters_alpha_
        )
        l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_19_parameters_ls1_ = (
            L_self_modules_blocks_modules_19_parameters_ls1_
        )
        l_self_modules_blocks_modules_19_modules_norm1_parameters_beta_ = (
            L_self_modules_blocks_modules_19_modules_norm1_parameters_beta_
        )
        l_self_modules_blocks_modules_19_modules_norm1_parameters_alpha_ = (
            L_self_modules_blocks_modules_19_modules_norm1_parameters_alpha_
        )
        l_self_modules_blocks_modules_19_modules_linear_tokens_parameters_weight_ = (
            L_self_modules_blocks_modules_19_modules_linear_tokens_parameters_weight_
        )
        l_self_modules_blocks_modules_19_modules_linear_tokens_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_linear_tokens_parameters_bias_
        )
        l_self_modules_blocks_modules_19_parameters_ls2_ = (
            L_self_modules_blocks_modules_19_parameters_ls2_
        )
        l_self_modules_blocks_modules_19_modules_norm2_parameters_beta_ = (
            L_self_modules_blocks_modules_19_modules_norm2_parameters_beta_
        )
        l_self_modules_blocks_modules_19_modules_norm2_parameters_alpha_ = (
            L_self_modules_blocks_modules_19_modules_norm2_parameters_alpha_
        )
        l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_20_parameters_ls1_ = (
            L_self_modules_blocks_modules_20_parameters_ls1_
        )
        l_self_modules_blocks_modules_20_modules_norm1_parameters_beta_ = (
            L_self_modules_blocks_modules_20_modules_norm1_parameters_beta_
        )
        l_self_modules_blocks_modules_20_modules_norm1_parameters_alpha_ = (
            L_self_modules_blocks_modules_20_modules_norm1_parameters_alpha_
        )
        l_self_modules_blocks_modules_20_modules_linear_tokens_parameters_weight_ = (
            L_self_modules_blocks_modules_20_modules_linear_tokens_parameters_weight_
        )
        l_self_modules_blocks_modules_20_modules_linear_tokens_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_linear_tokens_parameters_bias_
        )
        l_self_modules_blocks_modules_20_parameters_ls2_ = (
            L_self_modules_blocks_modules_20_parameters_ls2_
        )
        l_self_modules_blocks_modules_20_modules_norm2_parameters_beta_ = (
            L_self_modules_blocks_modules_20_modules_norm2_parameters_beta_
        )
        l_self_modules_blocks_modules_20_modules_norm2_parameters_alpha_ = (
            L_self_modules_blocks_modules_20_modules_norm2_parameters_alpha_
        )
        l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_21_parameters_ls1_ = (
            L_self_modules_blocks_modules_21_parameters_ls1_
        )
        l_self_modules_blocks_modules_21_modules_norm1_parameters_beta_ = (
            L_self_modules_blocks_modules_21_modules_norm1_parameters_beta_
        )
        l_self_modules_blocks_modules_21_modules_norm1_parameters_alpha_ = (
            L_self_modules_blocks_modules_21_modules_norm1_parameters_alpha_
        )
        l_self_modules_blocks_modules_21_modules_linear_tokens_parameters_weight_ = (
            L_self_modules_blocks_modules_21_modules_linear_tokens_parameters_weight_
        )
        l_self_modules_blocks_modules_21_modules_linear_tokens_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_linear_tokens_parameters_bias_
        )
        l_self_modules_blocks_modules_21_parameters_ls2_ = (
            L_self_modules_blocks_modules_21_parameters_ls2_
        )
        l_self_modules_blocks_modules_21_modules_norm2_parameters_beta_ = (
            L_self_modules_blocks_modules_21_modules_norm2_parameters_beta_
        )
        l_self_modules_blocks_modules_21_modules_norm2_parameters_alpha_ = (
            L_self_modules_blocks_modules_21_modules_norm2_parameters_alpha_
        )
        l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_22_parameters_ls1_ = (
            L_self_modules_blocks_modules_22_parameters_ls1_
        )
        l_self_modules_blocks_modules_22_modules_norm1_parameters_beta_ = (
            L_self_modules_blocks_modules_22_modules_norm1_parameters_beta_
        )
        l_self_modules_blocks_modules_22_modules_norm1_parameters_alpha_ = (
            L_self_modules_blocks_modules_22_modules_norm1_parameters_alpha_
        )
        l_self_modules_blocks_modules_22_modules_linear_tokens_parameters_weight_ = (
            L_self_modules_blocks_modules_22_modules_linear_tokens_parameters_weight_
        )
        l_self_modules_blocks_modules_22_modules_linear_tokens_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_linear_tokens_parameters_bias_
        )
        l_self_modules_blocks_modules_22_parameters_ls2_ = (
            L_self_modules_blocks_modules_22_parameters_ls2_
        )
        l_self_modules_blocks_modules_22_modules_norm2_parameters_beta_ = (
            L_self_modules_blocks_modules_22_modules_norm2_parameters_beta_
        )
        l_self_modules_blocks_modules_22_modules_norm2_parameters_alpha_ = (
            L_self_modules_blocks_modules_22_modules_norm2_parameters_alpha_
        )
        l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_23_parameters_ls1_ = (
            L_self_modules_blocks_modules_23_parameters_ls1_
        )
        l_self_modules_blocks_modules_23_modules_norm1_parameters_beta_ = (
            L_self_modules_blocks_modules_23_modules_norm1_parameters_beta_
        )
        l_self_modules_blocks_modules_23_modules_norm1_parameters_alpha_ = (
            L_self_modules_blocks_modules_23_modules_norm1_parameters_alpha_
        )
        l_self_modules_blocks_modules_23_modules_linear_tokens_parameters_weight_ = (
            L_self_modules_blocks_modules_23_modules_linear_tokens_parameters_weight_
        )
        l_self_modules_blocks_modules_23_modules_linear_tokens_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_linear_tokens_parameters_bias_
        )
        l_self_modules_blocks_modules_23_parameters_ls2_ = (
            L_self_modules_blocks_modules_23_parameters_ls2_
        )
        l_self_modules_blocks_modules_23_modules_norm2_parameters_beta_ = (
            L_self_modules_blocks_modules_23_modules_norm2_parameters_beta_
        )
        l_self_modules_blocks_modules_23_modules_norm2_parameters_alpha_ = (
            L_self_modules_blocks_modules_23_modules_norm2_parameters_alpha_
        )
        l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_norm_parameters_beta_ = L_self_modules_norm_parameters_beta_
        l_self_modules_norm_parameters_alpha_ = L_self_modules_norm_parameters_alpha_
        l_self_modules_head_parameters_weight_ = L_self_modules_head_parameters_weight_
        l_self_modules_head_parameters_bias_ = L_self_modules_head_parameters_bias_
        x = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_proj_parameters_weight_,
            l_self_modules_stem_modules_proj_parameters_bias_,
            (16, 16),
            (0, 0),
            (1, 1),
            1,
        )
        l_x_ = (
            l_self_modules_stem_modules_proj_parameters_weight_
        ) = l_self_modules_stem_modules_proj_parameters_bias_ = None
        flatten = x.flatten(2)
        x = None
        x_1 = flatten.transpose(1, 2)
        flatten = None
        addcmul = torch.addcmul(
            l_self_modules_blocks_modules_0_modules_norm1_parameters_beta_,
            l_self_modules_blocks_modules_0_modules_norm1_parameters_alpha_,
            x_1,
        )
        l_self_modules_blocks_modules_0_modules_norm1_parameters_beta_ = (
            l_self_modules_blocks_modules_0_modules_norm1_parameters_alpha_
        ) = None
        transpose_1 = addcmul.transpose(1, 2)
        addcmul = None
        linear = torch._C._nn.linear(
            transpose_1,
            l_self_modules_blocks_modules_0_modules_linear_tokens_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_linear_tokens_parameters_bias_,
        )
        transpose_1 = (
            l_self_modules_blocks_modules_0_modules_linear_tokens_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_linear_tokens_parameters_bias_
        ) = None
        transpose_2 = linear.transpose(1, 2)
        linear = None
        mul = l_self_modules_blocks_modules_0_parameters_ls1_ * transpose_2
        l_self_modules_blocks_modules_0_parameters_ls1_ = transpose_2 = None
        x_2 = x_1 + mul
        x_1 = mul = None
        addcmul_1 = torch.addcmul(
            l_self_modules_blocks_modules_0_modules_norm2_parameters_beta_,
            l_self_modules_blocks_modules_0_modules_norm2_parameters_alpha_,
            x_2,
        )
        l_self_modules_blocks_modules_0_modules_norm2_parameters_beta_ = (
            l_self_modules_blocks_modules_0_modules_norm2_parameters_alpha_
        ) = None
        x_3 = torch._C._nn.linear(
            addcmul_1,
            l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        addcmul_1 = l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_4 = torch._C._nn.gelu(x_3, approximate="none")
        x_3 = None
        x_5 = torch.nn.functional.dropout(x_4, 0.0, False, False)
        x_4 = None
        x_6 = torch._C._nn.linear(
            x_5,
            l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_5 = l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_7 = torch.nn.functional.dropout(x_6, 0.0, False, False)
        x_6 = None
        mul_1 = l_self_modules_blocks_modules_0_parameters_ls2_ * x_7
        l_self_modules_blocks_modules_0_parameters_ls2_ = x_7 = None
        x_8 = x_2 + mul_1
        x_2 = mul_1 = None
        addcmul_2 = torch.addcmul(
            l_self_modules_blocks_modules_1_modules_norm1_parameters_beta_,
            l_self_modules_blocks_modules_1_modules_norm1_parameters_alpha_,
            x_8,
        )
        l_self_modules_blocks_modules_1_modules_norm1_parameters_beta_ = (
            l_self_modules_blocks_modules_1_modules_norm1_parameters_alpha_
        ) = None
        transpose_3 = addcmul_2.transpose(1, 2)
        addcmul_2 = None
        linear_3 = torch._C._nn.linear(
            transpose_3,
            l_self_modules_blocks_modules_1_modules_linear_tokens_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_linear_tokens_parameters_bias_,
        )
        transpose_3 = (
            l_self_modules_blocks_modules_1_modules_linear_tokens_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_linear_tokens_parameters_bias_
        ) = None
        transpose_4 = linear_3.transpose(1, 2)
        linear_3 = None
        mul_2 = l_self_modules_blocks_modules_1_parameters_ls1_ * transpose_4
        l_self_modules_blocks_modules_1_parameters_ls1_ = transpose_4 = None
        x_9 = x_8 + mul_2
        x_8 = mul_2 = None
        addcmul_3 = torch.addcmul(
            l_self_modules_blocks_modules_1_modules_norm2_parameters_beta_,
            l_self_modules_blocks_modules_1_modules_norm2_parameters_alpha_,
            x_9,
        )
        l_self_modules_blocks_modules_1_modules_norm2_parameters_beta_ = (
            l_self_modules_blocks_modules_1_modules_norm2_parameters_alpha_
        ) = None
        x_10 = torch._C._nn.linear(
            addcmul_3,
            l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        addcmul_3 = l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_11 = torch._C._nn.gelu(x_10, approximate="none")
        x_10 = None
        x_12 = torch.nn.functional.dropout(x_11, 0.0, False, False)
        x_11 = None
        x_13 = torch._C._nn.linear(
            x_12,
            l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_12 = l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_14 = torch.nn.functional.dropout(x_13, 0.0, False, False)
        x_13 = None
        mul_3 = l_self_modules_blocks_modules_1_parameters_ls2_ * x_14
        l_self_modules_blocks_modules_1_parameters_ls2_ = x_14 = None
        x_15 = x_9 + mul_3
        x_9 = mul_3 = None
        addcmul_4 = torch.addcmul(
            l_self_modules_blocks_modules_2_modules_norm1_parameters_beta_,
            l_self_modules_blocks_modules_2_modules_norm1_parameters_alpha_,
            x_15,
        )
        l_self_modules_blocks_modules_2_modules_norm1_parameters_beta_ = (
            l_self_modules_blocks_modules_2_modules_norm1_parameters_alpha_
        ) = None
        transpose_5 = addcmul_4.transpose(1, 2)
        addcmul_4 = None
        linear_6 = torch._C._nn.linear(
            transpose_5,
            l_self_modules_blocks_modules_2_modules_linear_tokens_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_linear_tokens_parameters_bias_,
        )
        transpose_5 = (
            l_self_modules_blocks_modules_2_modules_linear_tokens_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_linear_tokens_parameters_bias_
        ) = None
        transpose_6 = linear_6.transpose(1, 2)
        linear_6 = None
        mul_4 = l_self_modules_blocks_modules_2_parameters_ls1_ * transpose_6
        l_self_modules_blocks_modules_2_parameters_ls1_ = transpose_6 = None
        x_16 = x_15 + mul_4
        x_15 = mul_4 = None
        addcmul_5 = torch.addcmul(
            l_self_modules_blocks_modules_2_modules_norm2_parameters_beta_,
            l_self_modules_blocks_modules_2_modules_norm2_parameters_alpha_,
            x_16,
        )
        l_self_modules_blocks_modules_2_modules_norm2_parameters_beta_ = (
            l_self_modules_blocks_modules_2_modules_norm2_parameters_alpha_
        ) = None
        x_17 = torch._C._nn.linear(
            addcmul_5,
            l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        addcmul_5 = l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_18 = torch._C._nn.gelu(x_17, approximate="none")
        x_17 = None
        x_19 = torch.nn.functional.dropout(x_18, 0.0, False, False)
        x_18 = None
        x_20 = torch._C._nn.linear(
            x_19,
            l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_19 = l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_21 = torch.nn.functional.dropout(x_20, 0.0, False, False)
        x_20 = None
        mul_5 = l_self_modules_blocks_modules_2_parameters_ls2_ * x_21
        l_self_modules_blocks_modules_2_parameters_ls2_ = x_21 = None
        x_22 = x_16 + mul_5
        x_16 = mul_5 = None
        addcmul_6 = torch.addcmul(
            l_self_modules_blocks_modules_3_modules_norm1_parameters_beta_,
            l_self_modules_blocks_modules_3_modules_norm1_parameters_alpha_,
            x_22,
        )
        l_self_modules_blocks_modules_3_modules_norm1_parameters_beta_ = (
            l_self_modules_blocks_modules_3_modules_norm1_parameters_alpha_
        ) = None
        transpose_7 = addcmul_6.transpose(1, 2)
        addcmul_6 = None
        linear_9 = torch._C._nn.linear(
            transpose_7,
            l_self_modules_blocks_modules_3_modules_linear_tokens_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_linear_tokens_parameters_bias_,
        )
        transpose_7 = (
            l_self_modules_blocks_modules_3_modules_linear_tokens_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_linear_tokens_parameters_bias_
        ) = None
        transpose_8 = linear_9.transpose(1, 2)
        linear_9 = None
        mul_6 = l_self_modules_blocks_modules_3_parameters_ls1_ * transpose_8
        l_self_modules_blocks_modules_3_parameters_ls1_ = transpose_8 = None
        x_23 = x_22 + mul_6
        x_22 = mul_6 = None
        addcmul_7 = torch.addcmul(
            l_self_modules_blocks_modules_3_modules_norm2_parameters_beta_,
            l_self_modules_blocks_modules_3_modules_norm2_parameters_alpha_,
            x_23,
        )
        l_self_modules_blocks_modules_3_modules_norm2_parameters_beta_ = (
            l_self_modules_blocks_modules_3_modules_norm2_parameters_alpha_
        ) = None
        x_24 = torch._C._nn.linear(
            addcmul_7,
            l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        addcmul_7 = l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_25 = torch._C._nn.gelu(x_24, approximate="none")
        x_24 = None
        x_26 = torch.nn.functional.dropout(x_25, 0.0, False, False)
        x_25 = None
        x_27 = torch._C._nn.linear(
            x_26,
            l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_26 = l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_28 = torch.nn.functional.dropout(x_27, 0.0, False, False)
        x_27 = None
        mul_7 = l_self_modules_blocks_modules_3_parameters_ls2_ * x_28
        l_self_modules_blocks_modules_3_parameters_ls2_ = x_28 = None
        x_29 = x_23 + mul_7
        x_23 = mul_7 = None
        addcmul_8 = torch.addcmul(
            l_self_modules_blocks_modules_4_modules_norm1_parameters_beta_,
            l_self_modules_blocks_modules_4_modules_norm1_parameters_alpha_,
            x_29,
        )
        l_self_modules_blocks_modules_4_modules_norm1_parameters_beta_ = (
            l_self_modules_blocks_modules_4_modules_norm1_parameters_alpha_
        ) = None
        transpose_9 = addcmul_8.transpose(1, 2)
        addcmul_8 = None
        linear_12 = torch._C._nn.linear(
            transpose_9,
            l_self_modules_blocks_modules_4_modules_linear_tokens_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_linear_tokens_parameters_bias_,
        )
        transpose_9 = (
            l_self_modules_blocks_modules_4_modules_linear_tokens_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_linear_tokens_parameters_bias_
        ) = None
        transpose_10 = linear_12.transpose(1, 2)
        linear_12 = None
        mul_8 = l_self_modules_blocks_modules_4_parameters_ls1_ * transpose_10
        l_self_modules_blocks_modules_4_parameters_ls1_ = transpose_10 = None
        x_30 = x_29 + mul_8
        x_29 = mul_8 = None
        addcmul_9 = torch.addcmul(
            l_self_modules_blocks_modules_4_modules_norm2_parameters_beta_,
            l_self_modules_blocks_modules_4_modules_norm2_parameters_alpha_,
            x_30,
        )
        l_self_modules_blocks_modules_4_modules_norm2_parameters_beta_ = (
            l_self_modules_blocks_modules_4_modules_norm2_parameters_alpha_
        ) = None
        x_31 = torch._C._nn.linear(
            addcmul_9,
            l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        addcmul_9 = l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_32 = torch._C._nn.gelu(x_31, approximate="none")
        x_31 = None
        x_33 = torch.nn.functional.dropout(x_32, 0.0, False, False)
        x_32 = None
        x_34 = torch._C._nn.linear(
            x_33,
            l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_33 = l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_35 = torch.nn.functional.dropout(x_34, 0.0, False, False)
        x_34 = None
        mul_9 = l_self_modules_blocks_modules_4_parameters_ls2_ * x_35
        l_self_modules_blocks_modules_4_parameters_ls2_ = x_35 = None
        x_36 = x_30 + mul_9
        x_30 = mul_9 = None
        addcmul_10 = torch.addcmul(
            l_self_modules_blocks_modules_5_modules_norm1_parameters_beta_,
            l_self_modules_blocks_modules_5_modules_norm1_parameters_alpha_,
            x_36,
        )
        l_self_modules_blocks_modules_5_modules_norm1_parameters_beta_ = (
            l_self_modules_blocks_modules_5_modules_norm1_parameters_alpha_
        ) = None
        transpose_11 = addcmul_10.transpose(1, 2)
        addcmul_10 = None
        linear_15 = torch._C._nn.linear(
            transpose_11,
            l_self_modules_blocks_modules_5_modules_linear_tokens_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_linear_tokens_parameters_bias_,
        )
        transpose_11 = (
            l_self_modules_blocks_modules_5_modules_linear_tokens_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_linear_tokens_parameters_bias_
        ) = None
        transpose_12 = linear_15.transpose(1, 2)
        linear_15 = None
        mul_10 = l_self_modules_blocks_modules_5_parameters_ls1_ * transpose_12
        l_self_modules_blocks_modules_5_parameters_ls1_ = transpose_12 = None
        x_37 = x_36 + mul_10
        x_36 = mul_10 = None
        addcmul_11 = torch.addcmul(
            l_self_modules_blocks_modules_5_modules_norm2_parameters_beta_,
            l_self_modules_blocks_modules_5_modules_norm2_parameters_alpha_,
            x_37,
        )
        l_self_modules_blocks_modules_5_modules_norm2_parameters_beta_ = (
            l_self_modules_blocks_modules_5_modules_norm2_parameters_alpha_
        ) = None
        x_38 = torch._C._nn.linear(
            addcmul_11,
            l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        addcmul_11 = l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_39 = torch._C._nn.gelu(x_38, approximate="none")
        x_38 = None
        x_40 = torch.nn.functional.dropout(x_39, 0.0, False, False)
        x_39 = None
        x_41 = torch._C._nn.linear(
            x_40,
            l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_40 = l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_42 = torch.nn.functional.dropout(x_41, 0.0, False, False)
        x_41 = None
        mul_11 = l_self_modules_blocks_modules_5_parameters_ls2_ * x_42
        l_self_modules_blocks_modules_5_parameters_ls2_ = x_42 = None
        x_43 = x_37 + mul_11
        x_37 = mul_11 = None
        addcmul_12 = torch.addcmul(
            l_self_modules_blocks_modules_6_modules_norm1_parameters_beta_,
            l_self_modules_blocks_modules_6_modules_norm1_parameters_alpha_,
            x_43,
        )
        l_self_modules_blocks_modules_6_modules_norm1_parameters_beta_ = (
            l_self_modules_blocks_modules_6_modules_norm1_parameters_alpha_
        ) = None
        transpose_13 = addcmul_12.transpose(1, 2)
        addcmul_12 = None
        linear_18 = torch._C._nn.linear(
            transpose_13,
            l_self_modules_blocks_modules_6_modules_linear_tokens_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_linear_tokens_parameters_bias_,
        )
        transpose_13 = (
            l_self_modules_blocks_modules_6_modules_linear_tokens_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_linear_tokens_parameters_bias_
        ) = None
        transpose_14 = linear_18.transpose(1, 2)
        linear_18 = None
        mul_12 = l_self_modules_blocks_modules_6_parameters_ls1_ * transpose_14
        l_self_modules_blocks_modules_6_parameters_ls1_ = transpose_14 = None
        x_44 = x_43 + mul_12
        x_43 = mul_12 = None
        addcmul_13 = torch.addcmul(
            l_self_modules_blocks_modules_6_modules_norm2_parameters_beta_,
            l_self_modules_blocks_modules_6_modules_norm2_parameters_alpha_,
            x_44,
        )
        l_self_modules_blocks_modules_6_modules_norm2_parameters_beta_ = (
            l_self_modules_blocks_modules_6_modules_norm2_parameters_alpha_
        ) = None
        x_45 = torch._C._nn.linear(
            addcmul_13,
            l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        addcmul_13 = l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_46 = torch._C._nn.gelu(x_45, approximate="none")
        x_45 = None
        x_47 = torch.nn.functional.dropout(x_46, 0.0, False, False)
        x_46 = None
        x_48 = torch._C._nn.linear(
            x_47,
            l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_47 = l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_49 = torch.nn.functional.dropout(x_48, 0.0, False, False)
        x_48 = None
        mul_13 = l_self_modules_blocks_modules_6_parameters_ls2_ * x_49
        l_self_modules_blocks_modules_6_parameters_ls2_ = x_49 = None
        x_50 = x_44 + mul_13
        x_44 = mul_13 = None
        addcmul_14 = torch.addcmul(
            l_self_modules_blocks_modules_7_modules_norm1_parameters_beta_,
            l_self_modules_blocks_modules_7_modules_norm1_parameters_alpha_,
            x_50,
        )
        l_self_modules_blocks_modules_7_modules_norm1_parameters_beta_ = (
            l_self_modules_blocks_modules_7_modules_norm1_parameters_alpha_
        ) = None
        transpose_15 = addcmul_14.transpose(1, 2)
        addcmul_14 = None
        linear_21 = torch._C._nn.linear(
            transpose_15,
            l_self_modules_blocks_modules_7_modules_linear_tokens_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_linear_tokens_parameters_bias_,
        )
        transpose_15 = (
            l_self_modules_blocks_modules_7_modules_linear_tokens_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_linear_tokens_parameters_bias_
        ) = None
        transpose_16 = linear_21.transpose(1, 2)
        linear_21 = None
        mul_14 = l_self_modules_blocks_modules_7_parameters_ls1_ * transpose_16
        l_self_modules_blocks_modules_7_parameters_ls1_ = transpose_16 = None
        x_51 = x_50 + mul_14
        x_50 = mul_14 = None
        addcmul_15 = torch.addcmul(
            l_self_modules_blocks_modules_7_modules_norm2_parameters_beta_,
            l_self_modules_blocks_modules_7_modules_norm2_parameters_alpha_,
            x_51,
        )
        l_self_modules_blocks_modules_7_modules_norm2_parameters_beta_ = (
            l_self_modules_blocks_modules_7_modules_norm2_parameters_alpha_
        ) = None
        x_52 = torch._C._nn.linear(
            addcmul_15,
            l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        addcmul_15 = l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_53 = torch._C._nn.gelu(x_52, approximate="none")
        x_52 = None
        x_54 = torch.nn.functional.dropout(x_53, 0.0, False, False)
        x_53 = None
        x_55 = torch._C._nn.linear(
            x_54,
            l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_54 = l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_56 = torch.nn.functional.dropout(x_55, 0.0, False, False)
        x_55 = None
        mul_15 = l_self_modules_blocks_modules_7_parameters_ls2_ * x_56
        l_self_modules_blocks_modules_7_parameters_ls2_ = x_56 = None
        x_57 = x_51 + mul_15
        x_51 = mul_15 = None
        addcmul_16 = torch.addcmul(
            l_self_modules_blocks_modules_8_modules_norm1_parameters_beta_,
            l_self_modules_blocks_modules_8_modules_norm1_parameters_alpha_,
            x_57,
        )
        l_self_modules_blocks_modules_8_modules_norm1_parameters_beta_ = (
            l_self_modules_blocks_modules_8_modules_norm1_parameters_alpha_
        ) = None
        transpose_17 = addcmul_16.transpose(1, 2)
        addcmul_16 = None
        linear_24 = torch._C._nn.linear(
            transpose_17,
            l_self_modules_blocks_modules_8_modules_linear_tokens_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_linear_tokens_parameters_bias_,
        )
        transpose_17 = (
            l_self_modules_blocks_modules_8_modules_linear_tokens_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_linear_tokens_parameters_bias_
        ) = None
        transpose_18 = linear_24.transpose(1, 2)
        linear_24 = None
        mul_16 = l_self_modules_blocks_modules_8_parameters_ls1_ * transpose_18
        l_self_modules_blocks_modules_8_parameters_ls1_ = transpose_18 = None
        x_58 = x_57 + mul_16
        x_57 = mul_16 = None
        addcmul_17 = torch.addcmul(
            l_self_modules_blocks_modules_8_modules_norm2_parameters_beta_,
            l_self_modules_blocks_modules_8_modules_norm2_parameters_alpha_,
            x_58,
        )
        l_self_modules_blocks_modules_8_modules_norm2_parameters_beta_ = (
            l_self_modules_blocks_modules_8_modules_norm2_parameters_alpha_
        ) = None
        x_59 = torch._C._nn.linear(
            addcmul_17,
            l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        addcmul_17 = l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_60 = torch._C._nn.gelu(x_59, approximate="none")
        x_59 = None
        x_61 = torch.nn.functional.dropout(x_60, 0.0, False, False)
        x_60 = None
        x_62 = torch._C._nn.linear(
            x_61,
            l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_61 = l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_63 = torch.nn.functional.dropout(x_62, 0.0, False, False)
        x_62 = None
        mul_17 = l_self_modules_blocks_modules_8_parameters_ls2_ * x_63
        l_self_modules_blocks_modules_8_parameters_ls2_ = x_63 = None
        x_64 = x_58 + mul_17
        x_58 = mul_17 = None
        addcmul_18 = torch.addcmul(
            l_self_modules_blocks_modules_9_modules_norm1_parameters_beta_,
            l_self_modules_blocks_modules_9_modules_norm1_parameters_alpha_,
            x_64,
        )
        l_self_modules_blocks_modules_9_modules_norm1_parameters_beta_ = (
            l_self_modules_blocks_modules_9_modules_norm1_parameters_alpha_
        ) = None
        transpose_19 = addcmul_18.transpose(1, 2)
        addcmul_18 = None
        linear_27 = torch._C._nn.linear(
            transpose_19,
            l_self_modules_blocks_modules_9_modules_linear_tokens_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_linear_tokens_parameters_bias_,
        )
        transpose_19 = (
            l_self_modules_blocks_modules_9_modules_linear_tokens_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_linear_tokens_parameters_bias_
        ) = None
        transpose_20 = linear_27.transpose(1, 2)
        linear_27 = None
        mul_18 = l_self_modules_blocks_modules_9_parameters_ls1_ * transpose_20
        l_self_modules_blocks_modules_9_parameters_ls1_ = transpose_20 = None
        x_65 = x_64 + mul_18
        x_64 = mul_18 = None
        addcmul_19 = torch.addcmul(
            l_self_modules_blocks_modules_9_modules_norm2_parameters_beta_,
            l_self_modules_blocks_modules_9_modules_norm2_parameters_alpha_,
            x_65,
        )
        l_self_modules_blocks_modules_9_modules_norm2_parameters_beta_ = (
            l_self_modules_blocks_modules_9_modules_norm2_parameters_alpha_
        ) = None
        x_66 = torch._C._nn.linear(
            addcmul_19,
            l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        addcmul_19 = l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_67 = torch._C._nn.gelu(x_66, approximate="none")
        x_66 = None
        x_68 = torch.nn.functional.dropout(x_67, 0.0, False, False)
        x_67 = None
        x_69 = torch._C._nn.linear(
            x_68,
            l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_68 = l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_70 = torch.nn.functional.dropout(x_69, 0.0, False, False)
        x_69 = None
        mul_19 = l_self_modules_blocks_modules_9_parameters_ls2_ * x_70
        l_self_modules_blocks_modules_9_parameters_ls2_ = x_70 = None
        x_71 = x_65 + mul_19
        x_65 = mul_19 = None
        addcmul_20 = torch.addcmul(
            l_self_modules_blocks_modules_10_modules_norm1_parameters_beta_,
            l_self_modules_blocks_modules_10_modules_norm1_parameters_alpha_,
            x_71,
        )
        l_self_modules_blocks_modules_10_modules_norm1_parameters_beta_ = (
            l_self_modules_blocks_modules_10_modules_norm1_parameters_alpha_
        ) = None
        transpose_21 = addcmul_20.transpose(1, 2)
        addcmul_20 = None
        linear_30 = torch._C._nn.linear(
            transpose_21,
            l_self_modules_blocks_modules_10_modules_linear_tokens_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_linear_tokens_parameters_bias_,
        )
        transpose_21 = (
            l_self_modules_blocks_modules_10_modules_linear_tokens_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_linear_tokens_parameters_bias_
        ) = None
        transpose_22 = linear_30.transpose(1, 2)
        linear_30 = None
        mul_20 = l_self_modules_blocks_modules_10_parameters_ls1_ * transpose_22
        l_self_modules_blocks_modules_10_parameters_ls1_ = transpose_22 = None
        x_72 = x_71 + mul_20
        x_71 = mul_20 = None
        addcmul_21 = torch.addcmul(
            l_self_modules_blocks_modules_10_modules_norm2_parameters_beta_,
            l_self_modules_blocks_modules_10_modules_norm2_parameters_alpha_,
            x_72,
        )
        l_self_modules_blocks_modules_10_modules_norm2_parameters_beta_ = (
            l_self_modules_blocks_modules_10_modules_norm2_parameters_alpha_
        ) = None
        x_73 = torch._C._nn.linear(
            addcmul_21,
            l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        addcmul_21 = l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_74 = torch._C._nn.gelu(x_73, approximate="none")
        x_73 = None
        x_75 = torch.nn.functional.dropout(x_74, 0.0, False, False)
        x_74 = None
        x_76 = torch._C._nn.linear(
            x_75,
            l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_75 = l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_77 = torch.nn.functional.dropout(x_76, 0.0, False, False)
        x_76 = None
        mul_21 = l_self_modules_blocks_modules_10_parameters_ls2_ * x_77
        l_self_modules_blocks_modules_10_parameters_ls2_ = x_77 = None
        x_78 = x_72 + mul_21
        x_72 = mul_21 = None
        addcmul_22 = torch.addcmul(
            l_self_modules_blocks_modules_11_modules_norm1_parameters_beta_,
            l_self_modules_blocks_modules_11_modules_norm1_parameters_alpha_,
            x_78,
        )
        l_self_modules_blocks_modules_11_modules_norm1_parameters_beta_ = (
            l_self_modules_blocks_modules_11_modules_norm1_parameters_alpha_
        ) = None
        transpose_23 = addcmul_22.transpose(1, 2)
        addcmul_22 = None
        linear_33 = torch._C._nn.linear(
            transpose_23,
            l_self_modules_blocks_modules_11_modules_linear_tokens_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_linear_tokens_parameters_bias_,
        )
        transpose_23 = (
            l_self_modules_blocks_modules_11_modules_linear_tokens_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_linear_tokens_parameters_bias_
        ) = None
        transpose_24 = linear_33.transpose(1, 2)
        linear_33 = None
        mul_22 = l_self_modules_blocks_modules_11_parameters_ls1_ * transpose_24
        l_self_modules_blocks_modules_11_parameters_ls1_ = transpose_24 = None
        x_79 = x_78 + mul_22
        x_78 = mul_22 = None
        addcmul_23 = torch.addcmul(
            l_self_modules_blocks_modules_11_modules_norm2_parameters_beta_,
            l_self_modules_blocks_modules_11_modules_norm2_parameters_alpha_,
            x_79,
        )
        l_self_modules_blocks_modules_11_modules_norm2_parameters_beta_ = (
            l_self_modules_blocks_modules_11_modules_norm2_parameters_alpha_
        ) = None
        x_80 = torch._C._nn.linear(
            addcmul_23,
            l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        addcmul_23 = l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_81 = torch._C._nn.gelu(x_80, approximate="none")
        x_80 = None
        x_82 = torch.nn.functional.dropout(x_81, 0.0, False, False)
        x_81 = None
        x_83 = torch._C._nn.linear(
            x_82,
            l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_82 = l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_84 = torch.nn.functional.dropout(x_83, 0.0, False, False)
        x_83 = None
        mul_23 = l_self_modules_blocks_modules_11_parameters_ls2_ * x_84
        l_self_modules_blocks_modules_11_parameters_ls2_ = x_84 = None
        x_85 = x_79 + mul_23
        x_79 = mul_23 = None
        addcmul_24 = torch.addcmul(
            l_self_modules_blocks_modules_12_modules_norm1_parameters_beta_,
            l_self_modules_blocks_modules_12_modules_norm1_parameters_alpha_,
            x_85,
        )
        l_self_modules_blocks_modules_12_modules_norm1_parameters_beta_ = (
            l_self_modules_blocks_modules_12_modules_norm1_parameters_alpha_
        ) = None
        transpose_25 = addcmul_24.transpose(1, 2)
        addcmul_24 = None
        linear_36 = torch._C._nn.linear(
            transpose_25,
            l_self_modules_blocks_modules_12_modules_linear_tokens_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_linear_tokens_parameters_bias_,
        )
        transpose_25 = (
            l_self_modules_blocks_modules_12_modules_linear_tokens_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_12_modules_linear_tokens_parameters_bias_
        ) = None
        transpose_26 = linear_36.transpose(1, 2)
        linear_36 = None
        mul_24 = l_self_modules_blocks_modules_12_parameters_ls1_ * transpose_26
        l_self_modules_blocks_modules_12_parameters_ls1_ = transpose_26 = None
        x_86 = x_85 + mul_24
        x_85 = mul_24 = None
        addcmul_25 = torch.addcmul(
            l_self_modules_blocks_modules_12_modules_norm2_parameters_beta_,
            l_self_modules_blocks_modules_12_modules_norm2_parameters_alpha_,
            x_86,
        )
        l_self_modules_blocks_modules_12_modules_norm2_parameters_beta_ = (
            l_self_modules_blocks_modules_12_modules_norm2_parameters_alpha_
        ) = None
        x_87 = torch._C._nn.linear(
            addcmul_25,
            l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        addcmul_25 = l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_88 = torch._C._nn.gelu(x_87, approximate="none")
        x_87 = None
        x_89 = torch.nn.functional.dropout(x_88, 0.0, False, False)
        x_88 = None
        x_90 = torch._C._nn.linear(
            x_89,
            l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_89 = l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_91 = torch.nn.functional.dropout(x_90, 0.0, False, False)
        x_90 = None
        mul_25 = l_self_modules_blocks_modules_12_parameters_ls2_ * x_91
        l_self_modules_blocks_modules_12_parameters_ls2_ = x_91 = None
        x_92 = x_86 + mul_25
        x_86 = mul_25 = None
        addcmul_26 = torch.addcmul(
            l_self_modules_blocks_modules_13_modules_norm1_parameters_beta_,
            l_self_modules_blocks_modules_13_modules_norm1_parameters_alpha_,
            x_92,
        )
        l_self_modules_blocks_modules_13_modules_norm1_parameters_beta_ = (
            l_self_modules_blocks_modules_13_modules_norm1_parameters_alpha_
        ) = None
        transpose_27 = addcmul_26.transpose(1, 2)
        addcmul_26 = None
        linear_39 = torch._C._nn.linear(
            transpose_27,
            l_self_modules_blocks_modules_13_modules_linear_tokens_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_linear_tokens_parameters_bias_,
        )
        transpose_27 = (
            l_self_modules_blocks_modules_13_modules_linear_tokens_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_13_modules_linear_tokens_parameters_bias_
        ) = None
        transpose_28 = linear_39.transpose(1, 2)
        linear_39 = None
        mul_26 = l_self_modules_blocks_modules_13_parameters_ls1_ * transpose_28
        l_self_modules_blocks_modules_13_parameters_ls1_ = transpose_28 = None
        x_93 = x_92 + mul_26
        x_92 = mul_26 = None
        addcmul_27 = torch.addcmul(
            l_self_modules_blocks_modules_13_modules_norm2_parameters_beta_,
            l_self_modules_blocks_modules_13_modules_norm2_parameters_alpha_,
            x_93,
        )
        l_self_modules_blocks_modules_13_modules_norm2_parameters_beta_ = (
            l_self_modules_blocks_modules_13_modules_norm2_parameters_alpha_
        ) = None
        x_94 = torch._C._nn.linear(
            addcmul_27,
            l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        addcmul_27 = l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_95 = torch._C._nn.gelu(x_94, approximate="none")
        x_94 = None
        x_96 = torch.nn.functional.dropout(x_95, 0.0, False, False)
        x_95 = None
        x_97 = torch._C._nn.linear(
            x_96,
            l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_96 = l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_98 = torch.nn.functional.dropout(x_97, 0.0, False, False)
        x_97 = None
        mul_27 = l_self_modules_blocks_modules_13_parameters_ls2_ * x_98
        l_self_modules_blocks_modules_13_parameters_ls2_ = x_98 = None
        x_99 = x_93 + mul_27
        x_93 = mul_27 = None
        addcmul_28 = torch.addcmul(
            l_self_modules_blocks_modules_14_modules_norm1_parameters_beta_,
            l_self_modules_blocks_modules_14_modules_norm1_parameters_alpha_,
            x_99,
        )
        l_self_modules_blocks_modules_14_modules_norm1_parameters_beta_ = (
            l_self_modules_blocks_modules_14_modules_norm1_parameters_alpha_
        ) = None
        transpose_29 = addcmul_28.transpose(1, 2)
        addcmul_28 = None
        linear_42 = torch._C._nn.linear(
            transpose_29,
            l_self_modules_blocks_modules_14_modules_linear_tokens_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_linear_tokens_parameters_bias_,
        )
        transpose_29 = (
            l_self_modules_blocks_modules_14_modules_linear_tokens_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_14_modules_linear_tokens_parameters_bias_
        ) = None
        transpose_30 = linear_42.transpose(1, 2)
        linear_42 = None
        mul_28 = l_self_modules_blocks_modules_14_parameters_ls1_ * transpose_30
        l_self_modules_blocks_modules_14_parameters_ls1_ = transpose_30 = None
        x_100 = x_99 + mul_28
        x_99 = mul_28 = None
        addcmul_29 = torch.addcmul(
            l_self_modules_blocks_modules_14_modules_norm2_parameters_beta_,
            l_self_modules_blocks_modules_14_modules_norm2_parameters_alpha_,
            x_100,
        )
        l_self_modules_blocks_modules_14_modules_norm2_parameters_beta_ = (
            l_self_modules_blocks_modules_14_modules_norm2_parameters_alpha_
        ) = None
        x_101 = torch._C._nn.linear(
            addcmul_29,
            l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        addcmul_29 = l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_102 = torch._C._nn.gelu(x_101, approximate="none")
        x_101 = None
        x_103 = torch.nn.functional.dropout(x_102, 0.0, False, False)
        x_102 = None
        x_104 = torch._C._nn.linear(
            x_103,
            l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_103 = l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_105 = torch.nn.functional.dropout(x_104, 0.0, False, False)
        x_104 = None
        mul_29 = l_self_modules_blocks_modules_14_parameters_ls2_ * x_105
        l_self_modules_blocks_modules_14_parameters_ls2_ = x_105 = None
        x_106 = x_100 + mul_29
        x_100 = mul_29 = None
        addcmul_30 = torch.addcmul(
            l_self_modules_blocks_modules_15_modules_norm1_parameters_beta_,
            l_self_modules_blocks_modules_15_modules_norm1_parameters_alpha_,
            x_106,
        )
        l_self_modules_blocks_modules_15_modules_norm1_parameters_beta_ = (
            l_self_modules_blocks_modules_15_modules_norm1_parameters_alpha_
        ) = None
        transpose_31 = addcmul_30.transpose(1, 2)
        addcmul_30 = None
        linear_45 = torch._C._nn.linear(
            transpose_31,
            l_self_modules_blocks_modules_15_modules_linear_tokens_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_linear_tokens_parameters_bias_,
        )
        transpose_31 = (
            l_self_modules_blocks_modules_15_modules_linear_tokens_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_15_modules_linear_tokens_parameters_bias_
        ) = None
        transpose_32 = linear_45.transpose(1, 2)
        linear_45 = None
        mul_30 = l_self_modules_blocks_modules_15_parameters_ls1_ * transpose_32
        l_self_modules_blocks_modules_15_parameters_ls1_ = transpose_32 = None
        x_107 = x_106 + mul_30
        x_106 = mul_30 = None
        addcmul_31 = torch.addcmul(
            l_self_modules_blocks_modules_15_modules_norm2_parameters_beta_,
            l_self_modules_blocks_modules_15_modules_norm2_parameters_alpha_,
            x_107,
        )
        l_self_modules_blocks_modules_15_modules_norm2_parameters_beta_ = (
            l_self_modules_blocks_modules_15_modules_norm2_parameters_alpha_
        ) = None
        x_108 = torch._C._nn.linear(
            addcmul_31,
            l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        addcmul_31 = l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_109 = torch._C._nn.gelu(x_108, approximate="none")
        x_108 = None
        x_110 = torch.nn.functional.dropout(x_109, 0.0, False, False)
        x_109 = None
        x_111 = torch._C._nn.linear(
            x_110,
            l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_110 = l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_112 = torch.nn.functional.dropout(x_111, 0.0, False, False)
        x_111 = None
        mul_31 = l_self_modules_blocks_modules_15_parameters_ls2_ * x_112
        l_self_modules_blocks_modules_15_parameters_ls2_ = x_112 = None
        x_113 = x_107 + mul_31
        x_107 = mul_31 = None
        addcmul_32 = torch.addcmul(
            l_self_modules_blocks_modules_16_modules_norm1_parameters_beta_,
            l_self_modules_blocks_modules_16_modules_norm1_parameters_alpha_,
            x_113,
        )
        l_self_modules_blocks_modules_16_modules_norm1_parameters_beta_ = (
            l_self_modules_blocks_modules_16_modules_norm1_parameters_alpha_
        ) = None
        transpose_33 = addcmul_32.transpose(1, 2)
        addcmul_32 = None
        linear_48 = torch._C._nn.linear(
            transpose_33,
            l_self_modules_blocks_modules_16_modules_linear_tokens_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_linear_tokens_parameters_bias_,
        )
        transpose_33 = (
            l_self_modules_blocks_modules_16_modules_linear_tokens_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_16_modules_linear_tokens_parameters_bias_
        ) = None
        transpose_34 = linear_48.transpose(1, 2)
        linear_48 = None
        mul_32 = l_self_modules_blocks_modules_16_parameters_ls1_ * transpose_34
        l_self_modules_blocks_modules_16_parameters_ls1_ = transpose_34 = None
        x_114 = x_113 + mul_32
        x_113 = mul_32 = None
        addcmul_33 = torch.addcmul(
            l_self_modules_blocks_modules_16_modules_norm2_parameters_beta_,
            l_self_modules_blocks_modules_16_modules_norm2_parameters_alpha_,
            x_114,
        )
        l_self_modules_blocks_modules_16_modules_norm2_parameters_beta_ = (
            l_self_modules_blocks_modules_16_modules_norm2_parameters_alpha_
        ) = None
        x_115 = torch._C._nn.linear(
            addcmul_33,
            l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        addcmul_33 = l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_116 = torch._C._nn.gelu(x_115, approximate="none")
        x_115 = None
        x_117 = torch.nn.functional.dropout(x_116, 0.0, False, False)
        x_116 = None
        x_118 = torch._C._nn.linear(
            x_117,
            l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_117 = l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_119 = torch.nn.functional.dropout(x_118, 0.0, False, False)
        x_118 = None
        mul_33 = l_self_modules_blocks_modules_16_parameters_ls2_ * x_119
        l_self_modules_blocks_modules_16_parameters_ls2_ = x_119 = None
        x_120 = x_114 + mul_33
        x_114 = mul_33 = None
        addcmul_34 = torch.addcmul(
            l_self_modules_blocks_modules_17_modules_norm1_parameters_beta_,
            l_self_modules_blocks_modules_17_modules_norm1_parameters_alpha_,
            x_120,
        )
        l_self_modules_blocks_modules_17_modules_norm1_parameters_beta_ = (
            l_self_modules_blocks_modules_17_modules_norm1_parameters_alpha_
        ) = None
        transpose_35 = addcmul_34.transpose(1, 2)
        addcmul_34 = None
        linear_51 = torch._C._nn.linear(
            transpose_35,
            l_self_modules_blocks_modules_17_modules_linear_tokens_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_linear_tokens_parameters_bias_,
        )
        transpose_35 = (
            l_self_modules_blocks_modules_17_modules_linear_tokens_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_17_modules_linear_tokens_parameters_bias_
        ) = None
        transpose_36 = linear_51.transpose(1, 2)
        linear_51 = None
        mul_34 = l_self_modules_blocks_modules_17_parameters_ls1_ * transpose_36
        l_self_modules_blocks_modules_17_parameters_ls1_ = transpose_36 = None
        x_121 = x_120 + mul_34
        x_120 = mul_34 = None
        addcmul_35 = torch.addcmul(
            l_self_modules_blocks_modules_17_modules_norm2_parameters_beta_,
            l_self_modules_blocks_modules_17_modules_norm2_parameters_alpha_,
            x_121,
        )
        l_self_modules_blocks_modules_17_modules_norm2_parameters_beta_ = (
            l_self_modules_blocks_modules_17_modules_norm2_parameters_alpha_
        ) = None
        x_122 = torch._C._nn.linear(
            addcmul_35,
            l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        addcmul_35 = l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_123 = torch._C._nn.gelu(x_122, approximate="none")
        x_122 = None
        x_124 = torch.nn.functional.dropout(x_123, 0.0, False, False)
        x_123 = None
        x_125 = torch._C._nn.linear(
            x_124,
            l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_124 = l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_126 = torch.nn.functional.dropout(x_125, 0.0, False, False)
        x_125 = None
        mul_35 = l_self_modules_blocks_modules_17_parameters_ls2_ * x_126
        l_self_modules_blocks_modules_17_parameters_ls2_ = x_126 = None
        x_127 = x_121 + mul_35
        x_121 = mul_35 = None
        addcmul_36 = torch.addcmul(
            l_self_modules_blocks_modules_18_modules_norm1_parameters_beta_,
            l_self_modules_blocks_modules_18_modules_norm1_parameters_alpha_,
            x_127,
        )
        l_self_modules_blocks_modules_18_modules_norm1_parameters_beta_ = (
            l_self_modules_blocks_modules_18_modules_norm1_parameters_alpha_
        ) = None
        transpose_37 = addcmul_36.transpose(1, 2)
        addcmul_36 = None
        linear_54 = torch._C._nn.linear(
            transpose_37,
            l_self_modules_blocks_modules_18_modules_linear_tokens_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_linear_tokens_parameters_bias_,
        )
        transpose_37 = (
            l_self_modules_blocks_modules_18_modules_linear_tokens_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_18_modules_linear_tokens_parameters_bias_
        ) = None
        transpose_38 = linear_54.transpose(1, 2)
        linear_54 = None
        mul_36 = l_self_modules_blocks_modules_18_parameters_ls1_ * transpose_38
        l_self_modules_blocks_modules_18_parameters_ls1_ = transpose_38 = None
        x_128 = x_127 + mul_36
        x_127 = mul_36 = None
        addcmul_37 = torch.addcmul(
            l_self_modules_blocks_modules_18_modules_norm2_parameters_beta_,
            l_self_modules_blocks_modules_18_modules_norm2_parameters_alpha_,
            x_128,
        )
        l_self_modules_blocks_modules_18_modules_norm2_parameters_beta_ = (
            l_self_modules_blocks_modules_18_modules_norm2_parameters_alpha_
        ) = None
        x_129 = torch._C._nn.linear(
            addcmul_37,
            l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        addcmul_37 = l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_130 = torch._C._nn.gelu(x_129, approximate="none")
        x_129 = None
        x_131 = torch.nn.functional.dropout(x_130, 0.0, False, False)
        x_130 = None
        x_132 = torch._C._nn.linear(
            x_131,
            l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_131 = l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_133 = torch.nn.functional.dropout(x_132, 0.0, False, False)
        x_132 = None
        mul_37 = l_self_modules_blocks_modules_18_parameters_ls2_ * x_133
        l_self_modules_blocks_modules_18_parameters_ls2_ = x_133 = None
        x_134 = x_128 + mul_37
        x_128 = mul_37 = None
        addcmul_38 = torch.addcmul(
            l_self_modules_blocks_modules_19_modules_norm1_parameters_beta_,
            l_self_modules_blocks_modules_19_modules_norm1_parameters_alpha_,
            x_134,
        )
        l_self_modules_blocks_modules_19_modules_norm1_parameters_beta_ = (
            l_self_modules_blocks_modules_19_modules_norm1_parameters_alpha_
        ) = None
        transpose_39 = addcmul_38.transpose(1, 2)
        addcmul_38 = None
        linear_57 = torch._C._nn.linear(
            transpose_39,
            l_self_modules_blocks_modules_19_modules_linear_tokens_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_linear_tokens_parameters_bias_,
        )
        transpose_39 = (
            l_self_modules_blocks_modules_19_modules_linear_tokens_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_19_modules_linear_tokens_parameters_bias_
        ) = None
        transpose_40 = linear_57.transpose(1, 2)
        linear_57 = None
        mul_38 = l_self_modules_blocks_modules_19_parameters_ls1_ * transpose_40
        l_self_modules_blocks_modules_19_parameters_ls1_ = transpose_40 = None
        x_135 = x_134 + mul_38
        x_134 = mul_38 = None
        addcmul_39 = torch.addcmul(
            l_self_modules_blocks_modules_19_modules_norm2_parameters_beta_,
            l_self_modules_blocks_modules_19_modules_norm2_parameters_alpha_,
            x_135,
        )
        l_self_modules_blocks_modules_19_modules_norm2_parameters_beta_ = (
            l_self_modules_blocks_modules_19_modules_norm2_parameters_alpha_
        ) = None
        x_136 = torch._C._nn.linear(
            addcmul_39,
            l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        addcmul_39 = l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_137 = torch._C._nn.gelu(x_136, approximate="none")
        x_136 = None
        x_138 = torch.nn.functional.dropout(x_137, 0.0, False, False)
        x_137 = None
        x_139 = torch._C._nn.linear(
            x_138,
            l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_138 = l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_140 = torch.nn.functional.dropout(x_139, 0.0, False, False)
        x_139 = None
        mul_39 = l_self_modules_blocks_modules_19_parameters_ls2_ * x_140
        l_self_modules_blocks_modules_19_parameters_ls2_ = x_140 = None
        x_141 = x_135 + mul_39
        x_135 = mul_39 = None
        addcmul_40 = torch.addcmul(
            l_self_modules_blocks_modules_20_modules_norm1_parameters_beta_,
            l_self_modules_blocks_modules_20_modules_norm1_parameters_alpha_,
            x_141,
        )
        l_self_modules_blocks_modules_20_modules_norm1_parameters_beta_ = (
            l_self_modules_blocks_modules_20_modules_norm1_parameters_alpha_
        ) = None
        transpose_41 = addcmul_40.transpose(1, 2)
        addcmul_40 = None
        linear_60 = torch._C._nn.linear(
            transpose_41,
            l_self_modules_blocks_modules_20_modules_linear_tokens_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_linear_tokens_parameters_bias_,
        )
        transpose_41 = (
            l_self_modules_blocks_modules_20_modules_linear_tokens_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_20_modules_linear_tokens_parameters_bias_
        ) = None
        transpose_42 = linear_60.transpose(1, 2)
        linear_60 = None
        mul_40 = l_self_modules_blocks_modules_20_parameters_ls1_ * transpose_42
        l_self_modules_blocks_modules_20_parameters_ls1_ = transpose_42 = None
        x_142 = x_141 + mul_40
        x_141 = mul_40 = None
        addcmul_41 = torch.addcmul(
            l_self_modules_blocks_modules_20_modules_norm2_parameters_beta_,
            l_self_modules_blocks_modules_20_modules_norm2_parameters_alpha_,
            x_142,
        )
        l_self_modules_blocks_modules_20_modules_norm2_parameters_beta_ = (
            l_self_modules_blocks_modules_20_modules_norm2_parameters_alpha_
        ) = None
        x_143 = torch._C._nn.linear(
            addcmul_41,
            l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        addcmul_41 = l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_144 = torch._C._nn.gelu(x_143, approximate="none")
        x_143 = None
        x_145 = torch.nn.functional.dropout(x_144, 0.0, False, False)
        x_144 = None
        x_146 = torch._C._nn.linear(
            x_145,
            l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_145 = l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_147 = torch.nn.functional.dropout(x_146, 0.0, False, False)
        x_146 = None
        mul_41 = l_self_modules_blocks_modules_20_parameters_ls2_ * x_147
        l_self_modules_blocks_modules_20_parameters_ls2_ = x_147 = None
        x_148 = x_142 + mul_41
        x_142 = mul_41 = None
        addcmul_42 = torch.addcmul(
            l_self_modules_blocks_modules_21_modules_norm1_parameters_beta_,
            l_self_modules_blocks_modules_21_modules_norm1_parameters_alpha_,
            x_148,
        )
        l_self_modules_blocks_modules_21_modules_norm1_parameters_beta_ = (
            l_self_modules_blocks_modules_21_modules_norm1_parameters_alpha_
        ) = None
        transpose_43 = addcmul_42.transpose(1, 2)
        addcmul_42 = None
        linear_63 = torch._C._nn.linear(
            transpose_43,
            l_self_modules_blocks_modules_21_modules_linear_tokens_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_linear_tokens_parameters_bias_,
        )
        transpose_43 = (
            l_self_modules_blocks_modules_21_modules_linear_tokens_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_21_modules_linear_tokens_parameters_bias_
        ) = None
        transpose_44 = linear_63.transpose(1, 2)
        linear_63 = None
        mul_42 = l_self_modules_blocks_modules_21_parameters_ls1_ * transpose_44
        l_self_modules_blocks_modules_21_parameters_ls1_ = transpose_44 = None
        x_149 = x_148 + mul_42
        x_148 = mul_42 = None
        addcmul_43 = torch.addcmul(
            l_self_modules_blocks_modules_21_modules_norm2_parameters_beta_,
            l_self_modules_blocks_modules_21_modules_norm2_parameters_alpha_,
            x_149,
        )
        l_self_modules_blocks_modules_21_modules_norm2_parameters_beta_ = (
            l_self_modules_blocks_modules_21_modules_norm2_parameters_alpha_
        ) = None
        x_150 = torch._C._nn.linear(
            addcmul_43,
            l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        addcmul_43 = l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_151 = torch._C._nn.gelu(x_150, approximate="none")
        x_150 = None
        x_152 = torch.nn.functional.dropout(x_151, 0.0, False, False)
        x_151 = None
        x_153 = torch._C._nn.linear(
            x_152,
            l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_152 = l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_154 = torch.nn.functional.dropout(x_153, 0.0, False, False)
        x_153 = None
        mul_43 = l_self_modules_blocks_modules_21_parameters_ls2_ * x_154
        l_self_modules_blocks_modules_21_parameters_ls2_ = x_154 = None
        x_155 = x_149 + mul_43
        x_149 = mul_43 = None
        addcmul_44 = torch.addcmul(
            l_self_modules_blocks_modules_22_modules_norm1_parameters_beta_,
            l_self_modules_blocks_modules_22_modules_norm1_parameters_alpha_,
            x_155,
        )
        l_self_modules_blocks_modules_22_modules_norm1_parameters_beta_ = (
            l_self_modules_blocks_modules_22_modules_norm1_parameters_alpha_
        ) = None
        transpose_45 = addcmul_44.transpose(1, 2)
        addcmul_44 = None
        linear_66 = torch._C._nn.linear(
            transpose_45,
            l_self_modules_blocks_modules_22_modules_linear_tokens_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_linear_tokens_parameters_bias_,
        )
        transpose_45 = (
            l_self_modules_blocks_modules_22_modules_linear_tokens_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_22_modules_linear_tokens_parameters_bias_
        ) = None
        transpose_46 = linear_66.transpose(1, 2)
        linear_66 = None
        mul_44 = l_self_modules_blocks_modules_22_parameters_ls1_ * transpose_46
        l_self_modules_blocks_modules_22_parameters_ls1_ = transpose_46 = None
        x_156 = x_155 + mul_44
        x_155 = mul_44 = None
        addcmul_45 = torch.addcmul(
            l_self_modules_blocks_modules_22_modules_norm2_parameters_beta_,
            l_self_modules_blocks_modules_22_modules_norm2_parameters_alpha_,
            x_156,
        )
        l_self_modules_blocks_modules_22_modules_norm2_parameters_beta_ = (
            l_self_modules_blocks_modules_22_modules_norm2_parameters_alpha_
        ) = None
        x_157 = torch._C._nn.linear(
            addcmul_45,
            l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        addcmul_45 = l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_158 = torch._C._nn.gelu(x_157, approximate="none")
        x_157 = None
        x_159 = torch.nn.functional.dropout(x_158, 0.0, False, False)
        x_158 = None
        x_160 = torch._C._nn.linear(
            x_159,
            l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_159 = l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_161 = torch.nn.functional.dropout(x_160, 0.0, False, False)
        x_160 = None
        mul_45 = l_self_modules_blocks_modules_22_parameters_ls2_ * x_161
        l_self_modules_blocks_modules_22_parameters_ls2_ = x_161 = None
        x_162 = x_156 + mul_45
        x_156 = mul_45 = None
        addcmul_46 = torch.addcmul(
            l_self_modules_blocks_modules_23_modules_norm1_parameters_beta_,
            l_self_modules_blocks_modules_23_modules_norm1_parameters_alpha_,
            x_162,
        )
        l_self_modules_blocks_modules_23_modules_norm1_parameters_beta_ = (
            l_self_modules_blocks_modules_23_modules_norm1_parameters_alpha_
        ) = None
        transpose_47 = addcmul_46.transpose(1, 2)
        addcmul_46 = None
        linear_69 = torch._C._nn.linear(
            transpose_47,
            l_self_modules_blocks_modules_23_modules_linear_tokens_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_linear_tokens_parameters_bias_,
        )
        transpose_47 = (
            l_self_modules_blocks_modules_23_modules_linear_tokens_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_23_modules_linear_tokens_parameters_bias_
        ) = None
        transpose_48 = linear_69.transpose(1, 2)
        linear_69 = None
        mul_46 = l_self_modules_blocks_modules_23_parameters_ls1_ * transpose_48
        l_self_modules_blocks_modules_23_parameters_ls1_ = transpose_48 = None
        x_163 = x_162 + mul_46
        x_162 = mul_46 = None
        addcmul_47 = torch.addcmul(
            l_self_modules_blocks_modules_23_modules_norm2_parameters_beta_,
            l_self_modules_blocks_modules_23_modules_norm2_parameters_alpha_,
            x_163,
        )
        l_self_modules_blocks_modules_23_modules_norm2_parameters_beta_ = (
            l_self_modules_blocks_modules_23_modules_norm2_parameters_alpha_
        ) = None
        x_164 = torch._C._nn.linear(
            addcmul_47,
            l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        addcmul_47 = l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_165 = torch._C._nn.gelu(x_164, approximate="none")
        x_164 = None
        x_166 = torch.nn.functional.dropout(x_165, 0.0, False, False)
        x_165 = None
        x_167 = torch._C._nn.linear(
            x_166,
            l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_166 = l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_168 = torch.nn.functional.dropout(x_167, 0.0, False, False)
        x_167 = None
        mul_47 = l_self_modules_blocks_modules_23_parameters_ls2_ * x_168
        l_self_modules_blocks_modules_23_parameters_ls2_ = x_168 = None
        x_169 = x_163 + mul_47
        x_163 = mul_47 = None
        x_170 = torch.addcmul(
            l_self_modules_norm_parameters_beta_,
            l_self_modules_norm_parameters_alpha_,
            x_169,
        )
        l_self_modules_norm_parameters_beta_ = (
            l_self_modules_norm_parameters_alpha_
        ) = x_169 = None
        x_171 = x_170.mean(dim=1)
        x_170 = None
        x_172 = torch.nn.functional.dropout(x_171, 0.0, False, False)
        x_171 = None
        x_173 = torch._C._nn.linear(
            x_172,
            l_self_modules_head_parameters_weight_,
            l_self_modules_head_parameters_bias_,
        )
        x_172 = (
            l_self_modules_head_parameters_weight_
        ) = l_self_modules_head_parameters_bias_ = None
        return (x_173,)
