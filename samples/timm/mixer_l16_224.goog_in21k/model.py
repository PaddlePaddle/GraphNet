import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_tokens_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_tokens_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_tokens_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_tokens_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_tokens_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_tokens_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_tokens_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_tokens_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_tokens_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_tokens_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_tokens_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_tokens_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_tokens_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_tokens_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_tokens_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_tokens_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_tokens_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_tokens_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_tokens_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_tokens_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_tokens_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_tokens_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_tokens_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_tokens_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_tokens_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_tokens_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_tokens_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_tokens_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_tokens_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_tokens_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_tokens_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_tokens_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_tokens_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_tokens_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_tokens_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_tokens_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_tokens_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_tokens_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_tokens_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_tokens_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_tokens_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_tokens_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_tokens_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_tokens_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_tokens_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_tokens_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_tokens_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_tokens_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_tokens_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_tokens_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_tokens_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_tokens_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_tokens_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_tokens_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_tokens_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_tokens_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_tokens_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_tokens_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_tokens_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_tokens_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_tokens_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_tokens_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_tokens_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_tokens_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_tokens_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_tokens_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_tokens_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_tokens_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_tokens_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_tokens_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_tokens_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_tokens_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_tokens_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_tokens_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_tokens_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_tokens_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_tokens_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_tokens_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_tokens_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_tokens_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_tokens_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_tokens_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_tokens_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_tokens_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_tokens_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_tokens_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_tokens_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_tokens_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_tokens_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_tokens_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_tokens_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_tokens_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_tokens_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_tokens_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_tokens_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_tokens_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_mlp_tokens_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_0_modules_mlp_tokens_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_0_modules_mlp_tokens_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_0_modules_mlp_tokens_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_0_modules_mlp_tokens_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_0_modules_mlp_tokens_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_0_modules_mlp_tokens_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_0_modules_mlp_tokens_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_mlp_tokens_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_mlp_tokens_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_1_modules_mlp_tokens_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_1_modules_mlp_tokens_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_1_modules_mlp_tokens_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_1_modules_mlp_tokens_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_1_modules_mlp_tokens_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_1_modules_mlp_tokens_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_mlp_tokens_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_mlp_tokens_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_2_modules_mlp_tokens_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_2_modules_mlp_tokens_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_2_modules_mlp_tokens_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_2_modules_mlp_tokens_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_2_modules_mlp_tokens_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_2_modules_mlp_tokens_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_mlp_tokens_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_3_modules_mlp_tokens_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_3_modules_mlp_tokens_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_3_modules_mlp_tokens_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_3_modules_mlp_tokens_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_3_modules_mlp_tokens_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_3_modules_mlp_tokens_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_3_modules_mlp_tokens_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_mlp_tokens_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_4_modules_mlp_tokens_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_4_modules_mlp_tokens_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_4_modules_mlp_tokens_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_4_modules_mlp_tokens_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_4_modules_mlp_tokens_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_4_modules_mlp_tokens_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_4_modules_mlp_tokens_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_mlp_tokens_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_5_modules_mlp_tokens_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_5_modules_mlp_tokens_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_5_modules_mlp_tokens_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_5_modules_mlp_tokens_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_5_modules_mlp_tokens_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_5_modules_mlp_tokens_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_5_modules_mlp_tokens_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_mlp_tokens_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_6_modules_mlp_tokens_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_6_modules_mlp_tokens_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_6_modules_mlp_tokens_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_6_modules_mlp_tokens_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_6_modules_mlp_tokens_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_6_modules_mlp_tokens_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_6_modules_mlp_tokens_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_mlp_tokens_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_7_modules_mlp_tokens_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_7_modules_mlp_tokens_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_7_modules_mlp_tokens_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_7_modules_mlp_tokens_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_7_modules_mlp_tokens_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_7_modules_mlp_tokens_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_7_modules_mlp_tokens_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_mlp_tokens_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_8_modules_mlp_tokens_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_8_modules_mlp_tokens_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_8_modules_mlp_tokens_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_8_modules_mlp_tokens_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_8_modules_mlp_tokens_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_8_modules_mlp_tokens_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_8_modules_mlp_tokens_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_mlp_tokens_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_9_modules_mlp_tokens_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_9_modules_mlp_tokens_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_9_modules_mlp_tokens_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_9_modules_mlp_tokens_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_9_modules_mlp_tokens_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_9_modules_mlp_tokens_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_9_modules_mlp_tokens_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_10_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_10_modules_mlp_tokens_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_10_modules_mlp_tokens_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_10_modules_mlp_tokens_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_10_modules_mlp_tokens_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_10_modules_mlp_tokens_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_10_modules_mlp_tokens_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_10_modules_mlp_tokens_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_10_modules_mlp_tokens_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_10_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_11_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_11_modules_mlp_tokens_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_11_modules_mlp_tokens_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_11_modules_mlp_tokens_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_11_modules_mlp_tokens_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_11_modules_mlp_tokens_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_11_modules_mlp_tokens_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_11_modules_mlp_tokens_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_11_modules_mlp_tokens_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_11_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_12_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_12_modules_mlp_tokens_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_12_modules_mlp_tokens_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_12_modules_mlp_tokens_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_12_modules_mlp_tokens_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_12_modules_mlp_tokens_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_12_modules_mlp_tokens_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_12_modules_mlp_tokens_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_12_modules_mlp_tokens_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_12_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_12_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_12_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_13_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_13_modules_mlp_tokens_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_13_modules_mlp_tokens_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_13_modules_mlp_tokens_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_13_modules_mlp_tokens_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_13_modules_mlp_tokens_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_13_modules_mlp_tokens_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_13_modules_mlp_tokens_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_13_modules_mlp_tokens_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_13_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_13_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_13_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_14_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_14_modules_mlp_tokens_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_14_modules_mlp_tokens_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_14_modules_mlp_tokens_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_14_modules_mlp_tokens_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_14_modules_mlp_tokens_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_14_modules_mlp_tokens_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_14_modules_mlp_tokens_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_14_modules_mlp_tokens_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_14_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_14_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_14_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_15_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_15_modules_mlp_tokens_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_15_modules_mlp_tokens_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_15_modules_mlp_tokens_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_15_modules_mlp_tokens_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_15_modules_mlp_tokens_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_15_modules_mlp_tokens_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_15_modules_mlp_tokens_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_15_modules_mlp_tokens_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_15_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_15_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_15_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_16_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_16_modules_mlp_tokens_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_16_modules_mlp_tokens_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_16_modules_mlp_tokens_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_16_modules_mlp_tokens_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_16_modules_mlp_tokens_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_16_modules_mlp_tokens_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_16_modules_mlp_tokens_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_16_modules_mlp_tokens_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_16_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_16_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_16_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_17_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_17_modules_mlp_tokens_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_17_modules_mlp_tokens_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_17_modules_mlp_tokens_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_17_modules_mlp_tokens_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_17_modules_mlp_tokens_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_17_modules_mlp_tokens_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_17_modules_mlp_tokens_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_17_modules_mlp_tokens_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_17_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_17_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_17_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_18_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_18_modules_mlp_tokens_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_18_modules_mlp_tokens_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_18_modules_mlp_tokens_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_18_modules_mlp_tokens_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_18_modules_mlp_tokens_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_18_modules_mlp_tokens_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_18_modules_mlp_tokens_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_18_modules_mlp_tokens_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_18_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_18_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_18_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_19_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_19_modules_mlp_tokens_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_19_modules_mlp_tokens_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_19_modules_mlp_tokens_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_19_modules_mlp_tokens_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_19_modules_mlp_tokens_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_19_modules_mlp_tokens_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_19_modules_mlp_tokens_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_19_modules_mlp_tokens_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_19_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_19_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_19_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_20_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_20_modules_mlp_tokens_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_20_modules_mlp_tokens_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_20_modules_mlp_tokens_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_20_modules_mlp_tokens_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_20_modules_mlp_tokens_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_20_modules_mlp_tokens_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_20_modules_mlp_tokens_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_20_modules_mlp_tokens_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_20_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_20_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_20_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_21_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_21_modules_mlp_tokens_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_21_modules_mlp_tokens_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_21_modules_mlp_tokens_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_21_modules_mlp_tokens_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_21_modules_mlp_tokens_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_21_modules_mlp_tokens_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_21_modules_mlp_tokens_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_21_modules_mlp_tokens_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_21_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_21_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_21_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_22_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_22_modules_mlp_tokens_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_22_modules_mlp_tokens_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_22_modules_mlp_tokens_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_22_modules_mlp_tokens_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_22_modules_mlp_tokens_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_22_modules_mlp_tokens_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_22_modules_mlp_tokens_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_22_modules_mlp_tokens_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_22_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_22_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_22_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_23_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_23_modules_mlp_tokens_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_23_modules_mlp_tokens_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_23_modules_mlp_tokens_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_23_modules_mlp_tokens_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_23_modules_mlp_tokens_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_23_modules_mlp_tokens_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_23_modules_mlp_tokens_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_23_modules_mlp_tokens_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_23_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_23_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_23_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_norm_parameters_weight_ = L_self_modules_norm_parameters_weight_
        l_self_modules_norm_parameters_bias_ = L_self_modules_norm_parameters_bias_
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
        layer_norm = torch.nn.functional.layer_norm(
            x_1,
            (1024,),
            l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_
        ) = None
        transpose_1 = layer_norm.transpose(1, 2)
        layer_norm = None
        x_2 = torch._C._nn.linear(
            transpose_1,
            l_self_modules_blocks_modules_0_modules_mlp_tokens_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_tokens_modules_fc1_parameters_bias_,
        )
        transpose_1 = l_self_modules_blocks_modules_0_modules_mlp_tokens_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_mlp_tokens_modules_fc1_parameters_bias_ = (None)
        x_3 = torch._C._nn.gelu(x_2, approximate="none")
        x_2 = None
        x_4 = torch.nn.functional.dropout(x_3, 0.0, False, False)
        x_3 = None
        x_5 = torch._C._nn.linear(
            x_4,
            l_self_modules_blocks_modules_0_modules_mlp_tokens_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_tokens_modules_fc2_parameters_bias_,
        )
        x_4 = l_self_modules_blocks_modules_0_modules_mlp_tokens_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_mlp_tokens_modules_fc2_parameters_bias_ = (None)
        x_6 = torch.nn.functional.dropout(x_5, 0.0, False, False)
        x_5 = None
        transpose_2 = x_6.transpose(1, 2)
        x_6 = None
        x_7 = x_1 + transpose_2
        x_1 = transpose_2 = None
        layer_norm_1 = torch.nn.functional.layer_norm(
            x_7,
            (1024,),
            l_self_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_norm2_parameters_bias_
        ) = None
        x_8 = torch._C._nn.linear(
            layer_norm_1,
            l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_1 = l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_9 = torch._C._nn.gelu(x_8, approximate="none")
        x_8 = None
        x_10 = torch.nn.functional.dropout(x_9, 0.0, False, False)
        x_9 = None
        x_11 = torch._C._nn.linear(
            x_10,
            l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_10 = l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_12 = torch.nn.functional.dropout(x_11, 0.0, False, False)
        x_11 = None
        x_13 = x_7 + x_12
        x_7 = x_12 = None
        layer_norm_2 = torch.nn.functional.layer_norm(
            x_13,
            (1024,),
            l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_
        ) = None
        transpose_3 = layer_norm_2.transpose(1, 2)
        layer_norm_2 = None
        x_14 = torch._C._nn.linear(
            transpose_3,
            l_self_modules_blocks_modules_1_modules_mlp_tokens_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_tokens_modules_fc1_parameters_bias_,
        )
        transpose_3 = l_self_modules_blocks_modules_1_modules_mlp_tokens_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_mlp_tokens_modules_fc1_parameters_bias_ = (None)
        x_15 = torch._C._nn.gelu(x_14, approximate="none")
        x_14 = None
        x_16 = torch.nn.functional.dropout(x_15, 0.0, False, False)
        x_15 = None
        x_17 = torch._C._nn.linear(
            x_16,
            l_self_modules_blocks_modules_1_modules_mlp_tokens_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_tokens_modules_fc2_parameters_bias_,
        )
        x_16 = l_self_modules_blocks_modules_1_modules_mlp_tokens_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_mlp_tokens_modules_fc2_parameters_bias_ = (None)
        x_18 = torch.nn.functional.dropout(x_17, 0.0, False, False)
        x_17 = None
        transpose_4 = x_18.transpose(1, 2)
        x_18 = None
        x_19 = x_13 + transpose_4
        x_13 = transpose_4 = None
        layer_norm_3 = torch.nn.functional.layer_norm(
            x_19,
            (1024,),
            l_self_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_norm2_parameters_bias_
        ) = None
        x_20 = torch._C._nn.linear(
            layer_norm_3,
            l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_3 = l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_21 = torch._C._nn.gelu(x_20, approximate="none")
        x_20 = None
        x_22 = torch.nn.functional.dropout(x_21, 0.0, False, False)
        x_21 = None
        x_23 = torch._C._nn.linear(
            x_22,
            l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_22 = l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_24 = torch.nn.functional.dropout(x_23, 0.0, False, False)
        x_23 = None
        x_25 = x_19 + x_24
        x_19 = x_24 = None
        layer_norm_4 = torch.nn.functional.layer_norm(
            x_25,
            (1024,),
            l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_
        ) = None
        transpose_5 = layer_norm_4.transpose(1, 2)
        layer_norm_4 = None
        x_26 = torch._C._nn.linear(
            transpose_5,
            l_self_modules_blocks_modules_2_modules_mlp_tokens_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_tokens_modules_fc1_parameters_bias_,
        )
        transpose_5 = l_self_modules_blocks_modules_2_modules_mlp_tokens_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_mlp_tokens_modules_fc1_parameters_bias_ = (None)
        x_27 = torch._C._nn.gelu(x_26, approximate="none")
        x_26 = None
        x_28 = torch.nn.functional.dropout(x_27, 0.0, False, False)
        x_27 = None
        x_29 = torch._C._nn.linear(
            x_28,
            l_self_modules_blocks_modules_2_modules_mlp_tokens_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_tokens_modules_fc2_parameters_bias_,
        )
        x_28 = l_self_modules_blocks_modules_2_modules_mlp_tokens_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_mlp_tokens_modules_fc2_parameters_bias_ = (None)
        x_30 = torch.nn.functional.dropout(x_29, 0.0, False, False)
        x_29 = None
        transpose_6 = x_30.transpose(1, 2)
        x_30 = None
        x_31 = x_25 + transpose_6
        x_25 = transpose_6 = None
        layer_norm_5 = torch.nn.functional.layer_norm(
            x_31,
            (1024,),
            l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_
        ) = None
        x_32 = torch._C._nn.linear(
            layer_norm_5,
            l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_5 = l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_33 = torch._C._nn.gelu(x_32, approximate="none")
        x_32 = None
        x_34 = torch.nn.functional.dropout(x_33, 0.0, False, False)
        x_33 = None
        x_35 = torch._C._nn.linear(
            x_34,
            l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_34 = l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_36 = torch.nn.functional.dropout(x_35, 0.0, False, False)
        x_35 = None
        x_37 = x_31 + x_36
        x_31 = x_36 = None
        layer_norm_6 = torch.nn.functional.layer_norm(
            x_37,
            (1024,),
            l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_
        ) = None
        transpose_7 = layer_norm_6.transpose(1, 2)
        layer_norm_6 = None
        x_38 = torch._C._nn.linear(
            transpose_7,
            l_self_modules_blocks_modules_3_modules_mlp_tokens_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_tokens_modules_fc1_parameters_bias_,
        )
        transpose_7 = l_self_modules_blocks_modules_3_modules_mlp_tokens_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_3_modules_mlp_tokens_modules_fc1_parameters_bias_ = (None)
        x_39 = torch._C._nn.gelu(x_38, approximate="none")
        x_38 = None
        x_40 = torch.nn.functional.dropout(x_39, 0.0, False, False)
        x_39 = None
        x_41 = torch._C._nn.linear(
            x_40,
            l_self_modules_blocks_modules_3_modules_mlp_tokens_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_tokens_modules_fc2_parameters_bias_,
        )
        x_40 = l_self_modules_blocks_modules_3_modules_mlp_tokens_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_3_modules_mlp_tokens_modules_fc2_parameters_bias_ = (None)
        x_42 = torch.nn.functional.dropout(x_41, 0.0, False, False)
        x_41 = None
        transpose_8 = x_42.transpose(1, 2)
        x_42 = None
        x_43 = x_37 + transpose_8
        x_37 = transpose_8 = None
        layer_norm_7 = torch.nn.functional.layer_norm(
            x_43,
            (1024,),
            l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_
        ) = None
        x_44 = torch._C._nn.linear(
            layer_norm_7,
            l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_7 = l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_45 = torch._C._nn.gelu(x_44, approximate="none")
        x_44 = None
        x_46 = torch.nn.functional.dropout(x_45, 0.0, False, False)
        x_45 = None
        x_47 = torch._C._nn.linear(
            x_46,
            l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_46 = l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_48 = torch.nn.functional.dropout(x_47, 0.0, False, False)
        x_47 = None
        x_49 = x_43 + x_48
        x_43 = x_48 = None
        layer_norm_8 = torch.nn.functional.layer_norm(
            x_49,
            (1024,),
            l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_
        ) = None
        transpose_9 = layer_norm_8.transpose(1, 2)
        layer_norm_8 = None
        x_50 = torch._C._nn.linear(
            transpose_9,
            l_self_modules_blocks_modules_4_modules_mlp_tokens_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_tokens_modules_fc1_parameters_bias_,
        )
        transpose_9 = l_self_modules_blocks_modules_4_modules_mlp_tokens_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_4_modules_mlp_tokens_modules_fc1_parameters_bias_ = (None)
        x_51 = torch._C._nn.gelu(x_50, approximate="none")
        x_50 = None
        x_52 = torch.nn.functional.dropout(x_51, 0.0, False, False)
        x_51 = None
        x_53 = torch._C._nn.linear(
            x_52,
            l_self_modules_blocks_modules_4_modules_mlp_tokens_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_tokens_modules_fc2_parameters_bias_,
        )
        x_52 = l_self_modules_blocks_modules_4_modules_mlp_tokens_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_4_modules_mlp_tokens_modules_fc2_parameters_bias_ = (None)
        x_54 = torch.nn.functional.dropout(x_53, 0.0, False, False)
        x_53 = None
        transpose_10 = x_54.transpose(1, 2)
        x_54 = None
        x_55 = x_49 + transpose_10
        x_49 = transpose_10 = None
        layer_norm_9 = torch.nn.functional.layer_norm(
            x_55,
            (1024,),
            l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_
        ) = None
        x_56 = torch._C._nn.linear(
            layer_norm_9,
            l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_9 = l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_57 = torch._C._nn.gelu(x_56, approximate="none")
        x_56 = None
        x_58 = torch.nn.functional.dropout(x_57, 0.0, False, False)
        x_57 = None
        x_59 = torch._C._nn.linear(
            x_58,
            l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_58 = l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_60 = torch.nn.functional.dropout(x_59, 0.0, False, False)
        x_59 = None
        x_61 = x_55 + x_60
        x_55 = x_60 = None
        layer_norm_10 = torch.nn.functional.layer_norm(
            x_61,
            (1024,),
            l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_
        ) = None
        transpose_11 = layer_norm_10.transpose(1, 2)
        layer_norm_10 = None
        x_62 = torch._C._nn.linear(
            transpose_11,
            l_self_modules_blocks_modules_5_modules_mlp_tokens_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_tokens_modules_fc1_parameters_bias_,
        )
        transpose_11 = l_self_modules_blocks_modules_5_modules_mlp_tokens_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_5_modules_mlp_tokens_modules_fc1_parameters_bias_ = (None)
        x_63 = torch._C._nn.gelu(x_62, approximate="none")
        x_62 = None
        x_64 = torch.nn.functional.dropout(x_63, 0.0, False, False)
        x_63 = None
        x_65 = torch._C._nn.linear(
            x_64,
            l_self_modules_blocks_modules_5_modules_mlp_tokens_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_tokens_modules_fc2_parameters_bias_,
        )
        x_64 = l_self_modules_blocks_modules_5_modules_mlp_tokens_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_5_modules_mlp_tokens_modules_fc2_parameters_bias_ = (None)
        x_66 = torch.nn.functional.dropout(x_65, 0.0, False, False)
        x_65 = None
        transpose_12 = x_66.transpose(1, 2)
        x_66 = None
        x_67 = x_61 + transpose_12
        x_61 = transpose_12 = None
        layer_norm_11 = torch.nn.functional.layer_norm(
            x_67,
            (1024,),
            l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_
        ) = None
        x_68 = torch._C._nn.linear(
            layer_norm_11,
            l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_11 = l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_69 = torch._C._nn.gelu(x_68, approximate="none")
        x_68 = None
        x_70 = torch.nn.functional.dropout(x_69, 0.0, False, False)
        x_69 = None
        x_71 = torch._C._nn.linear(
            x_70,
            l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_70 = l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_72 = torch.nn.functional.dropout(x_71, 0.0, False, False)
        x_71 = None
        x_73 = x_67 + x_72
        x_67 = x_72 = None
        layer_norm_12 = torch.nn.functional.layer_norm(
            x_73,
            (1024,),
            l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_
        ) = None
        transpose_13 = layer_norm_12.transpose(1, 2)
        layer_norm_12 = None
        x_74 = torch._C._nn.linear(
            transpose_13,
            l_self_modules_blocks_modules_6_modules_mlp_tokens_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_tokens_modules_fc1_parameters_bias_,
        )
        transpose_13 = l_self_modules_blocks_modules_6_modules_mlp_tokens_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_mlp_tokens_modules_fc1_parameters_bias_ = (None)
        x_75 = torch._C._nn.gelu(x_74, approximate="none")
        x_74 = None
        x_76 = torch.nn.functional.dropout(x_75, 0.0, False, False)
        x_75 = None
        x_77 = torch._C._nn.linear(
            x_76,
            l_self_modules_blocks_modules_6_modules_mlp_tokens_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_tokens_modules_fc2_parameters_bias_,
        )
        x_76 = l_self_modules_blocks_modules_6_modules_mlp_tokens_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_6_modules_mlp_tokens_modules_fc2_parameters_bias_ = (None)
        x_78 = torch.nn.functional.dropout(x_77, 0.0, False, False)
        x_77 = None
        transpose_14 = x_78.transpose(1, 2)
        x_78 = None
        x_79 = x_73 + transpose_14
        x_73 = transpose_14 = None
        layer_norm_13 = torch.nn.functional.layer_norm(
            x_79,
            (1024,),
            l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_
        ) = None
        x_80 = torch._C._nn.linear(
            layer_norm_13,
            l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_13 = l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_81 = torch._C._nn.gelu(x_80, approximate="none")
        x_80 = None
        x_82 = torch.nn.functional.dropout(x_81, 0.0, False, False)
        x_81 = None
        x_83 = torch._C._nn.linear(
            x_82,
            l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_82 = l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_84 = torch.nn.functional.dropout(x_83, 0.0, False, False)
        x_83 = None
        x_85 = x_79 + x_84
        x_79 = x_84 = None
        layer_norm_14 = torch.nn.functional.layer_norm(
            x_85,
            (1024,),
            l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_
        ) = None
        transpose_15 = layer_norm_14.transpose(1, 2)
        layer_norm_14 = None
        x_86 = torch._C._nn.linear(
            transpose_15,
            l_self_modules_blocks_modules_7_modules_mlp_tokens_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_tokens_modules_fc1_parameters_bias_,
        )
        transpose_15 = l_self_modules_blocks_modules_7_modules_mlp_tokens_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_7_modules_mlp_tokens_modules_fc1_parameters_bias_ = (None)
        x_87 = torch._C._nn.gelu(x_86, approximate="none")
        x_86 = None
        x_88 = torch.nn.functional.dropout(x_87, 0.0, False, False)
        x_87 = None
        x_89 = torch._C._nn.linear(
            x_88,
            l_self_modules_blocks_modules_7_modules_mlp_tokens_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_tokens_modules_fc2_parameters_bias_,
        )
        x_88 = l_self_modules_blocks_modules_7_modules_mlp_tokens_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_7_modules_mlp_tokens_modules_fc2_parameters_bias_ = (None)
        x_90 = torch.nn.functional.dropout(x_89, 0.0, False, False)
        x_89 = None
        transpose_16 = x_90.transpose(1, 2)
        x_90 = None
        x_91 = x_85 + transpose_16
        x_85 = transpose_16 = None
        layer_norm_15 = torch.nn.functional.layer_norm(
            x_91,
            (1024,),
            l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_
        ) = None
        x_92 = torch._C._nn.linear(
            layer_norm_15,
            l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_15 = l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_93 = torch._C._nn.gelu(x_92, approximate="none")
        x_92 = None
        x_94 = torch.nn.functional.dropout(x_93, 0.0, False, False)
        x_93 = None
        x_95 = torch._C._nn.linear(
            x_94,
            l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_94 = l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_96 = torch.nn.functional.dropout(x_95, 0.0, False, False)
        x_95 = None
        x_97 = x_91 + x_96
        x_91 = x_96 = None
        layer_norm_16 = torch.nn.functional.layer_norm(
            x_97,
            (1024,),
            l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_
        ) = None
        transpose_17 = layer_norm_16.transpose(1, 2)
        layer_norm_16 = None
        x_98 = torch._C._nn.linear(
            transpose_17,
            l_self_modules_blocks_modules_8_modules_mlp_tokens_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_tokens_modules_fc1_parameters_bias_,
        )
        transpose_17 = l_self_modules_blocks_modules_8_modules_mlp_tokens_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_mlp_tokens_modules_fc1_parameters_bias_ = (None)
        x_99 = torch._C._nn.gelu(x_98, approximate="none")
        x_98 = None
        x_100 = torch.nn.functional.dropout(x_99, 0.0, False, False)
        x_99 = None
        x_101 = torch._C._nn.linear(
            x_100,
            l_self_modules_blocks_modules_8_modules_mlp_tokens_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_tokens_modules_fc2_parameters_bias_,
        )
        x_100 = l_self_modules_blocks_modules_8_modules_mlp_tokens_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_8_modules_mlp_tokens_modules_fc2_parameters_bias_ = (None)
        x_102 = torch.nn.functional.dropout(x_101, 0.0, False, False)
        x_101 = None
        transpose_18 = x_102.transpose(1, 2)
        x_102 = None
        x_103 = x_97 + transpose_18
        x_97 = transpose_18 = None
        layer_norm_17 = torch.nn.functional.layer_norm(
            x_103,
            (1024,),
            l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_
        ) = None
        x_104 = torch._C._nn.linear(
            layer_norm_17,
            l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_17 = l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_105 = torch._C._nn.gelu(x_104, approximate="none")
        x_104 = None
        x_106 = torch.nn.functional.dropout(x_105, 0.0, False, False)
        x_105 = None
        x_107 = torch._C._nn.linear(
            x_106,
            l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_106 = l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_108 = torch.nn.functional.dropout(x_107, 0.0, False, False)
        x_107 = None
        x_109 = x_103 + x_108
        x_103 = x_108 = None
        layer_norm_18 = torch.nn.functional.layer_norm(
            x_109,
            (1024,),
            l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_
        ) = None
        transpose_19 = layer_norm_18.transpose(1, 2)
        layer_norm_18 = None
        x_110 = torch._C._nn.linear(
            transpose_19,
            l_self_modules_blocks_modules_9_modules_mlp_tokens_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_tokens_modules_fc1_parameters_bias_,
        )
        transpose_19 = l_self_modules_blocks_modules_9_modules_mlp_tokens_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_9_modules_mlp_tokens_modules_fc1_parameters_bias_ = (None)
        x_111 = torch._C._nn.gelu(x_110, approximate="none")
        x_110 = None
        x_112 = torch.nn.functional.dropout(x_111, 0.0, False, False)
        x_111 = None
        x_113 = torch._C._nn.linear(
            x_112,
            l_self_modules_blocks_modules_9_modules_mlp_tokens_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_tokens_modules_fc2_parameters_bias_,
        )
        x_112 = l_self_modules_blocks_modules_9_modules_mlp_tokens_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_9_modules_mlp_tokens_modules_fc2_parameters_bias_ = (None)
        x_114 = torch.nn.functional.dropout(x_113, 0.0, False, False)
        x_113 = None
        transpose_20 = x_114.transpose(1, 2)
        x_114 = None
        x_115 = x_109 + transpose_20
        x_109 = transpose_20 = None
        layer_norm_19 = torch.nn.functional.layer_norm(
            x_115,
            (1024,),
            l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_
        ) = None
        x_116 = torch._C._nn.linear(
            layer_norm_19,
            l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_19 = l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_117 = torch._C._nn.gelu(x_116, approximate="none")
        x_116 = None
        x_118 = torch.nn.functional.dropout(x_117, 0.0, False, False)
        x_117 = None
        x_119 = torch._C._nn.linear(
            x_118,
            l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_118 = l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_120 = torch.nn.functional.dropout(x_119, 0.0, False, False)
        x_119 = None
        x_121 = x_115 + x_120
        x_115 = x_120 = None
        layer_norm_20 = torch.nn.functional.layer_norm(
            x_121,
            (1024,),
            l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_
        ) = None
        transpose_21 = layer_norm_20.transpose(1, 2)
        layer_norm_20 = None
        x_122 = torch._C._nn.linear(
            transpose_21,
            l_self_modules_blocks_modules_10_modules_mlp_tokens_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_tokens_modules_fc1_parameters_bias_,
        )
        transpose_21 = l_self_modules_blocks_modules_10_modules_mlp_tokens_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_10_modules_mlp_tokens_modules_fc1_parameters_bias_ = (None)
        x_123 = torch._C._nn.gelu(x_122, approximate="none")
        x_122 = None
        x_124 = torch.nn.functional.dropout(x_123, 0.0, False, False)
        x_123 = None
        x_125 = torch._C._nn.linear(
            x_124,
            l_self_modules_blocks_modules_10_modules_mlp_tokens_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_tokens_modules_fc2_parameters_bias_,
        )
        x_124 = l_self_modules_blocks_modules_10_modules_mlp_tokens_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_10_modules_mlp_tokens_modules_fc2_parameters_bias_ = (None)
        x_126 = torch.nn.functional.dropout(x_125, 0.0, False, False)
        x_125 = None
        transpose_22 = x_126.transpose(1, 2)
        x_126 = None
        x_127 = x_121 + transpose_22
        x_121 = transpose_22 = None
        layer_norm_21 = torch.nn.functional.layer_norm(
            x_127,
            (1024,),
            l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_
        ) = None
        x_128 = torch._C._nn.linear(
            layer_norm_21,
            l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_21 = l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_129 = torch._C._nn.gelu(x_128, approximate="none")
        x_128 = None
        x_130 = torch.nn.functional.dropout(x_129, 0.0, False, False)
        x_129 = None
        x_131 = torch._C._nn.linear(
            x_130,
            l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_130 = l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_132 = torch.nn.functional.dropout(x_131, 0.0, False, False)
        x_131 = None
        x_133 = x_127 + x_132
        x_127 = x_132 = None
        layer_norm_22 = torch.nn.functional.layer_norm(
            x_133,
            (1024,),
            l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_
        ) = None
        transpose_23 = layer_norm_22.transpose(1, 2)
        layer_norm_22 = None
        x_134 = torch._C._nn.linear(
            transpose_23,
            l_self_modules_blocks_modules_11_modules_mlp_tokens_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_tokens_modules_fc1_parameters_bias_,
        )
        transpose_23 = l_self_modules_blocks_modules_11_modules_mlp_tokens_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_11_modules_mlp_tokens_modules_fc1_parameters_bias_ = (None)
        x_135 = torch._C._nn.gelu(x_134, approximate="none")
        x_134 = None
        x_136 = torch.nn.functional.dropout(x_135, 0.0, False, False)
        x_135 = None
        x_137 = torch._C._nn.linear(
            x_136,
            l_self_modules_blocks_modules_11_modules_mlp_tokens_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_tokens_modules_fc2_parameters_bias_,
        )
        x_136 = l_self_modules_blocks_modules_11_modules_mlp_tokens_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_11_modules_mlp_tokens_modules_fc2_parameters_bias_ = (None)
        x_138 = torch.nn.functional.dropout(x_137, 0.0, False, False)
        x_137 = None
        transpose_24 = x_138.transpose(1, 2)
        x_138 = None
        x_139 = x_133 + transpose_24
        x_133 = transpose_24 = None
        layer_norm_23 = torch.nn.functional.layer_norm(
            x_139,
            (1024,),
            l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_
        ) = None
        x_140 = torch._C._nn.linear(
            layer_norm_23,
            l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_23 = l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_141 = torch._C._nn.gelu(x_140, approximate="none")
        x_140 = None
        x_142 = torch.nn.functional.dropout(x_141, 0.0, False, False)
        x_141 = None
        x_143 = torch._C._nn.linear(
            x_142,
            l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_142 = l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_144 = torch.nn.functional.dropout(x_143, 0.0, False, False)
        x_143 = None
        x_145 = x_139 + x_144
        x_139 = x_144 = None
        layer_norm_24 = torch.nn.functional.layer_norm(
            x_145,
            (1024,),
            l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_
        ) = None
        transpose_25 = layer_norm_24.transpose(1, 2)
        layer_norm_24 = None
        x_146 = torch._C._nn.linear(
            transpose_25,
            l_self_modules_blocks_modules_12_modules_mlp_tokens_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_tokens_modules_fc1_parameters_bias_,
        )
        transpose_25 = l_self_modules_blocks_modules_12_modules_mlp_tokens_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_12_modules_mlp_tokens_modules_fc1_parameters_bias_ = (None)
        x_147 = torch._C._nn.gelu(x_146, approximate="none")
        x_146 = None
        x_148 = torch.nn.functional.dropout(x_147, 0.0, False, False)
        x_147 = None
        x_149 = torch._C._nn.linear(
            x_148,
            l_self_modules_blocks_modules_12_modules_mlp_tokens_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_tokens_modules_fc2_parameters_bias_,
        )
        x_148 = l_self_modules_blocks_modules_12_modules_mlp_tokens_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_12_modules_mlp_tokens_modules_fc2_parameters_bias_ = (None)
        x_150 = torch.nn.functional.dropout(x_149, 0.0, False, False)
        x_149 = None
        transpose_26 = x_150.transpose(1, 2)
        x_150 = None
        x_151 = x_145 + transpose_26
        x_145 = transpose_26 = None
        layer_norm_25 = torch.nn.functional.layer_norm(
            x_151,
            (1024,),
            l_self_modules_blocks_modules_12_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_12_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_norm2_parameters_bias_
        ) = None
        x_152 = torch._C._nn.linear(
            layer_norm_25,
            l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_25 = l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_153 = torch._C._nn.gelu(x_152, approximate="none")
        x_152 = None
        x_154 = torch.nn.functional.dropout(x_153, 0.0, False, False)
        x_153 = None
        x_155 = torch._C._nn.linear(
            x_154,
            l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_154 = l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_156 = torch.nn.functional.dropout(x_155, 0.0, False, False)
        x_155 = None
        x_157 = x_151 + x_156
        x_151 = x_156 = None
        layer_norm_26 = torch.nn.functional.layer_norm(
            x_157,
            (1024,),
            l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_
        ) = None
        transpose_27 = layer_norm_26.transpose(1, 2)
        layer_norm_26 = None
        x_158 = torch._C._nn.linear(
            transpose_27,
            l_self_modules_blocks_modules_13_modules_mlp_tokens_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_tokens_modules_fc1_parameters_bias_,
        )
        transpose_27 = l_self_modules_blocks_modules_13_modules_mlp_tokens_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_13_modules_mlp_tokens_modules_fc1_parameters_bias_ = (None)
        x_159 = torch._C._nn.gelu(x_158, approximate="none")
        x_158 = None
        x_160 = torch.nn.functional.dropout(x_159, 0.0, False, False)
        x_159 = None
        x_161 = torch._C._nn.linear(
            x_160,
            l_self_modules_blocks_modules_13_modules_mlp_tokens_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_tokens_modules_fc2_parameters_bias_,
        )
        x_160 = l_self_modules_blocks_modules_13_modules_mlp_tokens_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_13_modules_mlp_tokens_modules_fc2_parameters_bias_ = (None)
        x_162 = torch.nn.functional.dropout(x_161, 0.0, False, False)
        x_161 = None
        transpose_28 = x_162.transpose(1, 2)
        x_162 = None
        x_163 = x_157 + transpose_28
        x_157 = transpose_28 = None
        layer_norm_27 = torch.nn.functional.layer_norm(
            x_163,
            (1024,),
            l_self_modules_blocks_modules_13_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_13_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_norm2_parameters_bias_
        ) = None
        x_164 = torch._C._nn.linear(
            layer_norm_27,
            l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_27 = l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_165 = torch._C._nn.gelu(x_164, approximate="none")
        x_164 = None
        x_166 = torch.nn.functional.dropout(x_165, 0.0, False, False)
        x_165 = None
        x_167 = torch._C._nn.linear(
            x_166,
            l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_166 = l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_168 = torch.nn.functional.dropout(x_167, 0.0, False, False)
        x_167 = None
        x_169 = x_163 + x_168
        x_163 = x_168 = None
        layer_norm_28 = torch.nn.functional.layer_norm(
            x_169,
            (1024,),
            l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_
        ) = None
        transpose_29 = layer_norm_28.transpose(1, 2)
        layer_norm_28 = None
        x_170 = torch._C._nn.linear(
            transpose_29,
            l_self_modules_blocks_modules_14_modules_mlp_tokens_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_tokens_modules_fc1_parameters_bias_,
        )
        transpose_29 = l_self_modules_blocks_modules_14_modules_mlp_tokens_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_14_modules_mlp_tokens_modules_fc1_parameters_bias_ = (None)
        x_171 = torch._C._nn.gelu(x_170, approximate="none")
        x_170 = None
        x_172 = torch.nn.functional.dropout(x_171, 0.0, False, False)
        x_171 = None
        x_173 = torch._C._nn.linear(
            x_172,
            l_self_modules_blocks_modules_14_modules_mlp_tokens_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_tokens_modules_fc2_parameters_bias_,
        )
        x_172 = l_self_modules_blocks_modules_14_modules_mlp_tokens_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_14_modules_mlp_tokens_modules_fc2_parameters_bias_ = (None)
        x_174 = torch.nn.functional.dropout(x_173, 0.0, False, False)
        x_173 = None
        transpose_30 = x_174.transpose(1, 2)
        x_174 = None
        x_175 = x_169 + transpose_30
        x_169 = transpose_30 = None
        layer_norm_29 = torch.nn.functional.layer_norm(
            x_175,
            (1024,),
            l_self_modules_blocks_modules_14_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_14_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_norm2_parameters_bias_
        ) = None
        x_176 = torch._C._nn.linear(
            layer_norm_29,
            l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_29 = l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_177 = torch._C._nn.gelu(x_176, approximate="none")
        x_176 = None
        x_178 = torch.nn.functional.dropout(x_177, 0.0, False, False)
        x_177 = None
        x_179 = torch._C._nn.linear(
            x_178,
            l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_178 = l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_180 = torch.nn.functional.dropout(x_179, 0.0, False, False)
        x_179 = None
        x_181 = x_175 + x_180
        x_175 = x_180 = None
        layer_norm_30 = torch.nn.functional.layer_norm(
            x_181,
            (1024,),
            l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_
        ) = None
        transpose_31 = layer_norm_30.transpose(1, 2)
        layer_norm_30 = None
        x_182 = torch._C._nn.linear(
            transpose_31,
            l_self_modules_blocks_modules_15_modules_mlp_tokens_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_tokens_modules_fc1_parameters_bias_,
        )
        transpose_31 = l_self_modules_blocks_modules_15_modules_mlp_tokens_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_15_modules_mlp_tokens_modules_fc1_parameters_bias_ = (None)
        x_183 = torch._C._nn.gelu(x_182, approximate="none")
        x_182 = None
        x_184 = torch.nn.functional.dropout(x_183, 0.0, False, False)
        x_183 = None
        x_185 = torch._C._nn.linear(
            x_184,
            l_self_modules_blocks_modules_15_modules_mlp_tokens_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_tokens_modules_fc2_parameters_bias_,
        )
        x_184 = l_self_modules_blocks_modules_15_modules_mlp_tokens_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_15_modules_mlp_tokens_modules_fc2_parameters_bias_ = (None)
        x_186 = torch.nn.functional.dropout(x_185, 0.0, False, False)
        x_185 = None
        transpose_32 = x_186.transpose(1, 2)
        x_186 = None
        x_187 = x_181 + transpose_32
        x_181 = transpose_32 = None
        layer_norm_31 = torch.nn.functional.layer_norm(
            x_187,
            (1024,),
            l_self_modules_blocks_modules_15_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_15_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_norm2_parameters_bias_
        ) = None
        x_188 = torch._C._nn.linear(
            layer_norm_31,
            l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_31 = l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_189 = torch._C._nn.gelu(x_188, approximate="none")
        x_188 = None
        x_190 = torch.nn.functional.dropout(x_189, 0.0, False, False)
        x_189 = None
        x_191 = torch._C._nn.linear(
            x_190,
            l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_190 = l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_192 = torch.nn.functional.dropout(x_191, 0.0, False, False)
        x_191 = None
        x_193 = x_187 + x_192
        x_187 = x_192 = None
        layer_norm_32 = torch.nn.functional.layer_norm(
            x_193,
            (1024,),
            l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_
        ) = None
        transpose_33 = layer_norm_32.transpose(1, 2)
        layer_norm_32 = None
        x_194 = torch._C._nn.linear(
            transpose_33,
            l_self_modules_blocks_modules_16_modules_mlp_tokens_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_tokens_modules_fc1_parameters_bias_,
        )
        transpose_33 = l_self_modules_blocks_modules_16_modules_mlp_tokens_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_16_modules_mlp_tokens_modules_fc1_parameters_bias_ = (None)
        x_195 = torch._C._nn.gelu(x_194, approximate="none")
        x_194 = None
        x_196 = torch.nn.functional.dropout(x_195, 0.0, False, False)
        x_195 = None
        x_197 = torch._C._nn.linear(
            x_196,
            l_self_modules_blocks_modules_16_modules_mlp_tokens_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_tokens_modules_fc2_parameters_bias_,
        )
        x_196 = l_self_modules_blocks_modules_16_modules_mlp_tokens_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_16_modules_mlp_tokens_modules_fc2_parameters_bias_ = (None)
        x_198 = torch.nn.functional.dropout(x_197, 0.0, False, False)
        x_197 = None
        transpose_34 = x_198.transpose(1, 2)
        x_198 = None
        x_199 = x_193 + transpose_34
        x_193 = transpose_34 = None
        layer_norm_33 = torch.nn.functional.layer_norm(
            x_199,
            (1024,),
            l_self_modules_blocks_modules_16_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_16_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_norm2_parameters_bias_
        ) = None
        x_200 = torch._C._nn.linear(
            layer_norm_33,
            l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_33 = l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_201 = torch._C._nn.gelu(x_200, approximate="none")
        x_200 = None
        x_202 = torch.nn.functional.dropout(x_201, 0.0, False, False)
        x_201 = None
        x_203 = torch._C._nn.linear(
            x_202,
            l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_202 = l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_204 = torch.nn.functional.dropout(x_203, 0.0, False, False)
        x_203 = None
        x_205 = x_199 + x_204
        x_199 = x_204 = None
        layer_norm_34 = torch.nn.functional.layer_norm(
            x_205,
            (1024,),
            l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_
        ) = None
        transpose_35 = layer_norm_34.transpose(1, 2)
        layer_norm_34 = None
        x_206 = torch._C._nn.linear(
            transpose_35,
            l_self_modules_blocks_modules_17_modules_mlp_tokens_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_tokens_modules_fc1_parameters_bias_,
        )
        transpose_35 = l_self_modules_blocks_modules_17_modules_mlp_tokens_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_17_modules_mlp_tokens_modules_fc1_parameters_bias_ = (None)
        x_207 = torch._C._nn.gelu(x_206, approximate="none")
        x_206 = None
        x_208 = torch.nn.functional.dropout(x_207, 0.0, False, False)
        x_207 = None
        x_209 = torch._C._nn.linear(
            x_208,
            l_self_modules_blocks_modules_17_modules_mlp_tokens_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_tokens_modules_fc2_parameters_bias_,
        )
        x_208 = l_self_modules_blocks_modules_17_modules_mlp_tokens_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_17_modules_mlp_tokens_modules_fc2_parameters_bias_ = (None)
        x_210 = torch.nn.functional.dropout(x_209, 0.0, False, False)
        x_209 = None
        transpose_36 = x_210.transpose(1, 2)
        x_210 = None
        x_211 = x_205 + transpose_36
        x_205 = transpose_36 = None
        layer_norm_35 = torch.nn.functional.layer_norm(
            x_211,
            (1024,),
            l_self_modules_blocks_modules_17_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_17_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_norm2_parameters_bias_
        ) = None
        x_212 = torch._C._nn.linear(
            layer_norm_35,
            l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_35 = l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_213 = torch._C._nn.gelu(x_212, approximate="none")
        x_212 = None
        x_214 = torch.nn.functional.dropout(x_213, 0.0, False, False)
        x_213 = None
        x_215 = torch._C._nn.linear(
            x_214,
            l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_214 = l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_216 = torch.nn.functional.dropout(x_215, 0.0, False, False)
        x_215 = None
        x_217 = x_211 + x_216
        x_211 = x_216 = None
        layer_norm_36 = torch.nn.functional.layer_norm(
            x_217,
            (1024,),
            l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_
        ) = None
        transpose_37 = layer_norm_36.transpose(1, 2)
        layer_norm_36 = None
        x_218 = torch._C._nn.linear(
            transpose_37,
            l_self_modules_blocks_modules_18_modules_mlp_tokens_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_tokens_modules_fc1_parameters_bias_,
        )
        transpose_37 = l_self_modules_blocks_modules_18_modules_mlp_tokens_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_18_modules_mlp_tokens_modules_fc1_parameters_bias_ = (None)
        x_219 = torch._C._nn.gelu(x_218, approximate="none")
        x_218 = None
        x_220 = torch.nn.functional.dropout(x_219, 0.0, False, False)
        x_219 = None
        x_221 = torch._C._nn.linear(
            x_220,
            l_self_modules_blocks_modules_18_modules_mlp_tokens_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_tokens_modules_fc2_parameters_bias_,
        )
        x_220 = l_self_modules_blocks_modules_18_modules_mlp_tokens_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_18_modules_mlp_tokens_modules_fc2_parameters_bias_ = (None)
        x_222 = torch.nn.functional.dropout(x_221, 0.0, False, False)
        x_221 = None
        transpose_38 = x_222.transpose(1, 2)
        x_222 = None
        x_223 = x_217 + transpose_38
        x_217 = transpose_38 = None
        layer_norm_37 = torch.nn.functional.layer_norm(
            x_223,
            (1024,),
            l_self_modules_blocks_modules_18_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_18_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_norm2_parameters_bias_
        ) = None
        x_224 = torch._C._nn.linear(
            layer_norm_37,
            l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_37 = l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_225 = torch._C._nn.gelu(x_224, approximate="none")
        x_224 = None
        x_226 = torch.nn.functional.dropout(x_225, 0.0, False, False)
        x_225 = None
        x_227 = torch._C._nn.linear(
            x_226,
            l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_226 = l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_228 = torch.nn.functional.dropout(x_227, 0.0, False, False)
        x_227 = None
        x_229 = x_223 + x_228
        x_223 = x_228 = None
        layer_norm_38 = torch.nn.functional.layer_norm(
            x_229,
            (1024,),
            l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_
        ) = None
        transpose_39 = layer_norm_38.transpose(1, 2)
        layer_norm_38 = None
        x_230 = torch._C._nn.linear(
            transpose_39,
            l_self_modules_blocks_modules_19_modules_mlp_tokens_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_tokens_modules_fc1_parameters_bias_,
        )
        transpose_39 = l_self_modules_blocks_modules_19_modules_mlp_tokens_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_19_modules_mlp_tokens_modules_fc1_parameters_bias_ = (None)
        x_231 = torch._C._nn.gelu(x_230, approximate="none")
        x_230 = None
        x_232 = torch.nn.functional.dropout(x_231, 0.0, False, False)
        x_231 = None
        x_233 = torch._C._nn.linear(
            x_232,
            l_self_modules_blocks_modules_19_modules_mlp_tokens_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_tokens_modules_fc2_parameters_bias_,
        )
        x_232 = l_self_modules_blocks_modules_19_modules_mlp_tokens_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_19_modules_mlp_tokens_modules_fc2_parameters_bias_ = (None)
        x_234 = torch.nn.functional.dropout(x_233, 0.0, False, False)
        x_233 = None
        transpose_40 = x_234.transpose(1, 2)
        x_234 = None
        x_235 = x_229 + transpose_40
        x_229 = transpose_40 = None
        layer_norm_39 = torch.nn.functional.layer_norm(
            x_235,
            (1024,),
            l_self_modules_blocks_modules_19_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_19_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_norm2_parameters_bias_
        ) = None
        x_236 = torch._C._nn.linear(
            layer_norm_39,
            l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_39 = l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_237 = torch._C._nn.gelu(x_236, approximate="none")
        x_236 = None
        x_238 = torch.nn.functional.dropout(x_237, 0.0, False, False)
        x_237 = None
        x_239 = torch._C._nn.linear(
            x_238,
            l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_238 = l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_240 = torch.nn.functional.dropout(x_239, 0.0, False, False)
        x_239 = None
        x_241 = x_235 + x_240
        x_235 = x_240 = None
        layer_norm_40 = torch.nn.functional.layer_norm(
            x_241,
            (1024,),
            l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_
        ) = None
        transpose_41 = layer_norm_40.transpose(1, 2)
        layer_norm_40 = None
        x_242 = torch._C._nn.linear(
            transpose_41,
            l_self_modules_blocks_modules_20_modules_mlp_tokens_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_tokens_modules_fc1_parameters_bias_,
        )
        transpose_41 = l_self_modules_blocks_modules_20_modules_mlp_tokens_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_20_modules_mlp_tokens_modules_fc1_parameters_bias_ = (None)
        x_243 = torch._C._nn.gelu(x_242, approximate="none")
        x_242 = None
        x_244 = torch.nn.functional.dropout(x_243, 0.0, False, False)
        x_243 = None
        x_245 = torch._C._nn.linear(
            x_244,
            l_self_modules_blocks_modules_20_modules_mlp_tokens_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_tokens_modules_fc2_parameters_bias_,
        )
        x_244 = l_self_modules_blocks_modules_20_modules_mlp_tokens_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_20_modules_mlp_tokens_modules_fc2_parameters_bias_ = (None)
        x_246 = torch.nn.functional.dropout(x_245, 0.0, False, False)
        x_245 = None
        transpose_42 = x_246.transpose(1, 2)
        x_246 = None
        x_247 = x_241 + transpose_42
        x_241 = transpose_42 = None
        layer_norm_41 = torch.nn.functional.layer_norm(
            x_247,
            (1024,),
            l_self_modules_blocks_modules_20_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_20_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_norm2_parameters_bias_
        ) = None
        x_248 = torch._C._nn.linear(
            layer_norm_41,
            l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_41 = l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_249 = torch._C._nn.gelu(x_248, approximate="none")
        x_248 = None
        x_250 = torch.nn.functional.dropout(x_249, 0.0, False, False)
        x_249 = None
        x_251 = torch._C._nn.linear(
            x_250,
            l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_250 = l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_252 = torch.nn.functional.dropout(x_251, 0.0, False, False)
        x_251 = None
        x_253 = x_247 + x_252
        x_247 = x_252 = None
        layer_norm_42 = torch.nn.functional.layer_norm(
            x_253,
            (1024,),
            l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_
        ) = None
        transpose_43 = layer_norm_42.transpose(1, 2)
        layer_norm_42 = None
        x_254 = torch._C._nn.linear(
            transpose_43,
            l_self_modules_blocks_modules_21_modules_mlp_tokens_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_tokens_modules_fc1_parameters_bias_,
        )
        transpose_43 = l_self_modules_blocks_modules_21_modules_mlp_tokens_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_21_modules_mlp_tokens_modules_fc1_parameters_bias_ = (None)
        x_255 = torch._C._nn.gelu(x_254, approximate="none")
        x_254 = None
        x_256 = torch.nn.functional.dropout(x_255, 0.0, False, False)
        x_255 = None
        x_257 = torch._C._nn.linear(
            x_256,
            l_self_modules_blocks_modules_21_modules_mlp_tokens_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_tokens_modules_fc2_parameters_bias_,
        )
        x_256 = l_self_modules_blocks_modules_21_modules_mlp_tokens_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_21_modules_mlp_tokens_modules_fc2_parameters_bias_ = (None)
        x_258 = torch.nn.functional.dropout(x_257, 0.0, False, False)
        x_257 = None
        transpose_44 = x_258.transpose(1, 2)
        x_258 = None
        x_259 = x_253 + transpose_44
        x_253 = transpose_44 = None
        layer_norm_43 = torch.nn.functional.layer_norm(
            x_259,
            (1024,),
            l_self_modules_blocks_modules_21_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_21_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_norm2_parameters_bias_
        ) = None
        x_260 = torch._C._nn.linear(
            layer_norm_43,
            l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_43 = l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_261 = torch._C._nn.gelu(x_260, approximate="none")
        x_260 = None
        x_262 = torch.nn.functional.dropout(x_261, 0.0, False, False)
        x_261 = None
        x_263 = torch._C._nn.linear(
            x_262,
            l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_262 = l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_264 = torch.nn.functional.dropout(x_263, 0.0, False, False)
        x_263 = None
        x_265 = x_259 + x_264
        x_259 = x_264 = None
        layer_norm_44 = torch.nn.functional.layer_norm(
            x_265,
            (1024,),
            l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_
        ) = None
        transpose_45 = layer_norm_44.transpose(1, 2)
        layer_norm_44 = None
        x_266 = torch._C._nn.linear(
            transpose_45,
            l_self_modules_blocks_modules_22_modules_mlp_tokens_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_tokens_modules_fc1_parameters_bias_,
        )
        transpose_45 = l_self_modules_blocks_modules_22_modules_mlp_tokens_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_22_modules_mlp_tokens_modules_fc1_parameters_bias_ = (None)
        x_267 = torch._C._nn.gelu(x_266, approximate="none")
        x_266 = None
        x_268 = torch.nn.functional.dropout(x_267, 0.0, False, False)
        x_267 = None
        x_269 = torch._C._nn.linear(
            x_268,
            l_self_modules_blocks_modules_22_modules_mlp_tokens_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_tokens_modules_fc2_parameters_bias_,
        )
        x_268 = l_self_modules_blocks_modules_22_modules_mlp_tokens_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_22_modules_mlp_tokens_modules_fc2_parameters_bias_ = (None)
        x_270 = torch.nn.functional.dropout(x_269, 0.0, False, False)
        x_269 = None
        transpose_46 = x_270.transpose(1, 2)
        x_270 = None
        x_271 = x_265 + transpose_46
        x_265 = transpose_46 = None
        layer_norm_45 = torch.nn.functional.layer_norm(
            x_271,
            (1024,),
            l_self_modules_blocks_modules_22_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_22_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_norm2_parameters_bias_
        ) = None
        x_272 = torch._C._nn.linear(
            layer_norm_45,
            l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_45 = l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_273 = torch._C._nn.gelu(x_272, approximate="none")
        x_272 = None
        x_274 = torch.nn.functional.dropout(x_273, 0.0, False, False)
        x_273 = None
        x_275 = torch._C._nn.linear(
            x_274,
            l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_274 = l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_276 = torch.nn.functional.dropout(x_275, 0.0, False, False)
        x_275 = None
        x_277 = x_271 + x_276
        x_271 = x_276 = None
        layer_norm_46 = torch.nn.functional.layer_norm(
            x_277,
            (1024,),
            l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_
        ) = None
        transpose_47 = layer_norm_46.transpose(1, 2)
        layer_norm_46 = None
        x_278 = torch._C._nn.linear(
            transpose_47,
            l_self_modules_blocks_modules_23_modules_mlp_tokens_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_tokens_modules_fc1_parameters_bias_,
        )
        transpose_47 = l_self_modules_blocks_modules_23_modules_mlp_tokens_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_23_modules_mlp_tokens_modules_fc1_parameters_bias_ = (None)
        x_279 = torch._C._nn.gelu(x_278, approximate="none")
        x_278 = None
        x_280 = torch.nn.functional.dropout(x_279, 0.0, False, False)
        x_279 = None
        x_281 = torch._C._nn.linear(
            x_280,
            l_self_modules_blocks_modules_23_modules_mlp_tokens_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_tokens_modules_fc2_parameters_bias_,
        )
        x_280 = l_self_modules_blocks_modules_23_modules_mlp_tokens_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_23_modules_mlp_tokens_modules_fc2_parameters_bias_ = (None)
        x_282 = torch.nn.functional.dropout(x_281, 0.0, False, False)
        x_281 = None
        transpose_48 = x_282.transpose(1, 2)
        x_282 = None
        x_283 = x_277 + transpose_48
        x_277 = transpose_48 = None
        layer_norm_47 = torch.nn.functional.layer_norm(
            x_283,
            (1024,),
            l_self_modules_blocks_modules_23_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_23_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_norm2_parameters_bias_
        ) = None
        x_284 = torch._C._nn.linear(
            layer_norm_47,
            l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_47 = l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_285 = torch._C._nn.gelu(x_284, approximate="none")
        x_284 = None
        x_286 = torch.nn.functional.dropout(x_285, 0.0, False, False)
        x_285 = None
        x_287 = torch._C._nn.linear(
            x_286,
            l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_286 = l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_288 = torch.nn.functional.dropout(x_287, 0.0, False, False)
        x_287 = None
        x_289 = x_283 + x_288
        x_283 = x_288 = None
        x_290 = torch.nn.functional.layer_norm(
            x_289,
            (1024,),
            l_self_modules_norm_parameters_weight_,
            l_self_modules_norm_parameters_bias_,
            1e-06,
        )
        x_289 = (
            l_self_modules_norm_parameters_weight_
        ) = l_self_modules_norm_parameters_bias_ = None
        x_291 = x_290.mean(dim=1)
        x_290 = None
        x_292 = torch.nn.functional.dropout(x_291, 0.0, False, False)
        x_291 = None
        x_293 = torch._C._nn.linear(
            x_292,
            l_self_modules_head_parameters_weight_,
            l_self_modules_head_parameters_bias_,
        )
        x_292 = (
            l_self_modules_head_parameters_weight_
        ) = l_self_modules_head_parameters_bias_ = None
        return (x_293,)
