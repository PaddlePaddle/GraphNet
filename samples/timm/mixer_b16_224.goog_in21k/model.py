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
            (768,),
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
            (768,),
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
            (768,),
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
            (768,),
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
            (768,),
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
            (768,),
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
            (768,),
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
            (768,),
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
            (768,),
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
            (768,),
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
            (768,),
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
            (768,),
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
            (768,),
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
            (768,),
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
            (768,),
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
            (768,),
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
            (768,),
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
            (768,),
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
            (768,),
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
            (768,),
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
            (768,),
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
            (768,),
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
            (768,),
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
            (768,),
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
        x_146 = torch.nn.functional.layer_norm(
            x_145,
            (768,),
            l_self_modules_norm_parameters_weight_,
            l_self_modules_norm_parameters_bias_,
            1e-06,
        )
        x_145 = (
            l_self_modules_norm_parameters_weight_
        ) = l_self_modules_norm_parameters_bias_ = None
        x_147 = x_146.mean(dim=1)
        x_146 = None
        x_148 = torch.nn.functional.dropout(x_147, 0.0, False, False)
        x_147 = None
        x_149 = torch._C._nn.linear(
            x_148,
            l_self_modules_head_parameters_weight_,
            l_self_modules_head_parameters_bias_,
        )
        x_148 = (
            l_self_modules_head_parameters_weight_
        ) = l_self_modules_head_parameters_bias_ = None
        return (x_149,)
