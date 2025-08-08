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
        x_86 = torch.addcmul(
            l_self_modules_norm_parameters_beta_,
            l_self_modules_norm_parameters_alpha_,
            x_85,
        )
        l_self_modules_norm_parameters_beta_ = (
            l_self_modules_norm_parameters_alpha_
        ) = x_85 = None
        x_87 = x_86.mean(dim=1)
        x_86 = None
        x_88 = torch.nn.functional.dropout(x_87, 0.0, False, False)
        x_87 = None
        x_89 = torch._C._nn.linear(
            x_88,
            l_self_modules_head_parameters_weight_,
            l_self_modules_head_parameters_bias_,
        )
        x_88 = (
            l_self_modules_head_parameters_weight_
        ) = l_self_modules_head_parameters_bias_ = None
        return (x_89,)
