import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_0_parameters_weight_ = (
            L_self_modules_stem_modules_0_parameters_weight_
        )
        l_self_modules_stem_modules_0_parameters_bias_ = (
            L_self_modules_stem_modules_0_parameters_bias_
        )
        l_x_ = L_x_
        l_self_modules_stem_modules_1_parameters_weight_ = (
            L_self_modules_stem_modules_1_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_parameters_gamma_ = (
            L_self_modules_stages_modules_0_modules_blocks_modules_0_parameters_gamma_
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_parameters_gamma_ = (
            L_self_modules_stages_modules_0_modules_blocks_modules_1_parameters_gamma_
        )
        l_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_weight_ = L_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_weight_
        l_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_weight_ = L_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_weight_
        l_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_bias_ = L_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_parameters_gamma_ = (
            L_self_modules_stages_modules_1_modules_blocks_modules_0_parameters_gamma_
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_parameters_gamma_ = (
            L_self_modules_stages_modules_1_modules_blocks_modules_1_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_weight_ = L_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_weight_ = L_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_bias_ = L_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_0_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_1_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_2_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_3_parameters_gamma_
        )
        l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_weight_ = L_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_weight_
        l_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_weight_ = L_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_weight_
        l_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_bias_ = L_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_parameters_gamma_ = (
            L_self_modules_stages_modules_3_modules_blocks_modules_0_parameters_gamma_
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_parameters_gamma_ = (
            L_self_modules_stages_modules_3_modules_blocks_modules_1_parameters_gamma_
        )
        l_self_modules_head_modules_norm_parameters_weight_ = (
            L_self_modules_head_modules_norm_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_fc_parameters_bias_
        )
        input_1 = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_0_parameters_weight_,
            l_self_modules_stem_modules_0_parameters_bias_,
            (4, 4),
            (0, 0),
            (1, 1),
            1,
        )
        l_x_ = (
            l_self_modules_stem_modules_0_parameters_weight_
        ) = l_self_modules_stem_modules_0_parameters_bias_ = None
        x = input_1.permute(0, 2, 3, 1)
        input_1 = None
        v = torch.var(x, dim=(-1,), keepdim=True)
        add = v + 1e-06
        v = None
        rsqrt = torch.rsqrt(add)
        add = None
        x_1 = x * rsqrt
        x = rsqrt = None
        x_2 = x_1 * l_self_modules_stem_modules_1_parameters_weight_
        x_1 = l_self_modules_stem_modules_1_parameters_weight_ = None
        x_3 = x_2.permute(0, 3, 1, 2)
        x_2 = None
        x_4 = torch.conv2d(
            x_3,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            32,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_bias_ = (None)
        x_5 = x_4.permute(0, 2, 3, 1)
        x_4 = None
        v_1 = torch.var(x_5, dim=(-1,), keepdim=True)
        add_1 = v_1 + 1e-06
        v_1 = None
        rsqrt_1 = torch.rsqrt(add_1)
        add_1 = None
        x_6 = x_5 * rsqrt_1
        x_5 = rsqrt_1 = None
        x_7 = (
            x_6
            * l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_
        )
        x_6 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_ = (None)
        x_8 = x_7.permute(0, 3, 1, 2)
        x_7 = None
        x_9 = torch.conv2d(
            x_8,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_8 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_10 = torch._C._nn.gelu(x_9)
        x_9 = None
        x_11 = torch.nn.functional.dropout(x_10, 0.0, False, False)
        x_10 = None
        x_12 = torch.conv2d(
            x_11,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_11 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_13 = torch.nn.functional.dropout(x_12, 0.0, False, False)
        x_12 = None
        reshape = l_self_modules_stages_modules_0_modules_blocks_modules_0_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_parameters_gamma_ = (
            None
        )
        x_14 = x_13.mul(reshape)
        x_13 = reshape = None
        x_15 = x_14 + x_3
        x_14 = x_3 = None
        x_16 = torch.conv2d(
            x_15,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            32,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_bias_ = (None)
        x_17 = x_16.permute(0, 2, 3, 1)
        x_16 = None
        v_2 = torch.var(x_17, dim=(-1,), keepdim=True)
        add_3 = v_2 + 1e-06
        v_2 = None
        rsqrt_2 = torch.rsqrt(add_3)
        add_3 = None
        x_18 = x_17 * rsqrt_2
        x_17 = rsqrt_2 = None
        x_19 = (
            x_18
            * l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_
        )
        x_18 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_ = (None)
        x_20 = x_19.permute(0, 3, 1, 2)
        x_19 = None
        x_21 = torch.conv2d(
            x_20,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_20 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_22 = torch._C._nn.gelu(x_21)
        x_21 = None
        x_23 = torch.nn.functional.dropout(x_22, 0.0, False, False)
        x_22 = None
        x_24 = torch.conv2d(
            x_23,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_23 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_25 = torch.nn.functional.dropout(x_24, 0.0, False, False)
        x_24 = None
        reshape_1 = l_self_modules_stages_modules_0_modules_blocks_modules_1_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_parameters_gamma_ = (
            None
        )
        x_26 = x_25.mul(reshape_1)
        x_25 = reshape_1 = None
        x_27 = x_26 + x_15
        x_26 = x_15 = None
        x_28 = x_27.permute(0, 2, 3, 1)
        x_27 = None
        v_3 = torch.var(x_28, dim=(-1,), keepdim=True)
        add_5 = v_3 + 1e-06
        v_3 = None
        rsqrt_3 = torch.rsqrt(add_5)
        add_5 = None
        x_29 = x_28 * rsqrt_3
        x_28 = rsqrt_3 = None
        x_30 = (
            x_29
            * l_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_weight_
        )
        x_29 = l_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_weight_ = (None)
        x_31 = x_30.permute(0, 3, 1, 2)
        x_30 = None
        input_2 = torch.conv2d(
            x_31,
            l_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_31 = l_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_weight_ = l_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_bias_ = (None)
        x_32 = torch.conv2d(
            input_2,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            64,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_bias_ = (None)
        x_33 = x_32.permute(0, 2, 3, 1)
        x_32 = None
        v_4 = torch.var(x_33, dim=(-1,), keepdim=True)
        add_6 = v_4 + 1e-06
        v_4 = None
        rsqrt_4 = torch.rsqrt(add_6)
        add_6 = None
        x_34 = x_33 * rsqrt_4
        x_33 = rsqrt_4 = None
        x_35 = (
            x_34
            * l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_
        )
        x_34 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_ = (None)
        x_36 = x_35.permute(0, 3, 1, 2)
        x_35 = None
        x_37 = torch.conv2d(
            x_36,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_36 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_38 = torch._C._nn.gelu(x_37)
        x_37 = None
        x_39 = torch.nn.functional.dropout(x_38, 0.0, False, False)
        x_38 = None
        x_40 = torch.conv2d(
            x_39,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_39 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_41 = torch.nn.functional.dropout(x_40, 0.0, False, False)
        x_40 = None
        reshape_2 = l_self_modules_stages_modules_1_modules_blocks_modules_0_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_parameters_gamma_ = (
            None
        )
        x_42 = x_41.mul(reshape_2)
        x_41 = reshape_2 = None
        x_43 = x_42 + input_2
        x_42 = input_2 = None
        x_44 = torch.conv2d(
            x_43,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            64,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_bias_ = (None)
        x_45 = x_44.permute(0, 2, 3, 1)
        x_44 = None
        v_5 = torch.var(x_45, dim=(-1,), keepdim=True)
        add_8 = v_5 + 1e-06
        v_5 = None
        rsqrt_5 = torch.rsqrt(add_8)
        add_8 = None
        x_46 = x_45 * rsqrt_5
        x_45 = rsqrt_5 = None
        x_47 = (
            x_46
            * l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_
        )
        x_46 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_ = (None)
        x_48 = x_47.permute(0, 3, 1, 2)
        x_47 = None
        x_49 = torch.conv2d(
            x_48,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_48 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_50 = torch._C._nn.gelu(x_49)
        x_49 = None
        x_51 = torch.nn.functional.dropout(x_50, 0.0, False, False)
        x_50 = None
        x_52 = torch.conv2d(
            x_51,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_51 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_53 = torch.nn.functional.dropout(x_52, 0.0, False, False)
        x_52 = None
        reshape_3 = l_self_modules_stages_modules_1_modules_blocks_modules_1_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_parameters_gamma_ = (
            None
        )
        x_54 = x_53.mul(reshape_3)
        x_53 = reshape_3 = None
        x_55 = x_54 + x_43
        x_54 = x_43 = None
        x_56 = x_55.permute(0, 2, 3, 1)
        x_55 = None
        v_6 = torch.var(x_56, dim=(-1,), keepdim=True)
        add_10 = v_6 + 1e-06
        v_6 = None
        rsqrt_6 = torch.rsqrt(add_10)
        add_10 = None
        x_57 = x_56 * rsqrt_6
        x_56 = rsqrt_6 = None
        x_58 = (
            x_57
            * l_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_weight_
        )
        x_57 = l_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_weight_ = (None)
        x_59 = x_58.permute(0, 3, 1, 2)
        x_58 = None
        input_3 = torch.conv2d(
            x_59,
            l_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_59 = l_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_weight_ = l_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_bias_ = (None)
        x_60 = torch.conv2d(
            input_3,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_bias_ = (None)
        x_61 = x_60.permute(0, 2, 3, 1)
        x_60 = None
        v_7 = torch.var(x_61, dim=(-1,), keepdim=True)
        add_11 = v_7 + 1e-06
        v_7 = None
        rsqrt_7 = torch.rsqrt(add_11)
        add_11 = None
        x_62 = x_61 * rsqrt_7
        x_61 = rsqrt_7 = None
        x_63 = (
            x_62
            * l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_
        )
        x_62 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_ = (None)
        x_64 = x_63.permute(0, 3, 1, 2)
        x_63 = None
        x_65 = torch.conv2d(
            x_64,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_64 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_66 = torch._C._nn.gelu(x_65)
        x_65 = None
        x_67 = torch.nn.functional.dropout(x_66, 0.0, False, False)
        x_66 = None
        x_68 = torch.conv2d(
            x_67,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_67 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_69 = torch.nn.functional.dropout(x_68, 0.0, False, False)
        x_68 = None
        reshape_4 = l_self_modules_stages_modules_2_modules_blocks_modules_0_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_parameters_gamma_ = (
            None
        )
        x_70 = x_69.mul(reshape_4)
        x_69 = reshape_4 = None
        x_71 = x_70 + input_3
        x_70 = input_3 = None
        x_72 = torch.conv2d(
            x_71,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_bias_ = (None)
        x_73 = x_72.permute(0, 2, 3, 1)
        x_72 = None
        v_8 = torch.var(x_73, dim=(-1,), keepdim=True)
        add_13 = v_8 + 1e-06
        v_8 = None
        rsqrt_8 = torch.rsqrt(add_13)
        add_13 = None
        x_74 = x_73 * rsqrt_8
        x_73 = rsqrt_8 = None
        x_75 = (
            x_74
            * l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_
        )
        x_74 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_ = (None)
        x_76 = x_75.permute(0, 3, 1, 2)
        x_75 = None
        x_77 = torch.conv2d(
            x_76,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_76 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_78 = torch._C._nn.gelu(x_77)
        x_77 = None
        x_79 = torch.nn.functional.dropout(x_78, 0.0, False, False)
        x_78 = None
        x_80 = torch.conv2d(
            x_79,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_79 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_81 = torch.nn.functional.dropout(x_80, 0.0, False, False)
        x_80 = None
        reshape_5 = l_self_modules_stages_modules_2_modules_blocks_modules_1_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_parameters_gamma_ = (
            None
        )
        x_82 = x_81.mul(reshape_5)
        x_81 = reshape_5 = None
        x_83 = x_82 + x_71
        x_82 = x_71 = None
        x_84 = torch.conv2d(
            x_83,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_bias_ = (None)
        x_85 = x_84.permute(0, 2, 3, 1)
        x_84 = None
        v_9 = torch.var(x_85, dim=(-1,), keepdim=True)
        add_15 = v_9 + 1e-06
        v_9 = None
        rsqrt_9 = torch.rsqrt(add_15)
        add_15 = None
        x_86 = x_85 * rsqrt_9
        x_85 = rsqrt_9 = None
        x_87 = (
            x_86
            * l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_
        )
        x_86 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_ = (None)
        x_88 = x_87.permute(0, 3, 1, 2)
        x_87 = None
        x_89 = torch.conv2d(
            x_88,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_88 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_90 = torch._C._nn.gelu(x_89)
        x_89 = None
        x_91 = torch.nn.functional.dropout(x_90, 0.0, False, False)
        x_90 = None
        x_92 = torch.conv2d(
            x_91,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_91 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_93 = torch.nn.functional.dropout(x_92, 0.0, False, False)
        x_92 = None
        reshape_6 = l_self_modules_stages_modules_2_modules_blocks_modules_2_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_parameters_gamma_ = (
            None
        )
        x_94 = x_93.mul(reshape_6)
        x_93 = reshape_6 = None
        x_95 = x_94 + x_83
        x_94 = x_83 = None
        x_96 = torch.conv2d(
            x_95,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_bias_ = (None)
        x_97 = x_96.permute(0, 2, 3, 1)
        x_96 = None
        v_10 = torch.var(x_97, dim=(-1,), keepdim=True)
        add_17 = v_10 + 1e-06
        v_10 = None
        rsqrt_10 = torch.rsqrt(add_17)
        add_17 = None
        x_98 = x_97 * rsqrt_10
        x_97 = rsqrt_10 = None
        x_99 = (
            x_98
            * l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_
        )
        x_98 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_ = (None)
        x_100 = x_99.permute(0, 3, 1, 2)
        x_99 = None
        x_101 = torch.conv2d(
            x_100,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_100 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_102 = torch._C._nn.gelu(x_101)
        x_101 = None
        x_103 = torch.nn.functional.dropout(x_102, 0.0, False, False)
        x_102 = None
        x_104 = torch.conv2d(
            x_103,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_103 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_105 = torch.nn.functional.dropout(x_104, 0.0, False, False)
        x_104 = None
        reshape_7 = l_self_modules_stages_modules_2_modules_blocks_modules_3_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_parameters_gamma_ = (
            None
        )
        x_106 = x_105.mul(reshape_7)
        x_105 = reshape_7 = None
        x_107 = x_106 + x_95
        x_106 = x_95 = None
        x_108 = x_107.permute(0, 2, 3, 1)
        x_107 = None
        v_11 = torch.var(x_108, dim=(-1,), keepdim=True)
        add_19 = v_11 + 1e-06
        v_11 = None
        rsqrt_11 = torch.rsqrt(add_19)
        add_19 = None
        x_109 = x_108 * rsqrt_11
        x_108 = rsqrt_11 = None
        x_110 = (
            x_109
            * l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_weight_
        )
        x_109 = l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_weight_ = (None)
        x_111 = x_110.permute(0, 3, 1, 2)
        x_110 = None
        input_4 = torch.conv2d(
            x_111,
            l_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_111 = l_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_bias_ = (None)
        x_112 = torch.conv2d(
            input_4,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_bias_ = (None)
        x_113 = x_112.permute(0, 2, 3, 1)
        x_112 = None
        v_12 = torch.var(x_113, dim=(-1,), keepdim=True)
        add_20 = v_12 + 1e-06
        v_12 = None
        rsqrt_12 = torch.rsqrt(add_20)
        add_20 = None
        x_114 = x_113 * rsqrt_12
        x_113 = rsqrt_12 = None
        x_115 = (
            x_114
            * l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_
        )
        x_114 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_ = (None)
        x_116 = x_115.permute(0, 3, 1, 2)
        x_115 = None
        x_117 = torch.conv2d(
            x_116,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_116 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_118 = torch._C._nn.gelu(x_117)
        x_117 = None
        x_119 = torch.nn.functional.dropout(x_118, 0.0, False, False)
        x_118 = None
        x_120 = torch.conv2d(
            x_119,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_119 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_121 = torch.nn.functional.dropout(x_120, 0.0, False, False)
        x_120 = None
        reshape_8 = l_self_modules_stages_modules_3_modules_blocks_modules_0_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_parameters_gamma_ = (
            None
        )
        x_122 = x_121.mul(reshape_8)
        x_121 = reshape_8 = None
        x_123 = x_122 + input_4
        x_122 = input_4 = None
        x_124 = torch.conv2d(
            x_123,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_bias_ = (None)
        x_125 = x_124.permute(0, 2, 3, 1)
        x_124 = None
        v_13 = torch.var(x_125, dim=(-1,), keepdim=True)
        add_22 = v_13 + 1e-06
        v_13 = None
        rsqrt_13 = torch.rsqrt(add_22)
        add_22 = None
        x_126 = x_125 * rsqrt_13
        x_125 = rsqrt_13 = None
        x_127 = (
            x_126
            * l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_
        )
        x_126 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_ = (None)
        x_128 = x_127.permute(0, 3, 1, 2)
        x_127 = None
        x_129 = torch.conv2d(
            x_128,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_128 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_130 = torch._C._nn.gelu(x_129)
        x_129 = None
        x_131 = torch.nn.functional.dropout(x_130, 0.0, False, False)
        x_130 = None
        x_132 = torch.conv2d(
            x_131,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_131 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_133 = torch.nn.functional.dropout(x_132, 0.0, False, False)
        x_132 = None
        reshape_9 = l_self_modules_stages_modules_3_modules_blocks_modules_1_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_parameters_gamma_ = (
            None
        )
        x_134 = x_133.mul(reshape_9)
        x_133 = reshape_9 = None
        x_135 = x_134 + x_123
        x_134 = x_123 = None
        x_136 = torch.nn.functional.adaptive_avg_pool2d(x_135, 1)
        x_135 = None
        x_137 = x_136.permute(0, 2, 3, 1)
        x_136 = None
        v_14 = torch.var(x_137, dim=(-1,), keepdim=True)
        add_24 = v_14 + 1e-06
        v_14 = None
        rsqrt_14 = torch.rsqrt(add_24)
        add_24 = None
        x_138 = x_137 * rsqrt_14
        x_137 = rsqrt_14 = None
        x_139 = x_138 * l_self_modules_head_modules_norm_parameters_weight_
        x_138 = l_self_modules_head_modules_norm_parameters_weight_ = None
        x_140 = x_139.permute(0, 3, 1, 2)
        x_139 = None
        x_141 = x_140.flatten(1, -1)
        x_140 = None
        x_142 = torch.nn.functional.dropout(x_141, 0.0, False, False)
        x_141 = None
        x_143 = torch._C._nn.linear(
            x_142,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_142 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_143,)
