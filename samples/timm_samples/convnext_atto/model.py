import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        s53: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_1_eps: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_eps: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_drop1_p: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_drop2_p: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_eps: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_drop1_p: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_drop2_p: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_0_eps: torch.Tensor,
        L_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_eps: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_drop1_p: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_drop2_p: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_eps: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_drop1_p: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_drop2_p: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_0_eps: torch.Tensor,
        L_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_eps: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_drop1_p: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_drop2_p: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_eps: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_drop1_p: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_drop2_p: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_eps: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_drop1_p: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_drop2_p: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_eps: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_drop1_p: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_drop2_p: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_eps: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_drop1_p: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_drop2_p: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_eps: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_drop1_p: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_drop2_p: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_0_eps: torch.Tensor,
        L_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_eps: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_drop1_p: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_drop2_p: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_eps: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_drop1_p: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_drop2_p: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_norm_eps: torch.Tensor,
        L_self_modules_head_modules_drop_p: torch.Tensor,
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
        l_self_modules_stem_modules_1_parameters_bias_ = (
            L_self_modules_stem_modules_1_parameters_bias_
        )
        l_self_modules_stem_modules_1_eps = L_self_modules_stem_modules_1_eps
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_eps = (
            L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_eps
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_drop1_p = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_drop1_p
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_drop2_p = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_drop2_p
        l_self_modules_stages_modules_0_modules_blocks_modules_0_parameters_gamma_ = (
            L_self_modules_stages_modules_0_modules_blocks_modules_0_parameters_gamma_
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_eps = (
            L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_eps
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_drop1_p = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_drop1_p
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_drop2_p = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_drop2_p
        l_self_modules_stages_modules_0_modules_blocks_modules_1_parameters_gamma_ = (
            L_self_modules_stages_modules_0_modules_blocks_modules_1_parameters_gamma_
        )
        l_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_weight_ = L_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_weight_
        l_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_bias_ = L_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_bias_
        l_self_modules_stages_modules_1_modules_downsample_modules_0_eps = (
            L_self_modules_stages_modules_1_modules_downsample_modules_0_eps
        )
        l_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_weight_ = L_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_weight_
        l_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_bias_ = L_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_eps = (
            L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_eps
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_drop1_p = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_drop1_p
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_drop2_p = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_drop2_p
        l_self_modules_stages_modules_1_modules_blocks_modules_0_parameters_gamma_ = (
            L_self_modules_stages_modules_1_modules_blocks_modules_0_parameters_gamma_
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_eps = (
            L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_eps
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_drop1_p = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_drop1_p
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_drop2_p = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_drop2_p
        l_self_modules_stages_modules_1_modules_blocks_modules_1_parameters_gamma_ = (
            L_self_modules_stages_modules_1_modules_blocks_modules_1_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_weight_ = L_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_bias_ = L_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_bias_
        l_self_modules_stages_modules_2_modules_downsample_modules_0_eps = (
            L_self_modules_stages_modules_2_modules_downsample_modules_0_eps
        )
        l_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_weight_ = L_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_bias_ = L_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_eps = (
            L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_eps
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_drop1_p = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_drop1_p
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_drop2_p = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_drop2_p
        l_self_modules_stages_modules_2_modules_blocks_modules_0_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_0_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_eps = (
            L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_eps
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_drop1_p = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_drop1_p
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_drop2_p = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_drop2_p
        l_self_modules_stages_modules_2_modules_blocks_modules_1_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_1_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_eps = (
            L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_eps
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_drop1_p = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_drop1_p
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_drop2_p = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_drop2_p
        l_self_modules_stages_modules_2_modules_blocks_modules_2_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_2_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_eps = (
            L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_eps
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_drop1_p = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_drop1_p
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_drop2_p = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_drop2_p
        l_self_modules_stages_modules_2_modules_blocks_modules_3_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_3_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_eps = (
            L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_eps
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_drop1_p = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_drop1_p
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_drop2_p = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_drop2_p
        l_self_modules_stages_modules_2_modules_blocks_modules_4_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_4_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_eps = (
            L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_eps
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_drop1_p = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_drop1_p
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_drop2_p = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_drop2_p
        l_self_modules_stages_modules_2_modules_blocks_modules_5_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_5_parameters_gamma_
        )
        l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_weight_ = L_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_weight_
        l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_bias_ = L_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_bias_
        l_self_modules_stages_modules_3_modules_downsample_modules_0_eps = (
            L_self_modules_stages_modules_3_modules_downsample_modules_0_eps
        )
        l_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_weight_ = L_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_weight_
        l_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_bias_ = L_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_eps = (
            L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_eps
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_drop1_p = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_drop1_p
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_drop2_p = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_drop2_p
        l_self_modules_stages_modules_3_modules_blocks_modules_0_parameters_gamma_ = (
            L_self_modules_stages_modules_3_modules_blocks_modules_0_parameters_gamma_
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_eps = (
            L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_eps
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_drop1_p = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_drop1_p
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_drop2_p = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_drop2_p
        l_self_modules_stages_modules_3_modules_blocks_modules_1_parameters_gamma_ = (
            L_self_modules_stages_modules_3_modules_blocks_modules_1_parameters_gamma_
        )
        l_self_modules_head_modules_norm_parameters_weight_ = (
            L_self_modules_head_modules_norm_parameters_weight_
        )
        l_self_modules_head_modules_norm_parameters_bias_ = (
            L_self_modules_head_modules_norm_parameters_bias_
        )
        l_self_modules_head_modules_norm_eps = L_self_modules_head_modules_norm_eps
        l_self_modules_head_modules_drop_p = L_self_modules_head_modules_drop_p
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
        item = l_self_modules_stem_modules_1_eps.item()
        l_self_modules_stem_modules_1_eps = None
        x_1 = torch.nn.functional.layer_norm(
            x,
            (40,),
            l_self_modules_stem_modules_1_parameters_weight_,
            l_self_modules_stem_modules_1_parameters_bias_,
            item,
        )
        x = (
            l_self_modules_stem_modules_1_parameters_weight_
        ) = l_self_modules_stem_modules_1_parameters_bias_ = item = None
        x_2 = x_1.permute(0, 3, 1, 2)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            40,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_bias_ = (None)
        x_4 = x_3.permute(0, 2, 3, 1)
        x_3 = None
        item_1 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_eps.item()
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_eps = None
        x_5 = torch.nn.functional.layer_norm(
            x_4,
            (40,),
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_bias_,
            item_1,
        )
        x_4 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_bias_ = (item_1) = (
            None
        )
        x_6 = x_5.permute(0, 3, 1, 2)
        x_5 = None
        x_7 = torch.conv2d(
            x_6,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_6 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_8 = torch.nn.functional.gelu(x_7)
        x_7 = None
        item_2 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_drop1_p.item()
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_drop1_p = (
            None
        )
        x_9 = torch.nn.functional.dropout(x_8, item_2, False, False)
        x_8 = item_2 = None
        x_10 = torch.conv2d(
            x_9,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_9 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        item_3 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_drop2_p.item()
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_drop2_p = (
            None
        )
        x_11 = torch.nn.functional.dropout(x_10, item_3, False, False)
        x_10 = item_3 = None
        reshape = l_self_modules_stages_modules_0_modules_blocks_modules_0_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_parameters_gamma_ = (
            None
        )
        x_12 = x_11.mul(reshape)
        x_11 = reshape = None
        x_13 = x_12 + x_2
        x_12 = x_2 = None
        x_14 = torch.conv2d(
            x_13,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            40,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_bias_ = (None)
        x_15 = x_14.permute(0, 2, 3, 1)
        x_14 = None
        item_4 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_eps.item()
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_eps = None
        x_16 = torch.nn.functional.layer_norm(
            x_15,
            (40,),
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_,
            item_4,
        )
        x_15 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_ = (item_4) = (
            None
        )
        x_17 = x_16.permute(0, 3, 1, 2)
        x_16 = None
        x_18 = torch.conv2d(
            x_17,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_17 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_19 = torch.nn.functional.gelu(x_18)
        x_18 = None
        item_5 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_drop1_p.item()
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_drop1_p = (
            None
        )
        x_20 = torch.nn.functional.dropout(x_19, item_5, False, False)
        x_19 = item_5 = None
        x_21 = torch.conv2d(
            x_20,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_20 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        item_6 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_drop2_p.item()
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_drop2_p = (
            None
        )
        x_22 = torch.nn.functional.dropout(x_21, item_6, False, False)
        x_21 = item_6 = None
        reshape_1 = l_self_modules_stages_modules_0_modules_blocks_modules_1_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_parameters_gamma_ = (
            None
        )
        x_23 = x_22.mul(reshape_1)
        x_22 = reshape_1 = None
        x_24 = x_23 + x_13
        x_23 = x_13 = None
        x_25 = x_24.permute(0, 2, 3, 1)
        x_24 = None
        item_7 = l_self_modules_stages_modules_1_modules_downsample_modules_0_eps.item()
        l_self_modules_stages_modules_1_modules_downsample_modules_0_eps = None
        x_26 = torch.nn.functional.layer_norm(
            x_25,
            (40,),
            l_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_weight_,
            l_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_bias_,
            item_7,
        )
        x_25 = l_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_weight_ = l_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_bias_ = (item_7) = (
            None
        )
        x_27 = x_26.permute(0, 3, 1, 2)
        x_26 = None
        input_2 = torch.conv2d(
            x_27,
            l_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_27 = l_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_weight_ = l_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_bias_ = (None)
        x_28 = torch.conv2d(
            input_2,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            80,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_bias_ = (None)
        x_29 = x_28.permute(0, 2, 3, 1)
        x_28 = None
        item_8 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_eps.item()
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_eps = None
        x_30 = torch.nn.functional.layer_norm(
            x_29,
            (80,),
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_,
            item_8,
        )
        x_29 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_ = (item_8) = (
            None
        )
        x_31 = x_30.permute(0, 3, 1, 2)
        x_30 = None
        x_32 = torch.conv2d(
            x_31,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_31 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_33 = torch.nn.functional.gelu(x_32)
        x_32 = None
        item_9 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_drop1_p.item()
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_drop1_p = (
            None
        )
        x_34 = torch.nn.functional.dropout(x_33, item_9, False, False)
        x_33 = item_9 = None
        x_35 = torch.conv2d(
            x_34,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_34 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        item_10 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_drop2_p.item()
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_drop2_p = (
            None
        )
        x_36 = torch.nn.functional.dropout(x_35, item_10, False, False)
        x_35 = item_10 = None
        reshape_2 = l_self_modules_stages_modules_1_modules_blocks_modules_0_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_parameters_gamma_ = (
            None
        )
        x_37 = x_36.mul(reshape_2)
        x_36 = reshape_2 = None
        x_38 = x_37 + input_2
        x_37 = input_2 = None
        x_39 = torch.conv2d(
            x_38,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            80,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_bias_ = (None)
        x_40 = x_39.permute(0, 2, 3, 1)
        x_39 = None
        item_11 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_eps.item()
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_eps = None
        x_41 = torch.nn.functional.layer_norm(
            x_40,
            (80,),
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_,
            item_11,
        )
        x_40 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_ = (item_11) = (
            None
        )
        x_42 = x_41.permute(0, 3, 1, 2)
        x_41 = None
        x_43 = torch.conv2d(
            x_42,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_42 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_44 = torch.nn.functional.gelu(x_43)
        x_43 = None
        item_12 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_drop1_p.item()
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_drop1_p = (
            None
        )
        x_45 = torch.nn.functional.dropout(x_44, item_12, False, False)
        x_44 = item_12 = None
        x_46 = torch.conv2d(
            x_45,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_45 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        item_13 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_drop2_p.item()
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_drop2_p = (
            None
        )
        x_47 = torch.nn.functional.dropout(x_46, item_13, False, False)
        x_46 = item_13 = None
        reshape_3 = l_self_modules_stages_modules_1_modules_blocks_modules_1_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_parameters_gamma_ = (
            None
        )
        x_48 = x_47.mul(reshape_3)
        x_47 = reshape_3 = None
        x_49 = x_48 + x_38
        x_48 = x_38 = None
        x_50 = x_49.permute(0, 2, 3, 1)
        x_49 = None
        item_14 = (
            l_self_modules_stages_modules_2_modules_downsample_modules_0_eps.item()
        )
        l_self_modules_stages_modules_2_modules_downsample_modules_0_eps = None
        x_51 = torch.nn.functional.layer_norm(
            x_50,
            (80,),
            l_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_weight_,
            l_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_bias_,
            item_14,
        )
        x_50 = l_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_weight_ = l_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_bias_ = (item_14) = (
            None
        )
        x_52 = x_51.permute(0, 3, 1, 2)
        x_51 = None
        input_3 = torch.conv2d(
            x_52,
            l_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_52 = l_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_weight_ = l_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_bias_ = (None)
        x_53 = torch.conv2d(
            input_3,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            160,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_bias_ = (None)
        x_54 = x_53.permute(0, 2, 3, 1)
        x_53 = None
        item_15 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_eps.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_eps = None
        x_55 = torch.nn.functional.layer_norm(
            x_54,
            (160,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_,
            item_15,
        )
        x_54 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_ = (item_15) = (
            None
        )
        x_56 = x_55.permute(0, 3, 1, 2)
        x_55 = None
        x_57 = torch.conv2d(
            x_56,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_56 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_58 = torch.nn.functional.gelu(x_57)
        x_57 = None
        item_16 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_drop1_p.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_drop1_p = (
            None
        )
        x_59 = torch.nn.functional.dropout(x_58, item_16, False, False)
        x_58 = item_16 = None
        x_60 = torch.conv2d(
            x_59,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_59 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        item_17 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_drop2_p.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_drop2_p = (
            None
        )
        x_61 = torch.nn.functional.dropout(x_60, item_17, False, False)
        x_60 = item_17 = None
        reshape_4 = l_self_modules_stages_modules_2_modules_blocks_modules_0_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_parameters_gamma_ = (
            None
        )
        x_62 = x_61.mul(reshape_4)
        x_61 = reshape_4 = None
        x_63 = x_62 + input_3
        x_62 = input_3 = None
        x_64 = torch.conv2d(
            x_63,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            160,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_bias_ = (None)
        x_65 = x_64.permute(0, 2, 3, 1)
        x_64 = None
        item_18 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_eps.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_eps = None
        x_66 = torch.nn.functional.layer_norm(
            x_65,
            (160,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_,
            item_18,
        )
        x_65 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_ = (item_18) = (
            None
        )
        x_67 = x_66.permute(0, 3, 1, 2)
        x_66 = None
        x_68 = torch.conv2d(
            x_67,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_67 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_69 = torch.nn.functional.gelu(x_68)
        x_68 = None
        item_19 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_drop1_p.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_drop1_p = (
            None
        )
        x_70 = torch.nn.functional.dropout(x_69, item_19, False, False)
        x_69 = item_19 = None
        x_71 = torch.conv2d(
            x_70,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_70 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        item_20 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_drop2_p.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_drop2_p = (
            None
        )
        x_72 = torch.nn.functional.dropout(x_71, item_20, False, False)
        x_71 = item_20 = None
        reshape_5 = l_self_modules_stages_modules_2_modules_blocks_modules_1_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_parameters_gamma_ = (
            None
        )
        x_73 = x_72.mul(reshape_5)
        x_72 = reshape_5 = None
        x_74 = x_73 + x_63
        x_73 = x_63 = None
        x_75 = torch.conv2d(
            x_74,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            160,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_bias_ = (None)
        x_76 = x_75.permute(0, 2, 3, 1)
        x_75 = None
        item_21 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_eps.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_eps = None
        x_77 = torch.nn.functional.layer_norm(
            x_76,
            (160,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_,
            item_21,
        )
        x_76 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_ = (item_21) = (
            None
        )
        x_78 = x_77.permute(0, 3, 1, 2)
        x_77 = None
        x_79 = torch.conv2d(
            x_78,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_78 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_80 = torch.nn.functional.gelu(x_79)
        x_79 = None
        item_22 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_drop1_p.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_drop1_p = (
            None
        )
        x_81 = torch.nn.functional.dropout(x_80, item_22, False, False)
        x_80 = item_22 = None
        x_82 = torch.conv2d(
            x_81,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_81 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        item_23 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_drop2_p.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_drop2_p = (
            None
        )
        x_83 = torch.nn.functional.dropout(x_82, item_23, False, False)
        x_82 = item_23 = None
        reshape_6 = l_self_modules_stages_modules_2_modules_blocks_modules_2_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_parameters_gamma_ = (
            None
        )
        x_84 = x_83.mul(reshape_6)
        x_83 = reshape_6 = None
        x_85 = x_84 + x_74
        x_84 = x_74 = None
        x_86 = torch.conv2d(
            x_85,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            160,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_bias_ = (None)
        x_87 = x_86.permute(0, 2, 3, 1)
        x_86 = None
        item_24 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_eps.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_eps = None
        x_88 = torch.nn.functional.layer_norm(
            x_87,
            (160,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_,
            item_24,
        )
        x_87 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_ = (item_24) = (
            None
        )
        x_89 = x_88.permute(0, 3, 1, 2)
        x_88 = None
        x_90 = torch.conv2d(
            x_89,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_89 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_91 = torch.nn.functional.gelu(x_90)
        x_90 = None
        item_25 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_drop1_p.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_drop1_p = (
            None
        )
        x_92 = torch.nn.functional.dropout(x_91, item_25, False, False)
        x_91 = item_25 = None
        x_93 = torch.conv2d(
            x_92,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_92 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        item_26 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_drop2_p.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_drop2_p = (
            None
        )
        x_94 = torch.nn.functional.dropout(x_93, item_26, False, False)
        x_93 = item_26 = None
        reshape_7 = l_self_modules_stages_modules_2_modules_blocks_modules_3_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_parameters_gamma_ = (
            None
        )
        x_95 = x_94.mul(reshape_7)
        x_94 = reshape_7 = None
        x_96 = x_95 + x_85
        x_95 = x_85 = None
        x_97 = torch.conv2d(
            x_96,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            160,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_dw_parameters_bias_ = (None)
        x_98 = x_97.permute(0, 2, 3, 1)
        x_97 = None
        item_27 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_eps.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_eps = None
        x_99 = torch.nn.functional.layer_norm(
            x_98,
            (160,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_,
            item_27,
        )
        x_98 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_ = (item_27) = (
            None
        )
        x_100 = x_99.permute(0, 3, 1, 2)
        x_99 = None
        x_101 = torch.conv2d(
            x_100,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_100 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_102 = torch.nn.functional.gelu(x_101)
        x_101 = None
        item_28 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_drop1_p.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_drop1_p = (
            None
        )
        x_103 = torch.nn.functional.dropout(x_102, item_28, False, False)
        x_102 = item_28 = None
        x_104 = torch.conv2d(
            x_103,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_103 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        item_29 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_drop2_p.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_drop2_p = (
            None
        )
        x_105 = torch.nn.functional.dropout(x_104, item_29, False, False)
        x_104 = item_29 = None
        reshape_8 = l_self_modules_stages_modules_2_modules_blocks_modules_4_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_parameters_gamma_ = (
            None
        )
        x_106 = x_105.mul(reshape_8)
        x_105 = reshape_8 = None
        x_107 = x_106 + x_96
        x_106 = x_96 = None
        x_108 = torch.conv2d(
            x_107,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            160,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_dw_parameters_bias_ = (None)
        x_109 = x_108.permute(0, 2, 3, 1)
        x_108 = None
        item_30 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_eps.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_eps = None
        x_110 = torch.nn.functional.layer_norm(
            x_109,
            (160,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_,
            item_30,
        )
        x_109 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_ = (item_30) = (
            None
        )
        x_111 = x_110.permute(0, 3, 1, 2)
        x_110 = None
        x_112 = torch.conv2d(
            x_111,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_111 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_113 = torch.nn.functional.gelu(x_112)
        x_112 = None
        item_31 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_drop1_p.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_drop1_p = (
            None
        )
        x_114 = torch.nn.functional.dropout(x_113, item_31, False, False)
        x_113 = item_31 = None
        x_115 = torch.conv2d(
            x_114,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_114 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        item_32 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_drop2_p.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_drop2_p = (
            None
        )
        x_116 = torch.nn.functional.dropout(x_115, item_32, False, False)
        x_115 = item_32 = None
        reshape_9 = l_self_modules_stages_modules_2_modules_blocks_modules_5_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_parameters_gamma_ = (
            None
        )
        x_117 = x_116.mul(reshape_9)
        x_116 = reshape_9 = None
        x_118 = x_117 + x_107
        x_117 = x_107 = None
        x_119 = x_118.permute(0, 2, 3, 1)
        x_118 = None
        item_33 = (
            l_self_modules_stages_modules_3_modules_downsample_modules_0_eps.item()
        )
        l_self_modules_stages_modules_3_modules_downsample_modules_0_eps = None
        x_120 = torch.nn.functional.layer_norm(
            x_119,
            (160,),
            l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_bias_,
            item_33,
        )
        x_119 = l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_bias_ = (item_33) = (
            None
        )
        x_121 = x_120.permute(0, 3, 1, 2)
        x_120 = None
        input_4 = torch.conv2d(
            x_121,
            l_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_121 = l_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_bias_ = (None)
        x_122 = torch.conv2d(
            input_4,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            320,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_bias_ = (None)
        x_123 = x_122.permute(0, 2, 3, 1)
        x_122 = None
        item_34 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_eps.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_eps = None
        x_124 = torch.nn.functional.layer_norm(
            x_123,
            (320,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_,
            item_34,
        )
        x_123 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_ = (item_34) = (
            None
        )
        x_125 = x_124.permute(0, 3, 1, 2)
        x_124 = None
        x_126 = torch.conv2d(
            x_125,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_125 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_127 = torch.nn.functional.gelu(x_126)
        x_126 = None
        item_35 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_drop1_p.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_drop1_p = (
            None
        )
        x_128 = torch.nn.functional.dropout(x_127, item_35, False, False)
        x_127 = item_35 = None
        x_129 = torch.conv2d(
            x_128,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_128 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        item_36 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_drop2_p.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_drop2_p = (
            None
        )
        x_130 = torch.nn.functional.dropout(x_129, item_36, False, False)
        x_129 = item_36 = None
        reshape_10 = l_self_modules_stages_modules_3_modules_blocks_modules_0_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_parameters_gamma_ = (
            None
        )
        x_131 = x_130.mul(reshape_10)
        x_130 = reshape_10 = None
        x_132 = x_131 + input_4
        x_131 = input_4 = None
        x_133 = torch.conv2d(
            x_132,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            320,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_bias_ = (None)
        x_134 = x_133.permute(0, 2, 3, 1)
        x_133 = None
        item_37 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_eps.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_eps = None
        x_135 = torch.nn.functional.layer_norm(
            x_134,
            (320,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_,
            item_37,
        )
        x_134 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_ = (item_37) = (
            None
        )
        x_136 = x_135.permute(0, 3, 1, 2)
        x_135 = None
        x_137 = torch.conv2d(
            x_136,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_136 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_138 = torch.nn.functional.gelu(x_137)
        x_137 = None
        item_38 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_drop1_p.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_drop1_p = (
            None
        )
        x_139 = torch.nn.functional.dropout(x_138, item_38, False, False)
        x_138 = item_38 = None
        x_140 = torch.conv2d(
            x_139,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_139 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        item_39 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_drop2_p.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_drop2_p = (
            None
        )
        x_141 = torch.nn.functional.dropout(x_140, item_39, False, False)
        x_140 = item_39 = None
        reshape_11 = l_self_modules_stages_modules_3_modules_blocks_modules_1_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_parameters_gamma_ = (
            None
        )
        x_142 = x_141.mul(reshape_11)
        x_141 = reshape_11 = None
        x_143 = x_142 + x_132
        x_142 = x_132 = None
        x_144 = torch.nn.functional.adaptive_avg_pool2d(x_143, 1)
        x_143 = None
        x_145 = x_144.permute(0, 2, 3, 1)
        x_144 = None
        item_40 = l_self_modules_head_modules_norm_eps.item()
        l_self_modules_head_modules_norm_eps = None
        x_146 = torch.nn.functional.layer_norm(
            x_145,
            (320,),
            l_self_modules_head_modules_norm_parameters_weight_,
            l_self_modules_head_modules_norm_parameters_bias_,
            item_40,
        )
        x_145 = (
            l_self_modules_head_modules_norm_parameters_weight_
        ) = l_self_modules_head_modules_norm_parameters_bias_ = item_40 = None
        x_147 = x_146.permute(0, 3, 1, 2)
        x_146 = None
        x_148 = x_147.flatten(1, -1)
        x_147 = None
        item_41 = l_self_modules_head_modules_drop_p.item()
        l_self_modules_head_modules_drop_p = None
        x_149 = torch.nn.functional.dropout(x_148, item_41, False, False)
        x_148 = item_41 = None
        x_150 = torch.nn.functional.linear(
            x_149,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_149 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_150,)
