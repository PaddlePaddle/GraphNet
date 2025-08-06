import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_conv_parameters_weight_
        )
        l_self_modules_stem_modules_conv_parameters_gain_ = (
            L_self_modules_stem_modules_conv_parameters_gain_
        )
        l_self_modules_stem_modules_conv_parameters_bias_ = (
            L_self_modules_stem_modules_conv_parameters_bias_
        )
        l_x_ = L_x_
        l_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_gain_ = L_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_gain_
        l_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_bias_
        l_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_gain_ = L_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_gain_
        l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_gain_ = L_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_gain_
        l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_gain_ = L_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_gain_
        l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_bias_
        )
        l_self_modules_head_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_fc_parameters_bias_
        )
        reshape = l_self_modules_stem_modules_conv_parameters_weight_.reshape(1, 64, -1)
        mul = l_self_modules_stem_modules_conv_parameters_gain_ * 0.14136494412567766
        l_self_modules_stem_modules_conv_parameters_gain_ = None
        view = mul.view(-1)
        mul = None
        batch_norm = torch.nn.functional.batch_norm(
            reshape, None, None, weight=view, training=True, momentum=0.0, eps=1e-05
        )
        reshape = view = None
        weight = batch_norm.reshape_as(
            l_self_modules_stem_modules_conv_parameters_weight_
        )
        batch_norm = l_self_modules_stem_modules_conv_parameters_weight_ = None
        input_1 = torch.conv2d(
            l_x_,
            weight,
            l_self_modules_stem_modules_conv_parameters_bias_,
            (2, 2),
            (3, 3),
            (1, 1),
            1,
        )
        l_x_ = weight = l_self_modules_stem_modules_conv_parameters_bias_ = None
        input_2 = torch.nn.functional.max_pool2d(
            input_1, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        input_1 = None
        relu = torch.nn.functional.relu(input_2, inplace=False)
        input_2 = None
        out = relu * 1.0
        relu = None
        reshape_1 = l_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_2 = (
            l_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_gain_
            * 0.21424485743045807
        )
        l_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_gain_ = (
            None
        )
        view_1 = mul_2.view(-1)
        mul_2 = None
        batch_norm_1 = torch.nn.functional.batch_norm(
            reshape_1, None, None, weight=view_1, training=True, momentum=0.0, eps=1e-05
        )
        reshape_1 = view_1 = None
        weight_1 = batch_norm_1.reshape_as(
            l_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_weight_
        )
        batch_norm_1 = l_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        shortcut = torch.conv2d(
            out,
            weight_1,
            l_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        weight_1 = l_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_bias_ = (None)
        reshape_2 = l_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_weight_.reshape(
            1, 64, -1
        )
        mul_3 = (
            l_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_gain_
            * 0.21424485743045807
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_gain_ = None
        view_2 = mul_3.view(-1)
        mul_3 = None
        batch_norm_2 = torch.nn.functional.batch_norm(
            reshape_2, None, None, weight=view_2, training=True, momentum=0.0, eps=1e-05
        )
        reshape_2 = view_2 = None
        weight_2 = batch_norm_2.reshape_as(
            l_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_weight_
        )
        batch_norm_2 = (
            l_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_weight_
        ) = None
        out_1 = torch.conv2d(
            out,
            weight_2,
            l_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out = (
            weight_2
        ) = (
            l_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_bias_
        ) = None
        relu_1 = torch.nn.functional.relu(out_1, inplace=True)
        out_1 = None
        reshape_3 = l_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_weight_.reshape(
            1, 64, -1
        )
        mul_4 = (
            l_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_gain_
            * 0.07141495247681935
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_gain_ = None
        view_3 = mul_4.view(-1)
        mul_4 = None
        batch_norm_3 = torch.nn.functional.batch_norm(
            reshape_3, None, None, weight=view_3, training=True, momentum=0.0, eps=1e-05
        )
        reshape_3 = view_3 = None
        weight_3 = batch_norm_3.reshape_as(
            l_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_weight_
        )
        batch_norm_3 = (
            l_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_weight_
        ) = None
        out_2 = torch.conv2d(
            relu_1,
            weight_3,
            l_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        relu_1 = (
            weight_3
        ) = (
            l_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_bias_
        ) = None
        relu_2 = torch.nn.functional.relu(out_2, inplace=False)
        out_2 = None
        reshape_4 = l_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_5 = (
            l_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_gain_
            * 0.21424485743045807
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_gain_ = None
        view_4 = mul_5.view(-1)
        mul_5 = None
        batch_norm_4 = torch.nn.functional.batch_norm(
            reshape_4, None, None, weight=view_4, training=True, momentum=0.0, eps=1e-05
        )
        reshape_4 = view_4 = None
        weight_4 = batch_norm_4.reshape_as(
            l_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_weight_
        )
        batch_norm_4 = (
            l_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_weight_
        ) = None
        out_3 = torch.conv2d(
            relu_2,
            weight_4,
            l_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        relu_2 = (
            weight_4
        ) = (
            l_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_bias_
        ) = None
        mul_6 = out_3 * 0.2
        out_3 = None
        out_4 = mul_6 + shortcut
        mul_6 = shortcut = None
        relu_3 = torch.nn.functional.relu(out_4, inplace=False)
        out_5 = relu_3 * 0.9805806756909201
        relu_3 = None
        reshape_5 = l_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_weight_.reshape(
            1, 64, -1
        )
        mul_8 = (
            l_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_gain_
            * 0.10712242871522903
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_gain_ = None
        view_5 = mul_8.view(-1)
        mul_8 = None
        batch_norm_5 = torch.nn.functional.batch_norm(
            reshape_5, None, None, weight=view_5, training=True, momentum=0.0, eps=1e-05
        )
        reshape_5 = view_5 = None
        weight_5 = batch_norm_5.reshape_as(
            l_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_weight_
        )
        batch_norm_5 = (
            l_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_weight_
        ) = None
        out_6 = torch.conv2d(
            out_5,
            weight_5,
            l_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_5 = (
            weight_5
        ) = (
            l_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_bias_
        ) = None
        relu_4 = torch.nn.functional.relu(out_6, inplace=True)
        out_6 = None
        reshape_6 = l_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_weight_.reshape(
            1, 64, -1
        )
        mul_9 = (
            l_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_gain_
            * 0.07141495247681935
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_gain_ = None
        view_6 = mul_9.view(-1)
        mul_9 = None
        batch_norm_6 = torch.nn.functional.batch_norm(
            reshape_6, None, None, weight=view_6, training=True, momentum=0.0, eps=1e-05
        )
        reshape_6 = view_6 = None
        weight_6 = batch_norm_6.reshape_as(
            l_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_weight_
        )
        batch_norm_6 = (
            l_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_weight_
        ) = None
        out_7 = torch.conv2d(
            relu_4,
            weight_6,
            l_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        relu_4 = (
            weight_6
        ) = (
            l_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_bias_
        ) = None
        relu_5 = torch.nn.functional.relu(out_7, inplace=False)
        out_7 = None
        reshape_7 = l_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_10 = (
            l_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_gain_
            * 0.21424485743045807
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_gain_ = None
        view_7 = mul_10.view(-1)
        mul_10 = None
        batch_norm_7 = torch.nn.functional.batch_norm(
            reshape_7, None, None, weight=view_7, training=True, momentum=0.0, eps=1e-05
        )
        reshape_7 = view_7 = None
        weight_7 = batch_norm_7.reshape_as(
            l_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_weight_
        )
        batch_norm_7 = (
            l_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_weight_
        ) = None
        out_8 = torch.conv2d(
            relu_5,
            weight_7,
            l_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        relu_5 = (
            weight_7
        ) = (
            l_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_bias_
        ) = None
        mul_11 = out_8 * 0.2
        out_8 = None
        out_9 = mul_11 + out_4
        mul_11 = out_4 = None
        relu_6 = torch.nn.functional.relu(out_9, inplace=False)
        out_10 = relu_6 * 0.9622504486493761
        relu_6 = None
        reshape_8 = l_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_weight_.reshape(
            1, 64, -1
        )
        mul_13 = (
            l_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_gain_
            * 0.10712242871522903
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_gain_ = None
        view_8 = mul_13.view(-1)
        mul_13 = None
        batch_norm_8 = torch.nn.functional.batch_norm(
            reshape_8, None, None, weight=view_8, training=True, momentum=0.0, eps=1e-05
        )
        reshape_8 = view_8 = None
        weight_8 = batch_norm_8.reshape_as(
            l_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_weight_
        )
        batch_norm_8 = (
            l_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_weight_
        ) = None
        out_11 = torch.conv2d(
            out_10,
            weight_8,
            l_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_10 = (
            weight_8
        ) = (
            l_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_bias_
        ) = None
        relu_7 = torch.nn.functional.relu(out_11, inplace=True)
        out_11 = None
        reshape_9 = l_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_weight_.reshape(
            1, 64, -1
        )
        mul_14 = (
            l_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_gain_
            * 0.07141495247681935
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_gain_ = None
        view_9 = mul_14.view(-1)
        mul_14 = None
        batch_norm_9 = torch.nn.functional.batch_norm(
            reshape_9, None, None, weight=view_9, training=True, momentum=0.0, eps=1e-05
        )
        reshape_9 = view_9 = None
        weight_9 = batch_norm_9.reshape_as(
            l_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_weight_
        )
        batch_norm_9 = (
            l_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_weight_
        ) = None
        out_12 = torch.conv2d(
            relu_7,
            weight_9,
            l_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        relu_7 = (
            weight_9
        ) = (
            l_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_bias_
        ) = None
        relu_8 = torch.nn.functional.relu(out_12, inplace=False)
        out_12 = None
        reshape_10 = l_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_15 = (
            l_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_gain_
            * 0.21424485743045807
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_gain_ = None
        view_10 = mul_15.view(-1)
        mul_15 = None
        batch_norm_10 = torch.nn.functional.batch_norm(
            reshape_10,
            None,
            None,
            weight=view_10,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_10 = view_10 = None
        weight_10 = batch_norm_10.reshape_as(
            l_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_weight_
        )
        batch_norm_10 = (
            l_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_weight_
        ) = None
        out_13 = torch.conv2d(
            relu_8,
            weight_10,
            l_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        relu_8 = (
            weight_10
        ) = (
            l_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_bias_
        ) = None
        mul_16 = out_13 * 0.2
        out_13 = None
        out_14 = mul_16 + out_9
        mul_16 = out_9 = None
        relu_9 = torch.nn.functional.relu(out_14, inplace=False)
        out_14 = None
        out_15 = relu_9 * 0.9449111825230679
        relu_9 = None
        avg_pool2d = torch._C._nn.avg_pool2d(out_15, 2, 2, 0, True, False, None)
        reshape_11 = l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_18 = (
            l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_gain_
            * 0.10712242871522903
        )
        l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_gain_ = (
            None
        )
        view_11 = mul_18.view(-1)
        mul_18 = None
        batch_norm_11 = torch.nn.functional.batch_norm(
            reshape_11,
            None,
            None,
            weight=view_11,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_11 = view_11 = None
        weight_11 = batch_norm_11.reshape_as(
            l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_
        )
        batch_norm_11 = l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        shortcut_1 = torch.conv2d(
            avg_pool2d,
            weight_11,
            l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        avg_pool2d = (
            weight_11
        ) = l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_bias_ = (None)
        reshape_12 = l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_19 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_gain_
            * 0.10712242871522903
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_gain_ = None
        view_12 = mul_19.view(-1)
        mul_19 = None
        batch_norm_12 = torch.nn.functional.batch_norm(
            reshape_12,
            None,
            None,
            weight=view_12,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_12 = view_12 = None
        weight_12 = batch_norm_12.reshape_as(
            l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_weight_
        )
        batch_norm_12 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_weight_
        ) = None
        out_16 = torch.conv2d(
            out_15,
            weight_12,
            l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_15 = (
            weight_12
        ) = (
            l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_bias_
        ) = None
        relu_10 = torch.nn.functional.relu(out_16, inplace=True)
        out_16 = None
        reshape_13 = l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_20 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_gain_
            * 0.050497997174473994
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_gain_ = None
        view_13 = mul_20.view(-1)
        mul_20 = None
        batch_norm_13 = torch.nn.functional.batch_norm(
            reshape_13,
            None,
            None,
            weight=view_13,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_13 = view_13 = None
        weight_13 = batch_norm_13.reshape_as(
            l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_weight_
        )
        batch_norm_13 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_weight_
        ) = None
        out_17 = torch.conv2d(
            relu_10,
            weight_13,
            l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        relu_10 = (
            weight_13
        ) = (
            l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_bias_
        ) = None
        relu_11 = torch.nn.functional.relu(out_17, inplace=False)
        out_17 = None
        reshape_14 = l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_21 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_gain_
            * 0.151493991523422
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_gain_ = None
        view_14 = mul_21.view(-1)
        mul_21 = None
        batch_norm_14 = torch.nn.functional.batch_norm(
            reshape_14,
            None,
            None,
            weight=view_14,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_14 = view_14 = None
        weight_14 = batch_norm_14.reshape_as(
            l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_weight_
        )
        batch_norm_14 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_weight_
        ) = None
        out_18 = torch.conv2d(
            relu_11,
            weight_14,
            l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        relu_11 = (
            weight_14
        ) = (
            l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_bias_
        ) = None
        mul_22 = out_18 * 0.2
        out_18 = None
        out_19 = mul_22 + shortcut_1
        mul_22 = shortcut_1 = None
        relu_12 = torch.nn.functional.relu(out_19, inplace=False)
        out_20 = relu_12 * 0.9805806756909201
        relu_12 = None
        reshape_15 = l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_24 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_gain_
            * 0.075746995761711
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_gain_ = None
        view_15 = mul_24.view(-1)
        mul_24 = None
        batch_norm_15 = torch.nn.functional.batch_norm(
            reshape_15,
            None,
            None,
            weight=view_15,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_15 = view_15 = None
        weight_15 = batch_norm_15.reshape_as(
            l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_weight_
        )
        batch_norm_15 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_weight_
        ) = None
        out_21 = torch.conv2d(
            out_20,
            weight_15,
            l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_20 = (
            weight_15
        ) = (
            l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_bias_
        ) = None
        relu_13 = torch.nn.functional.relu(out_21, inplace=True)
        out_21 = None
        reshape_16 = l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_25 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_gain_
            * 0.050497997174473994
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_gain_ = None
        view_16 = mul_25.view(-1)
        mul_25 = None
        batch_norm_16 = torch.nn.functional.batch_norm(
            reshape_16,
            None,
            None,
            weight=view_16,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_16 = view_16 = None
        weight_16 = batch_norm_16.reshape_as(
            l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_weight_
        )
        batch_norm_16 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_weight_
        ) = None
        out_22 = torch.conv2d(
            relu_13,
            weight_16,
            l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        relu_13 = (
            weight_16
        ) = (
            l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_bias_
        ) = None
        relu_14 = torch.nn.functional.relu(out_22, inplace=False)
        out_22 = None
        reshape_17 = l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_26 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_gain_
            * 0.151493991523422
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_gain_ = None
        view_17 = mul_26.view(-1)
        mul_26 = None
        batch_norm_17 = torch.nn.functional.batch_norm(
            reshape_17,
            None,
            None,
            weight=view_17,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_17 = view_17 = None
        weight_17 = batch_norm_17.reshape_as(
            l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_weight_
        )
        batch_norm_17 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_weight_
        ) = None
        out_23 = torch.conv2d(
            relu_14,
            weight_17,
            l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        relu_14 = (
            weight_17
        ) = (
            l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_bias_
        ) = None
        mul_27 = out_23 * 0.2
        out_23 = None
        out_24 = mul_27 + out_19
        mul_27 = out_19 = None
        relu_15 = torch.nn.functional.relu(out_24, inplace=False)
        out_25 = relu_15 * 0.9622504486493761
        relu_15 = None
        reshape_18 = l_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_29 = (
            l_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_gain_
            * 0.075746995761711
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_gain_ = None
        view_18 = mul_29.view(-1)
        mul_29 = None
        batch_norm_18 = torch.nn.functional.batch_norm(
            reshape_18,
            None,
            None,
            weight=view_18,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_18 = view_18 = None
        weight_18 = batch_norm_18.reshape_as(
            l_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_weight_
        )
        batch_norm_18 = (
            l_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_weight_
        ) = None
        out_26 = torch.conv2d(
            out_25,
            weight_18,
            l_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_25 = (
            weight_18
        ) = (
            l_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_bias_
        ) = None
        relu_16 = torch.nn.functional.relu(out_26, inplace=True)
        out_26 = None
        reshape_19 = l_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_30 = (
            l_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_gain_
            * 0.050497997174473994
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_gain_ = None
        view_19 = mul_30.view(-1)
        mul_30 = None
        batch_norm_19 = torch.nn.functional.batch_norm(
            reshape_19,
            None,
            None,
            weight=view_19,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_19 = view_19 = None
        weight_19 = batch_norm_19.reshape_as(
            l_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_weight_
        )
        batch_norm_19 = (
            l_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_weight_
        ) = None
        out_27 = torch.conv2d(
            relu_16,
            weight_19,
            l_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        relu_16 = (
            weight_19
        ) = (
            l_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_bias_
        ) = None
        relu_17 = torch.nn.functional.relu(out_27, inplace=False)
        out_27 = None
        reshape_20 = l_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_31 = (
            l_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_gain_
            * 0.151493991523422
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_gain_ = None
        view_20 = mul_31.view(-1)
        mul_31 = None
        batch_norm_20 = torch.nn.functional.batch_norm(
            reshape_20,
            None,
            None,
            weight=view_20,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_20 = view_20 = None
        weight_20 = batch_norm_20.reshape_as(
            l_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_weight_
        )
        batch_norm_20 = (
            l_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_weight_
        ) = None
        out_28 = torch.conv2d(
            relu_17,
            weight_20,
            l_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        relu_17 = (
            weight_20
        ) = (
            l_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_bias_
        ) = None
        mul_32 = out_28 * 0.2
        out_28 = None
        out_29 = mul_32 + out_24
        mul_32 = out_24 = None
        relu_18 = torch.nn.functional.relu(out_29, inplace=False)
        out_30 = relu_18 * 0.9449111825230679
        relu_18 = None
        reshape_21 = l_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_34 = (
            l_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_gain_
            * 0.075746995761711
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_gain_ = None
        view_21 = mul_34.view(-1)
        mul_34 = None
        batch_norm_21 = torch.nn.functional.batch_norm(
            reshape_21,
            None,
            None,
            weight=view_21,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_21 = view_21 = None
        weight_21 = batch_norm_21.reshape_as(
            l_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_weight_
        )
        batch_norm_21 = (
            l_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_weight_
        ) = None
        out_31 = torch.conv2d(
            out_30,
            weight_21,
            l_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_30 = (
            weight_21
        ) = (
            l_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_bias_
        ) = None
        relu_19 = torch.nn.functional.relu(out_31, inplace=True)
        out_31 = None
        reshape_22 = l_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_35 = (
            l_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_gain_
            * 0.050497997174473994
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_gain_ = None
        view_22 = mul_35.view(-1)
        mul_35 = None
        batch_norm_22 = torch.nn.functional.batch_norm(
            reshape_22,
            None,
            None,
            weight=view_22,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_22 = view_22 = None
        weight_22 = batch_norm_22.reshape_as(
            l_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_weight_
        )
        batch_norm_22 = (
            l_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_weight_
        ) = None
        out_32 = torch.conv2d(
            relu_19,
            weight_22,
            l_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        relu_19 = (
            weight_22
        ) = (
            l_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_bias_
        ) = None
        relu_20 = torch.nn.functional.relu(out_32, inplace=False)
        out_32 = None
        reshape_23 = l_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_36 = (
            l_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_gain_
            * 0.151493991523422
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_gain_ = None
        view_23 = mul_36.view(-1)
        mul_36 = None
        batch_norm_23 = torch.nn.functional.batch_norm(
            reshape_23,
            None,
            None,
            weight=view_23,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_23 = view_23 = None
        weight_23 = batch_norm_23.reshape_as(
            l_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_weight_
        )
        batch_norm_23 = (
            l_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_weight_
        ) = None
        out_33 = torch.conv2d(
            relu_20,
            weight_23,
            l_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        relu_20 = (
            weight_23
        ) = (
            l_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_bias_
        ) = None
        mul_37 = out_33 * 0.2
        out_33 = None
        out_34 = mul_37 + out_29
        mul_37 = out_29 = None
        relu_21 = torch.nn.functional.relu(out_34, inplace=False)
        out_34 = None
        out_35 = relu_21 * 0.9284766908852592
        relu_21 = None
        avg_pool2d_1 = torch._C._nn.avg_pool2d(out_35, 2, 2, 0, True, False, None)
        reshape_24 = l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_.reshape(
            1, 1024, -1
        )
        mul_39 = (
            l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_gain_
            * 0.075746995761711
        )
        l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_gain_ = (
            None
        )
        view_24 = mul_39.view(-1)
        mul_39 = None
        batch_norm_24 = torch.nn.functional.batch_norm(
            reshape_24,
            None,
            None,
            weight=view_24,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_24 = view_24 = None
        weight_24 = batch_norm_24.reshape_as(
            l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_
        )
        batch_norm_24 = l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        shortcut_2 = torch.conv2d(
            avg_pool2d_1,
            weight_24,
            l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        avg_pool2d_1 = (
            weight_24
        ) = l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_bias_ = (None)
        reshape_25 = l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_40 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_gain_
            * 0.075746995761711
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_gain_ = None
        view_25 = mul_40.view(-1)
        mul_40 = None
        batch_norm_25 = torch.nn.functional.batch_norm(
            reshape_25,
            None,
            None,
            weight=view_25,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_25 = view_25 = None
        weight_25 = batch_norm_25.reshape_as(
            l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_weight_
        )
        batch_norm_25 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_weight_
        ) = None
        out_36 = torch.conv2d(
            out_35,
            weight_25,
            l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_35 = (
            weight_25
        ) = (
            l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_bias_
        ) = None
        relu_22 = torch.nn.functional.relu(out_36, inplace=True)
        out_36 = None
        reshape_26 = l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_41 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_gain_
            * 0.035707476238409676
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_gain_ = None
        view_26 = mul_41.view(-1)
        mul_41 = None
        batch_norm_26 = torch.nn.functional.batch_norm(
            reshape_26,
            None,
            None,
            weight=view_26,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_26 = view_26 = None
        weight_26 = batch_norm_26.reshape_as(
            l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_weight_
        )
        batch_norm_26 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_weight_
        ) = None
        out_37 = torch.conv2d(
            relu_22,
            weight_26,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        relu_22 = (
            weight_26
        ) = (
            l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_bias_
        ) = None
        relu_23 = torch.nn.functional.relu(out_37, inplace=False)
        out_37 = None
        reshape_27 = l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_weight_.reshape(
            1, 1024, -1
        )
        mul_42 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_gain_
            * 0.10712242871522903
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_gain_ = None
        view_27 = mul_42.view(-1)
        mul_42 = None
        batch_norm_27 = torch.nn.functional.batch_norm(
            reshape_27,
            None,
            None,
            weight=view_27,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_27 = view_27 = None
        weight_27 = batch_norm_27.reshape_as(
            l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_weight_
        )
        batch_norm_27 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_weight_
        ) = None
        out_38 = torch.conv2d(
            relu_23,
            weight_27,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        relu_23 = (
            weight_27
        ) = (
            l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_bias_
        ) = None
        mul_43 = out_38 * 0.2
        out_38 = None
        out_39 = mul_43 + shortcut_2
        mul_43 = shortcut_2 = None
        relu_24 = torch.nn.functional.relu(out_39, inplace=False)
        out_40 = relu_24 * 0.9805806756909201
        relu_24 = None
        reshape_28 = l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_45 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_gain_
            * 0.05356121435761452
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_gain_ = None
        view_28 = mul_45.view(-1)
        mul_45 = None
        batch_norm_28 = torch.nn.functional.batch_norm(
            reshape_28,
            None,
            None,
            weight=view_28,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_28 = view_28 = None
        weight_28 = batch_norm_28.reshape_as(
            l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_weight_
        )
        batch_norm_28 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_weight_
        ) = None
        out_41 = torch.conv2d(
            out_40,
            weight_28,
            l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_40 = (
            weight_28
        ) = (
            l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_bias_
        ) = None
        relu_25 = torch.nn.functional.relu(out_41, inplace=True)
        out_41 = None
        reshape_29 = l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_46 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_gain_
            * 0.035707476238409676
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_gain_ = None
        view_29 = mul_46.view(-1)
        mul_46 = None
        batch_norm_29 = torch.nn.functional.batch_norm(
            reshape_29,
            None,
            None,
            weight=view_29,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_29 = view_29 = None
        weight_29 = batch_norm_29.reshape_as(
            l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_weight_
        )
        batch_norm_29 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_weight_
        ) = None
        out_42 = torch.conv2d(
            relu_25,
            weight_29,
            l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        relu_25 = (
            weight_29
        ) = (
            l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_bias_
        ) = None
        relu_26 = torch.nn.functional.relu(out_42, inplace=False)
        out_42 = None
        reshape_30 = l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_weight_.reshape(
            1, 1024, -1
        )
        mul_47 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_gain_
            * 0.10712242871522903
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_gain_ = None
        view_30 = mul_47.view(-1)
        mul_47 = None
        batch_norm_30 = torch.nn.functional.batch_norm(
            reshape_30,
            None,
            None,
            weight=view_30,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_30 = view_30 = None
        weight_30 = batch_norm_30.reshape_as(
            l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_weight_
        )
        batch_norm_30 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_weight_
        ) = None
        out_43 = torch.conv2d(
            relu_26,
            weight_30,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        relu_26 = (
            weight_30
        ) = (
            l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_bias_
        ) = None
        mul_48 = out_43 * 0.2
        out_43 = None
        out_44 = mul_48 + out_39
        mul_48 = out_39 = None
        relu_27 = torch.nn.functional.relu(out_44, inplace=False)
        out_45 = relu_27 * 0.9622504486493761
        relu_27 = None
        reshape_31 = l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_50 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_gain_
            * 0.05356121435761452
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_gain_ = None
        view_31 = mul_50.view(-1)
        mul_50 = None
        batch_norm_31 = torch.nn.functional.batch_norm(
            reshape_31,
            None,
            None,
            weight=view_31,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_31 = view_31 = None
        weight_31 = batch_norm_31.reshape_as(
            l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_weight_
        )
        batch_norm_31 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_weight_
        ) = None
        out_46 = torch.conv2d(
            out_45,
            weight_31,
            l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_45 = (
            weight_31
        ) = (
            l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_bias_
        ) = None
        relu_28 = torch.nn.functional.relu(out_46, inplace=True)
        out_46 = None
        reshape_32 = l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_51 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_gain_
            * 0.035707476238409676
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_gain_ = None
        view_32 = mul_51.view(-1)
        mul_51 = None
        batch_norm_32 = torch.nn.functional.batch_norm(
            reshape_32,
            None,
            None,
            weight=view_32,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_32 = view_32 = None
        weight_32 = batch_norm_32.reshape_as(
            l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_weight_
        )
        batch_norm_32 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_weight_
        ) = None
        out_47 = torch.conv2d(
            relu_28,
            weight_32,
            l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        relu_28 = (
            weight_32
        ) = (
            l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_bias_
        ) = None
        relu_29 = torch.nn.functional.relu(out_47, inplace=False)
        out_47 = None
        reshape_33 = l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_weight_.reshape(
            1, 1024, -1
        )
        mul_52 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_gain_
            * 0.10712242871522903
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_gain_ = None
        view_33 = mul_52.view(-1)
        mul_52 = None
        batch_norm_33 = torch.nn.functional.batch_norm(
            reshape_33,
            None,
            None,
            weight=view_33,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_33 = view_33 = None
        weight_33 = batch_norm_33.reshape_as(
            l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_weight_
        )
        batch_norm_33 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_weight_
        ) = None
        out_48 = torch.conv2d(
            relu_29,
            weight_33,
            l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        relu_29 = (
            weight_33
        ) = (
            l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_bias_
        ) = None
        mul_53 = out_48 * 0.2
        out_48 = None
        out_49 = mul_53 + out_44
        mul_53 = out_44 = None
        relu_30 = torch.nn.functional.relu(out_49, inplace=False)
        out_50 = relu_30 * 0.9449111825230679
        relu_30 = None
        reshape_34 = l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_55 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_gain_
            * 0.05356121435761452
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_gain_ = None
        view_34 = mul_55.view(-1)
        mul_55 = None
        batch_norm_34 = torch.nn.functional.batch_norm(
            reshape_34,
            None,
            None,
            weight=view_34,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_34 = view_34 = None
        weight_34 = batch_norm_34.reshape_as(
            l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_weight_
        )
        batch_norm_34 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_weight_
        ) = None
        out_51 = torch.conv2d(
            out_50,
            weight_34,
            l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_50 = (
            weight_34
        ) = (
            l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_bias_
        ) = None
        relu_31 = torch.nn.functional.relu(out_51, inplace=True)
        out_51 = None
        reshape_35 = l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_56 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_gain_
            * 0.035707476238409676
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_gain_ = None
        view_35 = mul_56.view(-1)
        mul_56 = None
        batch_norm_35 = torch.nn.functional.batch_norm(
            reshape_35,
            None,
            None,
            weight=view_35,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_35 = view_35 = None
        weight_35 = batch_norm_35.reshape_as(
            l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_weight_
        )
        batch_norm_35 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_weight_
        ) = None
        out_52 = torch.conv2d(
            relu_31,
            weight_35,
            l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        relu_31 = (
            weight_35
        ) = (
            l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_bias_
        ) = None
        relu_32 = torch.nn.functional.relu(out_52, inplace=False)
        out_52 = None
        reshape_36 = l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_weight_.reshape(
            1, 1024, -1
        )
        mul_57 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_gain_
            * 0.10712242871522903
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_gain_ = None
        view_36 = mul_57.view(-1)
        mul_57 = None
        batch_norm_36 = torch.nn.functional.batch_norm(
            reshape_36,
            None,
            None,
            weight=view_36,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_36 = view_36 = None
        weight_36 = batch_norm_36.reshape_as(
            l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_weight_
        )
        batch_norm_36 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_weight_
        ) = None
        out_53 = torch.conv2d(
            relu_32,
            weight_36,
            l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        relu_32 = (
            weight_36
        ) = (
            l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_bias_
        ) = None
        mul_58 = out_53 * 0.2
        out_53 = None
        out_54 = mul_58 + out_49
        mul_58 = out_49 = None
        relu_33 = torch.nn.functional.relu(out_54, inplace=False)
        out_55 = relu_33 * 0.9284766908852592
        relu_33 = None
        reshape_37 = l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_60 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_gain_
            * 0.05356121435761452
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_gain_ = None
        view_37 = mul_60.view(-1)
        mul_60 = None
        batch_norm_37 = torch.nn.functional.batch_norm(
            reshape_37,
            None,
            None,
            weight=view_37,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_37 = view_37 = None
        weight_37 = batch_norm_37.reshape_as(
            l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_weight_
        )
        batch_norm_37 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_weight_
        ) = None
        out_56 = torch.conv2d(
            out_55,
            weight_37,
            l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_55 = (
            weight_37
        ) = (
            l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_bias_
        ) = None
        relu_34 = torch.nn.functional.relu(out_56, inplace=True)
        out_56 = None
        reshape_38 = l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_61 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_gain_
            * 0.035707476238409676
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_gain_ = None
        view_38 = mul_61.view(-1)
        mul_61 = None
        batch_norm_38 = torch.nn.functional.batch_norm(
            reshape_38,
            None,
            None,
            weight=view_38,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_38 = view_38 = None
        weight_38 = batch_norm_38.reshape_as(
            l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_weight_
        )
        batch_norm_38 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_weight_
        ) = None
        out_57 = torch.conv2d(
            relu_34,
            weight_38,
            l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        relu_34 = (
            weight_38
        ) = (
            l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_bias_
        ) = None
        relu_35 = torch.nn.functional.relu(out_57, inplace=False)
        out_57 = None
        reshape_39 = l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_weight_.reshape(
            1, 1024, -1
        )
        mul_62 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_gain_
            * 0.10712242871522903
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_gain_ = None
        view_39 = mul_62.view(-1)
        mul_62 = None
        batch_norm_39 = torch.nn.functional.batch_norm(
            reshape_39,
            None,
            None,
            weight=view_39,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_39 = view_39 = None
        weight_39 = batch_norm_39.reshape_as(
            l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_weight_
        )
        batch_norm_39 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_weight_
        ) = None
        out_58 = torch.conv2d(
            relu_35,
            weight_39,
            l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        relu_35 = (
            weight_39
        ) = (
            l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_bias_
        ) = None
        mul_63 = out_58 * 0.2
        out_58 = None
        out_59 = mul_63 + out_54
        mul_63 = out_54 = None
        relu_36 = torch.nn.functional.relu(out_59, inplace=False)
        out_60 = relu_36 * 0.9128709291752768
        relu_36 = None
        reshape_40 = l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_65 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_gain_
            * 0.05356121435761452
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_gain_ = None
        view_40 = mul_65.view(-1)
        mul_65 = None
        batch_norm_40 = torch.nn.functional.batch_norm(
            reshape_40,
            None,
            None,
            weight=view_40,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_40 = view_40 = None
        weight_40 = batch_norm_40.reshape_as(
            l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_weight_
        )
        batch_norm_40 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_weight_
        ) = None
        out_61 = torch.conv2d(
            out_60,
            weight_40,
            l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_60 = (
            weight_40
        ) = (
            l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_bias_
        ) = None
        relu_37 = torch.nn.functional.relu(out_61, inplace=True)
        out_61 = None
        reshape_41 = l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_66 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_gain_
            * 0.035707476238409676
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_gain_ = None
        view_41 = mul_66.view(-1)
        mul_66 = None
        batch_norm_41 = torch.nn.functional.batch_norm(
            reshape_41,
            None,
            None,
            weight=view_41,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_41 = view_41 = None
        weight_41 = batch_norm_41.reshape_as(
            l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_weight_
        )
        batch_norm_41 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_weight_
        ) = None
        out_62 = torch.conv2d(
            relu_37,
            weight_41,
            l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        relu_37 = (
            weight_41
        ) = (
            l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_bias_
        ) = None
        relu_38 = torch.nn.functional.relu(out_62, inplace=False)
        out_62 = None
        reshape_42 = l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_weight_.reshape(
            1, 1024, -1
        )
        mul_67 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_gain_
            * 0.10712242871522903
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_gain_ = None
        view_42 = mul_67.view(-1)
        mul_67 = None
        batch_norm_42 = torch.nn.functional.batch_norm(
            reshape_42,
            None,
            None,
            weight=view_42,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_42 = view_42 = None
        weight_42 = batch_norm_42.reshape_as(
            l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_weight_
        )
        batch_norm_42 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_weight_
        ) = None
        out_63 = torch.conv2d(
            relu_38,
            weight_42,
            l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        relu_38 = (
            weight_42
        ) = (
            l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_bias_
        ) = None
        mul_68 = out_63 * 0.2
        out_63 = None
        out_64 = mul_68 + out_59
        mul_68 = out_59 = None
        relu_39 = torch.nn.functional.relu(out_64, inplace=False)
        out_64 = None
        out_65 = relu_39 * 0.8980265101338745
        relu_39 = None
        avg_pool2d_2 = torch._C._nn.avg_pool2d(out_65, 2, 2, 0, True, False, None)
        reshape_43 = l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_weight_.reshape(
            1, 2048, -1
        )
        mul_70 = (
            l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_gain_
            * 0.05356121435761452
        )
        l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_gain_ = (
            None
        )
        view_43 = mul_70.view(-1)
        mul_70 = None
        batch_norm_43 = torch.nn.functional.batch_norm(
            reshape_43,
            None,
            None,
            weight=view_43,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_43 = view_43 = None
        weight_43 = batch_norm_43.reshape_as(
            l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_weight_
        )
        batch_norm_43 = l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        shortcut_3 = torch.conv2d(
            avg_pool2d_2,
            weight_43,
            l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        avg_pool2d_2 = (
            weight_43
        ) = l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_bias_ = (None)
        reshape_44 = l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_71 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_gain_
            * 0.05356121435761452
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_gain_ = None
        view_44 = mul_71.view(-1)
        mul_71 = None
        batch_norm_44 = torch.nn.functional.batch_norm(
            reshape_44,
            None,
            None,
            weight=view_44,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_44 = view_44 = None
        weight_44 = batch_norm_44.reshape_as(
            l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_weight_
        )
        batch_norm_44 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_weight_
        ) = None
        out_66 = torch.conv2d(
            out_65,
            weight_44,
            l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_65 = (
            weight_44
        ) = (
            l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_bias_
        ) = None
        relu_40 = torch.nn.functional.relu(out_66, inplace=True)
        out_66 = None
        reshape_45 = l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_72 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_gain_
            * 0.025248998587236997
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_gain_ = None
        view_45 = mul_72.view(-1)
        mul_72 = None
        batch_norm_45 = torch.nn.functional.batch_norm(
            reshape_45,
            None,
            None,
            weight=view_45,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_45 = view_45 = None
        weight_45 = batch_norm_45.reshape_as(
            l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_weight_
        )
        batch_norm_45 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_weight_
        ) = None
        out_67 = torch.conv2d(
            relu_40,
            weight_45,
            l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        relu_40 = (
            weight_45
        ) = (
            l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_bias_
        ) = None
        relu_41 = torch.nn.functional.relu(out_67, inplace=False)
        out_67 = None
        reshape_46 = l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_weight_.reshape(
            1, 2048, -1
        )
        mul_73 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_gain_
            * 0.075746995761711
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_gain_ = None
        view_46 = mul_73.view(-1)
        mul_73 = None
        batch_norm_46 = torch.nn.functional.batch_norm(
            reshape_46,
            None,
            None,
            weight=view_46,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_46 = view_46 = None
        weight_46 = batch_norm_46.reshape_as(
            l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_weight_
        )
        batch_norm_46 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_weight_
        ) = None
        out_68 = torch.conv2d(
            relu_41,
            weight_46,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        relu_41 = (
            weight_46
        ) = (
            l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_bias_
        ) = None
        mul_74 = out_68 * 0.2
        out_68 = None
        out_69 = mul_74 + shortcut_3
        mul_74 = shortcut_3 = None
        relu_42 = torch.nn.functional.relu(out_69, inplace=False)
        out_70 = relu_42 * 0.9805806756909201
        relu_42 = None
        reshape_47 = l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_76 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_gain_
            * 0.0378734978808555
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_gain_ = None
        view_47 = mul_76.view(-1)
        mul_76 = None
        batch_norm_47 = torch.nn.functional.batch_norm(
            reshape_47,
            None,
            None,
            weight=view_47,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_47 = view_47 = None
        weight_47 = batch_norm_47.reshape_as(
            l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_weight_
        )
        batch_norm_47 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_weight_
        ) = None
        out_71 = torch.conv2d(
            out_70,
            weight_47,
            l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_70 = (
            weight_47
        ) = (
            l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_bias_
        ) = None
        relu_43 = torch.nn.functional.relu(out_71, inplace=True)
        out_71 = None
        reshape_48 = l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_77 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_gain_
            * 0.025248998587236997
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_gain_ = None
        view_48 = mul_77.view(-1)
        mul_77 = None
        batch_norm_48 = torch.nn.functional.batch_norm(
            reshape_48,
            None,
            None,
            weight=view_48,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_48 = view_48 = None
        weight_48 = batch_norm_48.reshape_as(
            l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_weight_
        )
        batch_norm_48 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_weight_
        ) = None
        out_72 = torch.conv2d(
            relu_43,
            weight_48,
            l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        relu_43 = (
            weight_48
        ) = (
            l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_bias_
        ) = None
        relu_44 = torch.nn.functional.relu(out_72, inplace=False)
        out_72 = None
        reshape_49 = l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_weight_.reshape(
            1, 2048, -1
        )
        mul_78 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_gain_
            * 0.075746995761711
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_gain_ = None
        view_49 = mul_78.view(-1)
        mul_78 = None
        batch_norm_49 = torch.nn.functional.batch_norm(
            reshape_49,
            None,
            None,
            weight=view_49,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_49 = view_49 = None
        weight_49 = batch_norm_49.reshape_as(
            l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_weight_
        )
        batch_norm_49 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_weight_
        ) = None
        out_73 = torch.conv2d(
            relu_44,
            weight_49,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        relu_44 = (
            weight_49
        ) = (
            l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_bias_
        ) = None
        mul_79 = out_73 * 0.2
        out_73 = None
        out_74 = mul_79 + out_69
        mul_79 = out_69 = None
        relu_45 = torch.nn.functional.relu(out_74, inplace=False)
        out_75 = relu_45 * 0.9622504486493761
        relu_45 = None
        reshape_50 = l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_81 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_gain_
            * 0.0378734978808555
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_gain_ = None
        view_50 = mul_81.view(-1)
        mul_81 = None
        batch_norm_50 = torch.nn.functional.batch_norm(
            reshape_50,
            None,
            None,
            weight=view_50,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_50 = view_50 = None
        weight_50 = batch_norm_50.reshape_as(
            l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_weight_
        )
        batch_norm_50 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_weight_
        ) = None
        out_76 = torch.conv2d(
            out_75,
            weight_50,
            l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_75 = (
            weight_50
        ) = (
            l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_bias_
        ) = None
        relu_46 = torch.nn.functional.relu(out_76, inplace=True)
        out_76 = None
        reshape_51 = l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_82 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_gain_
            * 0.025248998587236997
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_gain_ = None
        view_51 = mul_82.view(-1)
        mul_82 = None
        batch_norm_51 = torch.nn.functional.batch_norm(
            reshape_51,
            None,
            None,
            weight=view_51,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_51 = view_51 = None
        weight_51 = batch_norm_51.reshape_as(
            l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_weight_
        )
        batch_norm_51 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_weight_
        ) = None
        out_77 = torch.conv2d(
            relu_46,
            weight_51,
            l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        relu_46 = (
            weight_51
        ) = (
            l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_bias_
        ) = None
        relu_47 = torch.nn.functional.relu(out_77, inplace=False)
        out_77 = None
        reshape_52 = l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_weight_.reshape(
            1, 2048, -1
        )
        mul_83 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_gain_
            * 0.075746995761711
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_gain_ = None
        view_52 = mul_83.view(-1)
        mul_83 = None
        batch_norm_52 = torch.nn.functional.batch_norm(
            reshape_52,
            None,
            None,
            weight=view_52,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_52 = view_52 = None
        weight_52 = batch_norm_52.reshape_as(
            l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_weight_
        )
        batch_norm_52 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_weight_
        ) = None
        out_78 = torch.conv2d(
            relu_47,
            weight_52,
            l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        relu_47 = (
            weight_52
        ) = (
            l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_bias_
        ) = None
        mul_84 = out_78 * 0.2
        out_78 = None
        out_79 = mul_84 + out_74
        mul_84 = out_74 = None
        x = torch.nn.functional.relu(out_79, inplace=False)
        out_79 = None
        x_1 = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x = None
        x_2 = x_1.flatten(1, -1)
        x_1 = None
        x_3 = torch.nn.functional.dropout(x_2, 0.0, False, False)
        x_2 = None
        x_4 = torch._C._nn.linear(
            x_3,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_3 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_4,)
