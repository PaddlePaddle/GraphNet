import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv4_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_final_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_final_conv_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_final_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_conv1_parameters_weight_ = (
            L_self_modules_stem_modules_conv1_parameters_weight_
        )
        l_self_modules_stem_modules_conv1_parameters_gain_ = (
            L_self_modules_stem_modules_conv1_parameters_gain_
        )
        l_self_modules_stem_modules_conv1_parameters_bias_ = (
            L_self_modules_stem_modules_conv1_parameters_bias_
        )
        l_x_ = L_x_
        l_self_modules_stem_modules_conv2_parameters_weight_ = (
            L_self_modules_stem_modules_conv2_parameters_weight_
        )
        l_self_modules_stem_modules_conv2_parameters_gain_ = (
            L_self_modules_stem_modules_conv2_parameters_gain_
        )
        l_self_modules_stem_modules_conv2_parameters_bias_ = (
            L_self_modules_stem_modules_conv2_parameters_bias_
        )
        l_self_modules_stem_modules_conv3_parameters_weight_ = (
            L_self_modules_stem_modules_conv3_parameters_weight_
        )
        l_self_modules_stem_modules_conv3_parameters_gain_ = (
            L_self_modules_stem_modules_conv3_parameters_gain_
        )
        l_self_modules_stem_modules_conv3_parameters_bias_ = (
            L_self_modules_stem_modules_conv3_parameters_bias_
        )
        l_self_modules_stem_modules_conv4_parameters_weight_ = (
            L_self_modules_stem_modules_conv4_parameters_weight_
        )
        l_self_modules_stem_modules_conv4_parameters_gain_ = (
            L_self_modules_stem_modules_conv4_parameters_gain_
        )
        l_self_modules_stem_modules_conv4_parameters_bias_ = (
            L_self_modules_stem_modules_conv4_parameters_bias_
        )
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
        l_self_modules_stages_modules_0_modules_0_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_0_modules_0_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_0_modules_0_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_0_modules_conv2b_parameters_bias_
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
        l_self_modules_stages_modules_0_modules_0_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_attn_last_modules_conv_parameters_weight_
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
        l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_conv_parameters_weight_
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
        l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_conv_parameters_weight_
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
        l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_conv_parameters_weight_
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
        l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_conv_parameters_weight_
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
        l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_conv_parameters_weight_
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
        l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_conv_parameters_weight_
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
        l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_conv_parameters_weight_
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
        l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_conv_parameters_weight_
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
        l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_bias_
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
        l_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_conv_parameters_weight_
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
        l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_bias_
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
        l_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_conv_parameters_weight_
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
        l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_bias_
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
        l_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_conv_parameters_weight_
        l_self_modules_final_conv_parameters_weight_ = (
            L_self_modules_final_conv_parameters_weight_
        )
        l_self_modules_final_conv_parameters_gain_ = (
            L_self_modules_final_conv_parameters_gain_
        )
        l_self_modules_final_conv_parameters_bias_ = (
            L_self_modules_final_conv_parameters_bias_
        )
        l_self_modules_head_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_fc_parameters_bias_
        )
        reshape = l_self_modules_stem_modules_conv1_parameters_weight_.reshape(
            1, 16, -1
        )
        mul = l_self_modules_stem_modules_conv1_parameters_gain_ * 0.34412564994580647
        l_self_modules_stem_modules_conv1_parameters_gain_ = None
        view = mul.view(-1)
        mul = None
        batch_norm = torch.nn.functional.batch_norm(
            reshape, None, None, weight=view, training=True, momentum=0.0, eps=1e-05
        )
        reshape = view = None
        weight = batch_norm.reshape_as(
            l_self_modules_stem_modules_conv1_parameters_weight_
        )
        batch_norm = l_self_modules_stem_modules_conv1_parameters_weight_ = None
        input_1 = torch.conv2d(
            l_x_,
            weight,
            l_self_modules_stem_modules_conv1_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = weight = l_self_modules_stem_modules_conv1_parameters_bias_ = None
        input_2 = torch.nn.functional.silu(input_1, inplace=True)
        input_1 = None
        reshape_1 = l_self_modules_stem_modules_conv2_parameters_weight_.reshape(
            1, 32, -1
        )
        mul_1 = l_self_modules_stem_modules_conv2_parameters_gain_ * 0.1490107774734497
        l_self_modules_stem_modules_conv2_parameters_gain_ = None
        view_1 = mul_1.view(-1)
        mul_1 = None
        batch_norm_1 = torch.nn.functional.batch_norm(
            reshape_1, None, None, weight=view_1, training=True, momentum=0.0, eps=1e-05
        )
        reshape_1 = view_1 = None
        weight_1 = batch_norm_1.reshape_as(
            l_self_modules_stem_modules_conv2_parameters_weight_
        )
        batch_norm_1 = l_self_modules_stem_modules_conv2_parameters_weight_ = None
        input_3 = torch.conv2d(
            input_2,
            weight_1,
            l_self_modules_stem_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_2 = weight_1 = l_self_modules_stem_modules_conv2_parameters_bias_ = None
        input_4 = torch.nn.functional.silu(input_3, inplace=True)
        input_3 = None
        reshape_2 = l_self_modules_stem_modules_conv3_parameters_weight_.reshape(
            1, 64, -1
        )
        mul_2 = l_self_modules_stem_modules_conv3_parameters_gain_ * 0.10536653122135592
        l_self_modules_stem_modules_conv3_parameters_gain_ = None
        view_2 = mul_2.view(-1)
        mul_2 = None
        batch_norm_2 = torch.nn.functional.batch_norm(
            reshape_2, None, None, weight=view_2, training=True, momentum=0.0, eps=1e-05
        )
        reshape_2 = view_2 = None
        weight_2 = batch_norm_2.reshape_as(
            l_self_modules_stem_modules_conv3_parameters_weight_
        )
        batch_norm_2 = l_self_modules_stem_modules_conv3_parameters_weight_ = None
        input_5 = torch.conv2d(
            input_4,
            weight_2,
            l_self_modules_stem_modules_conv3_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_4 = weight_2 = l_self_modules_stem_modules_conv3_parameters_bias_ = None
        input_6 = torch.nn.functional.silu(input_5, inplace=True)
        input_5 = None
        reshape_3 = l_self_modules_stem_modules_conv4_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_3 = l_self_modules_stem_modules_conv4_parameters_gain_ * 0.07450538873672485
        l_self_modules_stem_modules_conv4_parameters_gain_ = None
        view_3 = mul_3.view(-1)
        mul_3 = None
        batch_norm_3 = torch.nn.functional.batch_norm(
            reshape_3, None, None, weight=view_3, training=True, momentum=0.0, eps=1e-05
        )
        reshape_3 = view_3 = None
        weight_3 = batch_norm_3.reshape_as(
            l_self_modules_stem_modules_conv4_parameters_weight_
        )
        batch_norm_3 = l_self_modules_stem_modules_conv4_parameters_weight_ = None
        input_7 = torch.conv2d(
            input_6,
            weight_3,
            l_self_modules_stem_modules_conv4_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_6 = weight_3 = l_self_modules_stem_modules_conv4_parameters_bias_ = None
        silu_3 = torch.nn.functional.silu(input_7, inplace=False)
        input_7 = None
        out = silu_3 * 1.0
        silu_3 = None
        reshape_4 = l_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_5 = (
            l_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_gain_
            * 0.1580497968320339
        )
        l_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_gain_ = (
            None
        )
        view_4 = mul_5.view(-1)
        mul_5 = None
        batch_norm_4 = torch.nn.functional.batch_norm(
            reshape_4, None, None, weight=view_4, training=True, momentum=0.0, eps=1e-05
        )
        reshape_4 = view_4 = None
        weight_4 = batch_norm_4.reshape_as(
            l_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_weight_
        )
        batch_norm_4 = l_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        shortcut = torch.conv2d(
            out,
            weight_4,
            l_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        weight_4 = l_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_bias_ = (None)
        reshape_5 = l_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_weight_.reshape(
            1, 64, -1
        )
        mul_6 = (
            l_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_gain_
            * 0.1580497968320339
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_gain_ = None
        view_5 = mul_6.view(-1)
        mul_6 = None
        batch_norm_5 = torch.nn.functional.batch_norm(
            reshape_5, None, None, weight=view_5, training=True, momentum=0.0, eps=1e-05
        )
        reshape_5 = view_5 = None
        weight_5 = batch_norm_5.reshape_as(
            l_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_weight_
        )
        batch_norm_5 = (
            l_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_weight_
        ) = None
        out_1 = torch.conv2d(
            out,
            weight_5,
            l_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out = (
            weight_5
        ) = (
            l_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_bias_
        ) = None
        silu_4 = torch.nn.functional.silu(out_1, inplace=True)
        out_1 = None
        reshape_6 = l_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_weight_.reshape(
            1, 64, -1
        )
        mul_7 = (
            l_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_gain_ = None
        view_6 = mul_7.view(-1)
        mul_7 = None
        batch_norm_6 = torch.nn.functional.batch_norm(
            reshape_6, None, None, weight=view_6, training=True, momentum=0.0, eps=1e-05
        )
        reshape_6 = view_6 = None
        weight_6 = batch_norm_6.reshape_as(
            l_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_weight_
        )
        batch_norm_6 = (
            l_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_weight_
        ) = None
        out_2 = torch.conv2d(
            silu_4,
            weight_6,
            l_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_4 = (
            weight_6
        ) = (
            l_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_bias_
        ) = None
        silu_5 = torch.nn.functional.silu(out_2, inplace=True)
        out_2 = None
        reshape_7 = l_self_modules_stages_modules_0_modules_0_modules_conv2b_parameters_weight_.reshape(
            1, 64, -1
        )
        mul_8 = (
            l_self_modules_stages_modules_0_modules_0_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv2b_parameters_gain_ = None
        view_7 = mul_8.view(-1)
        mul_8 = None
        batch_norm_7 = torch.nn.functional.batch_norm(
            reshape_7, None, None, weight=view_7, training=True, momentum=0.0, eps=1e-05
        )
        reshape_7 = view_7 = None
        weight_7 = batch_norm_7.reshape_as(
            l_self_modules_stages_modules_0_modules_0_modules_conv2b_parameters_weight_
        )
        batch_norm_7 = (
            l_self_modules_stages_modules_0_modules_0_modules_conv2b_parameters_weight_
        ) = None
        out_3 = torch.conv2d(
            silu_5,
            weight_7,
            l_self_modules_stages_modules_0_modules_0_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_5 = (
            weight_7
        ) = (
            l_self_modules_stages_modules_0_modules_0_modules_conv2b_parameters_bias_
        ) = None
        silu_6 = torch.nn.functional.silu(out_3, inplace=False)
        out_3 = None
        reshape_8 = l_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_9 = (
            l_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_gain_
            * 0.22351616621017456
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_gain_ = None
        view_8 = mul_9.view(-1)
        mul_9 = None
        batch_norm_8 = torch.nn.functional.batch_norm(
            reshape_8, None, None, weight=view_8, training=True, momentum=0.0, eps=1e-05
        )
        reshape_8 = view_8 = None
        weight_8 = batch_norm_8.reshape_as(
            l_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_weight_
        )
        batch_norm_8 = (
            l_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_weight_
        ) = None
        out_4 = torch.conv2d(
            silu_6,
            weight_8,
            l_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_6 = (
            weight_8
        ) = (
            l_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_bias_
        ) = None
        mean = out_4.mean((2, 3))
        y = mean.view(1, 1, -1)
        mean = None
        y_1 = torch.conv1d(
            y,
            l_self_modules_stages_modules_0_modules_0_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y = l_self_modules_stages_modules_0_modules_0_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid = y_1.sigmoid()
        y_1 = None
        y_2 = sigmoid.view(1, -1, 1, 1)
        sigmoid = None
        expand_as = y_2.expand_as(out_4)
        y_2 = None
        mul_10 = out_4 * expand_as
        out_4 = expand_as = None
        out_5 = 2.0 * mul_10
        mul_10 = None
        mul_12 = out_5 * 0.2
        out_5 = None
        out_6 = mul_12 + shortcut
        mul_12 = shortcut = None
        silu_7 = torch.nn.functional.silu(out_6, inplace=False)
        out_6 = None
        out_7 = silu_7 * 0.9805806756909201
        silu_7 = None
        avg_pool2d = torch._C._nn.avg_pool2d(out_7, 2, 2, 0, True, False, None)
        reshape_9 = l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_14 = (
            l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_gain_
            * 0.11175808310508728
        )
        l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_gain_ = (
            None
        )
        view_11 = mul_14.view(-1)
        mul_14 = None
        batch_norm_9 = torch.nn.functional.batch_norm(
            reshape_9,
            None,
            None,
            weight=view_11,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_9 = view_11 = None
        weight_9 = batch_norm_9.reshape_as(
            l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_
        )
        batch_norm_9 = l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        shortcut_1 = torch.conv2d(
            avg_pool2d,
            weight_9,
            l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        avg_pool2d = (
            weight_9
        ) = l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_bias_ = (None)
        reshape_10 = l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_15 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_gain_
            * 0.11175808310508728
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_gain_ = None
        view_12 = mul_15.view(-1)
        mul_15 = None
        batch_norm_10 = torch.nn.functional.batch_norm(
            reshape_10,
            None,
            None,
            weight=view_12,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_10 = view_12 = None
        weight_10 = batch_norm_10.reshape_as(
            l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_weight_
        )
        batch_norm_10 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_weight_
        ) = None
        out_8 = torch.conv2d(
            out_7,
            weight_10,
            l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_7 = (
            weight_10
        ) = (
            l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_bias_
        ) = None
        silu_8 = torch.nn.functional.silu(out_8, inplace=True)
        out_8 = None
        reshape_11 = l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_16 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_gain_ = None
        view_13 = mul_16.view(-1)
        mul_16 = None
        batch_norm_11 = torch.nn.functional.batch_norm(
            reshape_11,
            None,
            None,
            weight=view_13,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_11 = view_13 = None
        weight_11 = batch_norm_11.reshape_as(
            l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_weight_
        )
        batch_norm_11 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_weight_
        ) = None
        out_9 = torch.conv2d(
            silu_8,
            weight_11,
            l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            2,
        )
        silu_8 = (
            weight_11
        ) = (
            l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_bias_
        ) = None
        silu_9 = torch.nn.functional.silu(out_9, inplace=True)
        out_9 = None
        reshape_12 = l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_17 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_gain_ = None
        view_14 = mul_17.view(-1)
        mul_17 = None
        batch_norm_12 = torch.nn.functional.batch_norm(
            reshape_12,
            None,
            None,
            weight=view_14,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_12 = view_14 = None
        weight_12 = batch_norm_12.reshape_as(
            l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_weight_
        )
        batch_norm_12 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_weight_
        ) = None
        out_10 = torch.conv2d(
            silu_9,
            weight_12,
            l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        silu_9 = (
            weight_12
        ) = (
            l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_bias_
        ) = None
        silu_10 = torch.nn.functional.silu(out_10, inplace=False)
        out_10 = None
        reshape_13 = l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_18 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_gain_
            * 0.1580497968320339
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_gain_ = None
        view_15 = mul_18.view(-1)
        mul_18 = None
        batch_norm_13 = torch.nn.functional.batch_norm(
            reshape_13,
            None,
            None,
            weight=view_15,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_13 = view_15 = None
        weight_13 = batch_norm_13.reshape_as(
            l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_weight_
        )
        batch_norm_13 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_weight_
        ) = None
        out_11 = torch.conv2d(
            silu_10,
            weight_13,
            l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_10 = (
            weight_13
        ) = (
            l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_bias_
        ) = None
        mean_1 = out_11.mean((2, 3))
        y_3 = mean_1.view(1, 1, -1)
        mean_1 = None
        y_4 = torch.conv1d(
            y_3,
            l_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_3 = l_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_1 = y_4.sigmoid()
        y_4 = None
        y_5 = sigmoid_1.view(1, -1, 1, 1)
        sigmoid_1 = None
        expand_as_1 = y_5.expand_as(out_11)
        y_5 = None
        mul_19 = out_11 * expand_as_1
        out_11 = expand_as_1 = None
        out_12 = 2.0 * mul_19
        mul_19 = None
        mul_21 = out_12 * 0.2
        out_12 = None
        out_13 = mul_21 + shortcut_1
        mul_21 = shortcut_1 = None
        silu_11 = torch.nn.functional.silu(out_13, inplace=False)
        out_14 = silu_11 * 0.9805806756909201
        silu_11 = None
        reshape_14 = l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_23 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_gain_
            * 0.07902489841601695
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_gain_ = None
        view_18 = mul_23.view(-1)
        mul_23 = None
        batch_norm_14 = torch.nn.functional.batch_norm(
            reshape_14,
            None,
            None,
            weight=view_18,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_14 = view_18 = None
        weight_14 = batch_norm_14.reshape_as(
            l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_weight_
        )
        batch_norm_14 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_weight_
        ) = None
        out_15 = torch.conv2d(
            out_14,
            weight_14,
            l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_14 = (
            weight_14
        ) = (
            l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_bias_
        ) = None
        silu_12 = torch.nn.functional.silu(out_15, inplace=True)
        out_15 = None
        reshape_15 = l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_24 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_gain_ = None
        view_19 = mul_24.view(-1)
        mul_24 = None
        batch_norm_15 = torch.nn.functional.batch_norm(
            reshape_15,
            None,
            None,
            weight=view_19,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_15 = view_19 = None
        weight_15 = batch_norm_15.reshape_as(
            l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_weight_
        )
        batch_norm_15 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_weight_
        ) = None
        out_16 = torch.conv2d(
            silu_12,
            weight_15,
            l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        silu_12 = (
            weight_15
        ) = (
            l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_bias_
        ) = None
        silu_13 = torch.nn.functional.silu(out_16, inplace=True)
        out_16 = None
        reshape_16 = l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_25 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_gain_ = None
        view_20 = mul_25.view(-1)
        mul_25 = None
        batch_norm_16 = torch.nn.functional.batch_norm(
            reshape_16,
            None,
            None,
            weight=view_20,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_16 = view_20 = None
        weight_16 = batch_norm_16.reshape_as(
            l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_weight_
        )
        batch_norm_16 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_weight_
        ) = None
        out_17 = torch.conv2d(
            silu_13,
            weight_16,
            l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        silu_13 = (
            weight_16
        ) = (
            l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_bias_
        ) = None
        silu_14 = torch.nn.functional.silu(out_17, inplace=False)
        out_17 = None
        reshape_17 = l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_26 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_gain_
            * 0.1580497968320339
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_gain_ = None
        view_21 = mul_26.view(-1)
        mul_26 = None
        batch_norm_17 = torch.nn.functional.batch_norm(
            reshape_17,
            None,
            None,
            weight=view_21,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_17 = view_21 = None
        weight_17 = batch_norm_17.reshape_as(
            l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_weight_
        )
        batch_norm_17 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_weight_
        ) = None
        out_18 = torch.conv2d(
            silu_14,
            weight_17,
            l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_14 = (
            weight_17
        ) = (
            l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_bias_
        ) = None
        mean_2 = out_18.mean((2, 3))
        y_6 = mean_2.view(1, 1, -1)
        mean_2 = None
        y_7 = torch.conv1d(
            y_6,
            l_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_6 = l_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_2 = y_7.sigmoid()
        y_7 = None
        y_8 = sigmoid_2.view(1, -1, 1, 1)
        sigmoid_2 = None
        expand_as_2 = y_8.expand_as(out_18)
        y_8 = None
        mul_27 = out_18 * expand_as_2
        out_18 = expand_as_2 = None
        out_19 = 2.0 * mul_27
        mul_27 = None
        mul_29 = out_19 * 0.2
        out_19 = None
        out_20 = mul_29 + out_13
        mul_29 = out_13 = None
        silu_15 = torch.nn.functional.silu(out_20, inplace=False)
        out_20 = None
        out_21 = silu_15 * 0.9622504486493761
        silu_15 = None
        avg_pool2d_1 = torch._C._nn.avg_pool2d(out_21, 2, 2, 0, True, False, None)
        reshape_18 = l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_31 = (
            l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_gain_
            * 0.07902489841601695
        )
        l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_gain_ = (
            None
        )
        view_24 = mul_31.view(-1)
        mul_31 = None
        batch_norm_18 = torch.nn.functional.batch_norm(
            reshape_18,
            None,
            None,
            weight=view_24,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_18 = view_24 = None
        weight_18 = batch_norm_18.reshape_as(
            l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_
        )
        batch_norm_18 = l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        shortcut_2 = torch.conv2d(
            avg_pool2d_1,
            weight_18,
            l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        avg_pool2d_1 = (
            weight_18
        ) = l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_bias_ = (None)
        reshape_19 = l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_32 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_gain_
            * 0.07902489841601695
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_gain_ = None
        view_25 = mul_32.view(-1)
        mul_32 = None
        batch_norm_19 = torch.nn.functional.batch_norm(
            reshape_19,
            None,
            None,
            weight=view_25,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_19 = view_25 = None
        weight_19 = batch_norm_19.reshape_as(
            l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_weight_
        )
        batch_norm_19 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_weight_
        ) = None
        out_22 = torch.conv2d(
            out_21,
            weight_19,
            l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_21 = (
            weight_19
        ) = (
            l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_bias_
        ) = None
        silu_16 = torch.nn.functional.silu(out_22, inplace=True)
        out_22 = None
        reshape_20 = l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_33 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_gain_ = None
        view_26 = mul_33.view(-1)
        mul_33 = None
        batch_norm_20 = torch.nn.functional.batch_norm(
            reshape_20,
            None,
            None,
            weight=view_26,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_20 = view_26 = None
        weight_20 = batch_norm_20.reshape_as(
            l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_weight_
        )
        batch_norm_20 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_weight_
        ) = None
        out_23 = torch.conv2d(
            silu_16,
            weight_20,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            6,
        )
        silu_16 = (
            weight_20
        ) = (
            l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_bias_
        ) = None
        silu_17 = torch.nn.functional.silu(out_23, inplace=True)
        out_23 = None
        reshape_21 = l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_34 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_gain_ = None
        view_27 = mul_34.view(-1)
        mul_34 = None
        batch_norm_21 = torch.nn.functional.batch_norm(
            reshape_21,
            None,
            None,
            weight=view_27,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_21 = view_27 = None
        weight_21 = batch_norm_21.reshape_as(
            l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_weight_
        )
        batch_norm_21 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_weight_
        ) = None
        out_24 = torch.conv2d(
            silu_17,
            weight_21,
            l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_17 = (
            weight_21
        ) = (
            l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_bias_
        ) = None
        silu_18 = torch.nn.functional.silu(out_24, inplace=False)
        out_24 = None
        reshape_22 = l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_35 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_gain_ = None
        view_28 = mul_35.view(-1)
        mul_35 = None
        batch_norm_22 = torch.nn.functional.batch_norm(
            reshape_22,
            None,
            None,
            weight=view_28,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_22 = view_28 = None
        weight_22 = batch_norm_22.reshape_as(
            l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_weight_
        )
        batch_norm_22 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_weight_
        ) = None
        out_25 = torch.conv2d(
            silu_18,
            weight_22,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_18 = (
            weight_22
        ) = (
            l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_bias_
        ) = None
        mean_3 = out_25.mean((2, 3))
        y_9 = mean_3.view(1, 1, -1)
        mean_3 = None
        y_10 = torch.conv1d(
            y_9,
            l_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_9 = l_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_3 = y_10.sigmoid()
        y_10 = None
        y_11 = sigmoid_3.view(1, -1, 1, 1)
        sigmoid_3 = None
        expand_as_3 = y_11.expand_as(out_25)
        y_11 = None
        mul_36 = out_25 * expand_as_3
        out_25 = expand_as_3 = None
        out_26 = 2.0 * mul_36
        mul_36 = None
        mul_38 = out_26 * 0.2
        out_26 = None
        out_27 = mul_38 + shortcut_2
        mul_38 = shortcut_2 = None
        silu_19 = torch.nn.functional.silu(out_27, inplace=False)
        out_28 = silu_19 * 0.9805806756909201
        silu_19 = None
        reshape_23 = l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_40 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_gain_ = None
        view_31 = mul_40.view(-1)
        mul_40 = None
        batch_norm_23 = torch.nn.functional.batch_norm(
            reshape_23,
            None,
            None,
            weight=view_31,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_23 = view_31 = None
        weight_23 = batch_norm_23.reshape_as(
            l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_weight_
        )
        batch_norm_23 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_weight_
        ) = None
        out_29 = torch.conv2d(
            out_28,
            weight_23,
            l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_28 = (
            weight_23
        ) = (
            l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_bias_
        ) = None
        silu_20 = torch.nn.functional.silu(out_29, inplace=True)
        out_29 = None
        reshape_24 = l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_41 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_gain_ = None
        view_32 = mul_41.view(-1)
        mul_41 = None
        batch_norm_24 = torch.nn.functional.batch_norm(
            reshape_24,
            None,
            None,
            weight=view_32,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_24 = view_32 = None
        weight_24 = batch_norm_24.reshape_as(
            l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_weight_
        )
        batch_norm_24 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_weight_
        ) = None
        out_30 = torch.conv2d(
            silu_20,
            weight_24,
            l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_20 = (
            weight_24
        ) = (
            l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_bias_
        ) = None
        silu_21 = torch.nn.functional.silu(out_30, inplace=True)
        out_30 = None
        reshape_25 = l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_42 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_gain_ = None
        view_33 = mul_42.view(-1)
        mul_42 = None
        batch_norm_25 = torch.nn.functional.batch_norm(
            reshape_25,
            None,
            None,
            weight=view_33,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_25 = view_33 = None
        weight_25 = batch_norm_25.reshape_as(
            l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_weight_
        )
        batch_norm_25 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_weight_
        ) = None
        out_31 = torch.conv2d(
            silu_21,
            weight_25,
            l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_21 = (
            weight_25
        ) = (
            l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_bias_
        ) = None
        silu_22 = torch.nn.functional.silu(out_31, inplace=False)
        out_31 = None
        reshape_26 = l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_43 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_gain_ = None
        view_34 = mul_43.view(-1)
        mul_43 = None
        batch_norm_26 = torch.nn.functional.batch_norm(
            reshape_26,
            None,
            None,
            weight=view_34,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_26 = view_34 = None
        weight_26 = batch_norm_26.reshape_as(
            l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_weight_
        )
        batch_norm_26 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_weight_
        ) = None
        out_32 = torch.conv2d(
            silu_22,
            weight_26,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_22 = (
            weight_26
        ) = (
            l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_bias_
        ) = None
        mean_4 = out_32.mean((2, 3))
        y_12 = mean_4.view(1, 1, -1)
        mean_4 = None
        y_13 = torch.conv1d(
            y_12,
            l_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_12 = l_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_4 = y_13.sigmoid()
        y_13 = None
        y_14 = sigmoid_4.view(1, -1, 1, 1)
        sigmoid_4 = None
        expand_as_4 = y_14.expand_as(out_32)
        y_14 = None
        mul_44 = out_32 * expand_as_4
        out_32 = expand_as_4 = None
        out_33 = 2.0 * mul_44
        mul_44 = None
        mul_46 = out_33 * 0.2
        out_33 = None
        out_34 = mul_46 + out_27
        mul_46 = out_27 = None
        silu_23 = torch.nn.functional.silu(out_34, inplace=False)
        out_35 = silu_23 * 0.9622504486493761
        silu_23 = None
        reshape_27 = l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_48 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_gain_ = None
        view_37 = mul_48.view(-1)
        mul_48 = None
        batch_norm_27 = torch.nn.functional.batch_norm(
            reshape_27,
            None,
            None,
            weight=view_37,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_27 = view_37 = None
        weight_27 = batch_norm_27.reshape_as(
            l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_weight_
        )
        batch_norm_27 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_weight_
        ) = None
        out_36 = torch.conv2d(
            out_35,
            weight_27,
            l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_35 = (
            weight_27
        ) = (
            l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_bias_
        ) = None
        silu_24 = torch.nn.functional.silu(out_36, inplace=True)
        out_36 = None
        reshape_28 = l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_49 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_gain_ = None
        view_38 = mul_49.view(-1)
        mul_49 = None
        batch_norm_28 = torch.nn.functional.batch_norm(
            reshape_28,
            None,
            None,
            weight=view_38,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_28 = view_38 = None
        weight_28 = batch_norm_28.reshape_as(
            l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_weight_
        )
        batch_norm_28 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_weight_
        ) = None
        out_37 = torch.conv2d(
            silu_24,
            weight_28,
            l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_24 = (
            weight_28
        ) = (
            l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_bias_
        ) = None
        silu_25 = torch.nn.functional.silu(out_37, inplace=True)
        out_37 = None
        reshape_29 = l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_50 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_gain_ = None
        view_39 = mul_50.view(-1)
        mul_50 = None
        batch_norm_29 = torch.nn.functional.batch_norm(
            reshape_29,
            None,
            None,
            weight=view_39,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_29 = view_39 = None
        weight_29 = batch_norm_29.reshape_as(
            l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_weight_
        )
        batch_norm_29 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_weight_
        ) = None
        out_38 = torch.conv2d(
            silu_25,
            weight_29,
            l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_25 = (
            weight_29
        ) = (
            l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_bias_
        ) = None
        silu_26 = torch.nn.functional.silu(out_38, inplace=False)
        out_38 = None
        reshape_30 = l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_51 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_gain_ = None
        view_40 = mul_51.view(-1)
        mul_51 = None
        batch_norm_30 = torch.nn.functional.batch_norm(
            reshape_30,
            None,
            None,
            weight=view_40,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_30 = view_40 = None
        weight_30 = batch_norm_30.reshape_as(
            l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_weight_
        )
        batch_norm_30 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_weight_
        ) = None
        out_39 = torch.conv2d(
            silu_26,
            weight_30,
            l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_26 = (
            weight_30
        ) = (
            l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_bias_
        ) = None
        mean_5 = out_39.mean((2, 3))
        y_15 = mean_5.view(1, 1, -1)
        mean_5 = None
        y_16 = torch.conv1d(
            y_15,
            l_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_15 = l_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_5 = y_16.sigmoid()
        y_16 = None
        y_17 = sigmoid_5.view(1, -1, 1, 1)
        sigmoid_5 = None
        expand_as_5 = y_17.expand_as(out_39)
        y_17 = None
        mul_52 = out_39 * expand_as_5
        out_39 = expand_as_5 = None
        out_40 = 2.0 * mul_52
        mul_52 = None
        mul_54 = out_40 * 0.2
        out_40 = None
        out_41 = mul_54 + out_34
        mul_54 = out_34 = None
        silu_27 = torch.nn.functional.silu(out_41, inplace=False)
        out_42 = silu_27 * 0.9449111825230679
        silu_27 = None
        reshape_31 = l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_56 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_gain_ = None
        view_43 = mul_56.view(-1)
        mul_56 = None
        batch_norm_31 = torch.nn.functional.batch_norm(
            reshape_31,
            None,
            None,
            weight=view_43,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_31 = view_43 = None
        weight_31 = batch_norm_31.reshape_as(
            l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_weight_
        )
        batch_norm_31 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_weight_
        ) = None
        out_43 = torch.conv2d(
            out_42,
            weight_31,
            l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_42 = (
            weight_31
        ) = (
            l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_bias_
        ) = None
        silu_28 = torch.nn.functional.silu(out_43, inplace=True)
        out_43 = None
        reshape_32 = l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_57 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_gain_ = None
        view_44 = mul_57.view(-1)
        mul_57 = None
        batch_norm_32 = torch.nn.functional.batch_norm(
            reshape_32,
            None,
            None,
            weight=view_44,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_32 = view_44 = None
        weight_32 = batch_norm_32.reshape_as(
            l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_weight_
        )
        batch_norm_32 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_weight_
        ) = None
        out_44 = torch.conv2d(
            silu_28,
            weight_32,
            l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_28 = (
            weight_32
        ) = (
            l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_bias_
        ) = None
        silu_29 = torch.nn.functional.silu(out_44, inplace=True)
        out_44 = None
        reshape_33 = l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_58 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_gain_ = None
        view_45 = mul_58.view(-1)
        mul_58 = None
        batch_norm_33 = torch.nn.functional.batch_norm(
            reshape_33,
            None,
            None,
            weight=view_45,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_33 = view_45 = None
        weight_33 = batch_norm_33.reshape_as(
            l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_weight_
        )
        batch_norm_33 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_weight_
        ) = None
        out_45 = torch.conv2d(
            silu_29,
            weight_33,
            l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_29 = (
            weight_33
        ) = (
            l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_bias_
        ) = None
        silu_30 = torch.nn.functional.silu(out_45, inplace=False)
        out_45 = None
        reshape_34 = l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_59 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_gain_ = None
        view_46 = mul_59.view(-1)
        mul_59 = None
        batch_norm_34 = torch.nn.functional.batch_norm(
            reshape_34,
            None,
            None,
            weight=view_46,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_34 = view_46 = None
        weight_34 = batch_norm_34.reshape_as(
            l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_weight_
        )
        batch_norm_34 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_weight_
        ) = None
        out_46 = torch.conv2d(
            silu_30,
            weight_34,
            l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_30 = (
            weight_34
        ) = (
            l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_bias_
        ) = None
        mean_6 = out_46.mean((2, 3))
        y_18 = mean_6.view(1, 1, -1)
        mean_6 = None
        y_19 = torch.conv1d(
            y_18,
            l_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_18 = l_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_6 = y_19.sigmoid()
        y_19 = None
        y_20 = sigmoid_6.view(1, -1, 1, 1)
        sigmoid_6 = None
        expand_as_6 = y_20.expand_as(out_46)
        y_20 = None
        mul_60 = out_46 * expand_as_6
        out_46 = expand_as_6 = None
        out_47 = 2.0 * mul_60
        mul_60 = None
        mul_62 = out_47 * 0.2
        out_47 = None
        out_48 = mul_62 + out_41
        mul_62 = out_41 = None
        silu_31 = torch.nn.functional.silu(out_48, inplace=False)
        out_49 = silu_31 * 0.9284766908852592
        silu_31 = None
        reshape_35 = l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_64 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_gain_ = None
        view_49 = mul_64.view(-1)
        mul_64 = None
        batch_norm_35 = torch.nn.functional.batch_norm(
            reshape_35,
            None,
            None,
            weight=view_49,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_35 = view_49 = None
        weight_35 = batch_norm_35.reshape_as(
            l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_weight_
        )
        batch_norm_35 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_weight_
        ) = None
        out_50 = torch.conv2d(
            out_49,
            weight_35,
            l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_49 = (
            weight_35
        ) = (
            l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_bias_
        ) = None
        silu_32 = torch.nn.functional.silu(out_50, inplace=True)
        out_50 = None
        reshape_36 = l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_65 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_gain_ = None
        view_50 = mul_65.view(-1)
        mul_65 = None
        batch_norm_36 = torch.nn.functional.batch_norm(
            reshape_36,
            None,
            None,
            weight=view_50,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_36 = view_50 = None
        weight_36 = batch_norm_36.reshape_as(
            l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_weight_
        )
        batch_norm_36 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_weight_
        ) = None
        out_51 = torch.conv2d(
            silu_32,
            weight_36,
            l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_32 = (
            weight_36
        ) = (
            l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_bias_
        ) = None
        silu_33 = torch.nn.functional.silu(out_51, inplace=True)
        out_51 = None
        reshape_37 = l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_66 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_gain_ = None
        view_51 = mul_66.view(-1)
        mul_66 = None
        batch_norm_37 = torch.nn.functional.batch_norm(
            reshape_37,
            None,
            None,
            weight=view_51,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_37 = view_51 = None
        weight_37 = batch_norm_37.reshape_as(
            l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_weight_
        )
        batch_norm_37 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_weight_
        ) = None
        out_52 = torch.conv2d(
            silu_33,
            weight_37,
            l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_33 = (
            weight_37
        ) = (
            l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_bias_
        ) = None
        silu_34 = torch.nn.functional.silu(out_52, inplace=False)
        out_52 = None
        reshape_38 = l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_67 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_gain_ = None
        view_52 = mul_67.view(-1)
        mul_67 = None
        batch_norm_38 = torch.nn.functional.batch_norm(
            reshape_38,
            None,
            None,
            weight=view_52,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_38 = view_52 = None
        weight_38 = batch_norm_38.reshape_as(
            l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_weight_
        )
        batch_norm_38 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_weight_
        ) = None
        out_53 = torch.conv2d(
            silu_34,
            weight_38,
            l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_34 = (
            weight_38
        ) = (
            l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_bias_
        ) = None
        mean_7 = out_53.mean((2, 3))
        y_21 = mean_7.view(1, 1, -1)
        mean_7 = None
        y_22 = torch.conv1d(
            y_21,
            l_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_21 = l_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_7 = y_22.sigmoid()
        y_22 = None
        y_23 = sigmoid_7.view(1, -1, 1, 1)
        sigmoid_7 = None
        expand_as_7 = y_23.expand_as(out_53)
        y_23 = None
        mul_68 = out_53 * expand_as_7
        out_53 = expand_as_7 = None
        out_54 = 2.0 * mul_68
        mul_68 = None
        mul_70 = out_54 * 0.2
        out_54 = None
        out_55 = mul_70 + out_48
        mul_70 = out_48 = None
        silu_35 = torch.nn.functional.silu(out_55, inplace=False)
        out_56 = silu_35 * 0.9128709291752768
        silu_35 = None
        reshape_39 = l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_72 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_gain_ = None
        view_55 = mul_72.view(-1)
        mul_72 = None
        batch_norm_39 = torch.nn.functional.batch_norm(
            reshape_39,
            None,
            None,
            weight=view_55,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_39 = view_55 = None
        weight_39 = batch_norm_39.reshape_as(
            l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_weight_
        )
        batch_norm_39 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_weight_
        ) = None
        out_57 = torch.conv2d(
            out_56,
            weight_39,
            l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_56 = (
            weight_39
        ) = (
            l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_bias_
        ) = None
        silu_36 = torch.nn.functional.silu(out_57, inplace=True)
        out_57 = None
        reshape_40 = l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_73 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_gain_ = None
        view_56 = mul_73.view(-1)
        mul_73 = None
        batch_norm_40 = torch.nn.functional.batch_norm(
            reshape_40,
            None,
            None,
            weight=view_56,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_40 = view_56 = None
        weight_40 = batch_norm_40.reshape_as(
            l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_weight_
        )
        batch_norm_40 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_weight_
        ) = None
        out_58 = torch.conv2d(
            silu_36,
            weight_40,
            l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_36 = (
            weight_40
        ) = (
            l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_bias_
        ) = None
        silu_37 = torch.nn.functional.silu(out_58, inplace=True)
        out_58 = None
        reshape_41 = l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_74 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_gain_ = None
        view_57 = mul_74.view(-1)
        mul_74 = None
        batch_norm_41 = torch.nn.functional.batch_norm(
            reshape_41,
            None,
            None,
            weight=view_57,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_41 = view_57 = None
        weight_41 = batch_norm_41.reshape_as(
            l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_weight_
        )
        batch_norm_41 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_weight_
        ) = None
        out_59 = torch.conv2d(
            silu_37,
            weight_41,
            l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_37 = (
            weight_41
        ) = (
            l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_bias_
        ) = None
        silu_38 = torch.nn.functional.silu(out_59, inplace=False)
        out_59 = None
        reshape_42 = l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_75 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_gain_ = None
        view_58 = mul_75.view(-1)
        mul_75 = None
        batch_norm_42 = torch.nn.functional.batch_norm(
            reshape_42,
            None,
            None,
            weight=view_58,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_42 = view_58 = None
        weight_42 = batch_norm_42.reshape_as(
            l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_weight_
        )
        batch_norm_42 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_weight_
        ) = None
        out_60 = torch.conv2d(
            silu_38,
            weight_42,
            l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_38 = (
            weight_42
        ) = (
            l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_bias_
        ) = None
        mean_8 = out_60.mean((2, 3))
        y_24 = mean_8.view(1, 1, -1)
        mean_8 = None
        y_25 = torch.conv1d(
            y_24,
            l_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_24 = l_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_8 = y_25.sigmoid()
        y_25 = None
        y_26 = sigmoid_8.view(1, -1, 1, 1)
        sigmoid_8 = None
        expand_as_8 = y_26.expand_as(out_60)
        y_26 = None
        mul_76 = out_60 * expand_as_8
        out_60 = expand_as_8 = None
        out_61 = 2.0 * mul_76
        mul_76 = None
        mul_78 = out_61 * 0.2
        out_61 = None
        out_62 = mul_78 + out_55
        mul_78 = out_55 = None
        silu_39 = torch.nn.functional.silu(out_62, inplace=False)
        out_62 = None
        out_63 = silu_39 * 0.8980265101338745
        silu_39 = None
        avg_pool2d_2 = torch._C._nn.avg_pool2d(out_63, 2, 2, 0, True, False, None)
        reshape_43 = l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_80 = (
            l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_gain_ = (
            None
        )
        view_61 = mul_80.view(-1)
        mul_80 = None
        batch_norm_43 = torch.nn.functional.batch_norm(
            reshape_43,
            None,
            None,
            weight=view_61,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_43 = view_61 = None
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
            1, 384, -1
        )
        mul_81 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_gain_ = None
        view_62 = mul_81.view(-1)
        mul_81 = None
        batch_norm_44 = torch.nn.functional.batch_norm(
            reshape_44,
            None,
            None,
            weight=view_62,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_44 = view_62 = None
        weight_44 = batch_norm_44.reshape_as(
            l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_weight_
        )
        batch_norm_44 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_weight_
        ) = None
        out_64 = torch.conv2d(
            out_63,
            weight_44,
            l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_63 = (
            weight_44
        ) = (
            l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_bias_
        ) = None
        silu_40 = torch.nn.functional.silu(out_64, inplace=True)
        out_64 = None
        reshape_45 = l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_82 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_gain_ = None
        view_63 = mul_82.view(-1)
        mul_82 = None
        batch_norm_45 = torch.nn.functional.batch_norm(
            reshape_45,
            None,
            None,
            weight=view_63,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_45 = view_63 = None
        weight_45 = batch_norm_45.reshape_as(
            l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_weight_
        )
        batch_norm_45 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_weight_
        ) = None
        out_65 = torch.conv2d(
            silu_40,
            weight_45,
            l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            6,
        )
        silu_40 = (
            weight_45
        ) = (
            l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_bias_
        ) = None
        silu_41 = torch.nn.functional.silu(out_65, inplace=True)
        out_65 = None
        reshape_46 = l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_83 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_gain_ = None
        view_64 = mul_83.view(-1)
        mul_83 = None
        batch_norm_46 = torch.nn.functional.batch_norm(
            reshape_46,
            None,
            None,
            weight=view_64,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_46 = view_64 = None
        weight_46 = batch_norm_46.reshape_as(
            l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_weight_
        )
        batch_norm_46 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_weight_
        ) = None
        out_66 = torch.conv2d(
            silu_41,
            weight_46,
            l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_41 = (
            weight_46
        ) = (
            l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_bias_
        ) = None
        silu_42 = torch.nn.functional.silu(out_66, inplace=False)
        out_66 = None
        reshape_47 = l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_84 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_gain_ = None
        view_65 = mul_84.view(-1)
        mul_84 = None
        batch_norm_47 = torch.nn.functional.batch_norm(
            reshape_47,
            None,
            None,
            weight=view_65,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_47 = view_65 = None
        weight_47 = batch_norm_47.reshape_as(
            l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_weight_
        )
        batch_norm_47 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_weight_
        ) = None
        out_67 = torch.conv2d(
            silu_42,
            weight_47,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_42 = (
            weight_47
        ) = (
            l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_bias_
        ) = None
        mean_9 = out_67.mean((2, 3))
        y_27 = mean_9.view(1, 1, -1)
        mean_9 = None
        y_28 = torch.conv1d(
            y_27,
            l_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_27 = l_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_9 = y_28.sigmoid()
        y_28 = None
        y_29 = sigmoid_9.view(1, -1, 1, 1)
        sigmoid_9 = None
        expand_as_9 = y_29.expand_as(out_67)
        y_29 = None
        mul_85 = out_67 * expand_as_9
        out_67 = expand_as_9 = None
        out_68 = 2.0 * mul_85
        mul_85 = None
        mul_87 = out_68 * 0.2
        out_68 = None
        out_69 = mul_87 + shortcut_3
        mul_87 = shortcut_3 = None
        silu_43 = torch.nn.functional.silu(out_69, inplace=False)
        out_70 = silu_43 * 0.9805806756909201
        silu_43 = None
        reshape_48 = l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_89 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_gain_ = None
        view_68 = mul_89.view(-1)
        mul_89 = None
        batch_norm_48 = torch.nn.functional.batch_norm(
            reshape_48,
            None,
            None,
            weight=view_68,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_48 = view_68 = None
        weight_48 = batch_norm_48.reshape_as(
            l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_weight_
        )
        batch_norm_48 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_weight_
        ) = None
        out_71 = torch.conv2d(
            out_70,
            weight_48,
            l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_70 = (
            weight_48
        ) = (
            l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_bias_
        ) = None
        silu_44 = torch.nn.functional.silu(out_71, inplace=True)
        out_71 = None
        reshape_49 = l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_90 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_gain_ = None
        view_69 = mul_90.view(-1)
        mul_90 = None
        batch_norm_49 = torch.nn.functional.batch_norm(
            reshape_49,
            None,
            None,
            weight=view_69,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_49 = view_69 = None
        weight_49 = batch_norm_49.reshape_as(
            l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_weight_
        )
        batch_norm_49 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_weight_
        ) = None
        out_72 = torch.conv2d(
            silu_44,
            weight_49,
            l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_44 = (
            weight_49
        ) = (
            l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_bias_
        ) = None
        silu_45 = torch.nn.functional.silu(out_72, inplace=True)
        out_72 = None
        reshape_50 = l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_91 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_gain_ = None
        view_70 = mul_91.view(-1)
        mul_91 = None
        batch_norm_50 = torch.nn.functional.batch_norm(
            reshape_50,
            None,
            None,
            weight=view_70,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_50 = view_70 = None
        weight_50 = batch_norm_50.reshape_as(
            l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_weight_
        )
        batch_norm_50 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_weight_
        ) = None
        out_73 = torch.conv2d(
            silu_45,
            weight_50,
            l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_45 = (
            weight_50
        ) = (
            l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_bias_
        ) = None
        silu_46 = torch.nn.functional.silu(out_73, inplace=False)
        out_73 = None
        reshape_51 = l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_92 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_gain_ = None
        view_71 = mul_92.view(-1)
        mul_92 = None
        batch_norm_51 = torch.nn.functional.batch_norm(
            reshape_51,
            None,
            None,
            weight=view_71,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_51 = view_71 = None
        weight_51 = batch_norm_51.reshape_as(
            l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_weight_
        )
        batch_norm_51 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_weight_
        ) = None
        out_74 = torch.conv2d(
            silu_46,
            weight_51,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_46 = (
            weight_51
        ) = (
            l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_bias_
        ) = None
        mean_10 = out_74.mean((2, 3))
        y_30 = mean_10.view(1, 1, -1)
        mean_10 = None
        y_31 = torch.conv1d(
            y_30,
            l_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_30 = l_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_10 = y_31.sigmoid()
        y_31 = None
        y_32 = sigmoid_10.view(1, -1, 1, 1)
        sigmoid_10 = None
        expand_as_10 = y_32.expand_as(out_74)
        y_32 = None
        mul_93 = out_74 * expand_as_10
        out_74 = expand_as_10 = None
        out_75 = 2.0 * mul_93
        mul_93 = None
        mul_95 = out_75 * 0.2
        out_75 = None
        out_76 = mul_95 + out_69
        mul_95 = out_69 = None
        silu_47 = torch.nn.functional.silu(out_76, inplace=False)
        out_77 = silu_47 * 0.9622504486493761
        silu_47 = None
        reshape_52 = l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_97 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_gain_ = None
        view_74 = mul_97.view(-1)
        mul_97 = None
        batch_norm_52 = torch.nn.functional.batch_norm(
            reshape_52,
            None,
            None,
            weight=view_74,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_52 = view_74 = None
        weight_52 = batch_norm_52.reshape_as(
            l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_weight_
        )
        batch_norm_52 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_weight_
        ) = None
        out_78 = torch.conv2d(
            out_77,
            weight_52,
            l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_77 = (
            weight_52
        ) = (
            l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_bias_
        ) = None
        silu_48 = torch.nn.functional.silu(out_78, inplace=True)
        out_78 = None
        reshape_53 = l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_98 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_gain_ = None
        view_75 = mul_98.view(-1)
        mul_98 = None
        batch_norm_53 = torch.nn.functional.batch_norm(
            reshape_53,
            None,
            None,
            weight=view_75,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_53 = view_75 = None
        weight_53 = batch_norm_53.reshape_as(
            l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_weight_
        )
        batch_norm_53 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_weight_
        ) = None
        out_79 = torch.conv2d(
            silu_48,
            weight_53,
            l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_48 = (
            weight_53
        ) = (
            l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_bias_
        ) = None
        silu_49 = torch.nn.functional.silu(out_79, inplace=True)
        out_79 = None
        reshape_54 = l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_99 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_gain_ = None
        view_76 = mul_99.view(-1)
        mul_99 = None
        batch_norm_54 = torch.nn.functional.batch_norm(
            reshape_54,
            None,
            None,
            weight=view_76,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_54 = view_76 = None
        weight_54 = batch_norm_54.reshape_as(
            l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_weight_
        )
        batch_norm_54 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_weight_
        ) = None
        out_80 = torch.conv2d(
            silu_49,
            weight_54,
            l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_49 = (
            weight_54
        ) = (
            l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_bias_
        ) = None
        silu_50 = torch.nn.functional.silu(out_80, inplace=False)
        out_80 = None
        reshape_55 = l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_100 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_gain_ = None
        view_77 = mul_100.view(-1)
        mul_100 = None
        batch_norm_55 = torch.nn.functional.batch_norm(
            reshape_55,
            None,
            None,
            weight=view_77,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_55 = view_77 = None
        weight_55 = batch_norm_55.reshape_as(
            l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_weight_
        )
        batch_norm_55 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_weight_
        ) = None
        out_81 = torch.conv2d(
            silu_50,
            weight_55,
            l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_50 = (
            weight_55
        ) = (
            l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_bias_
        ) = None
        mean_11 = out_81.mean((2, 3))
        y_33 = mean_11.view(1, 1, -1)
        mean_11 = None
        y_34 = torch.conv1d(
            y_33,
            l_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_33 = l_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_11 = y_34.sigmoid()
        y_34 = None
        y_35 = sigmoid_11.view(1, -1, 1, 1)
        sigmoid_11 = None
        expand_as_11 = y_35.expand_as(out_81)
        y_35 = None
        mul_101 = out_81 * expand_as_11
        out_81 = expand_as_11 = None
        out_82 = 2.0 * mul_101
        mul_101 = None
        mul_103 = out_82 * 0.2
        out_82 = None
        out_83 = mul_103 + out_76
        mul_103 = out_76 = None
        reshape_56 = l_self_modules_final_conv_parameters_weight_.reshape(1, 2304, -1)
        mul_104 = l_self_modules_final_conv_parameters_gain_ * 0.04562504637317021
        l_self_modules_final_conv_parameters_gain_ = None
        view_80 = mul_104.view(-1)
        mul_104 = None
        batch_norm_56 = torch.nn.functional.batch_norm(
            reshape_56,
            None,
            None,
            weight=view_80,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_56 = view_80 = None
        weight_56 = batch_norm_56.reshape_as(
            l_self_modules_final_conv_parameters_weight_
        )
        batch_norm_56 = l_self_modules_final_conv_parameters_weight_ = None
        x = torch.conv2d(
            out_83,
            weight_56,
            l_self_modules_final_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_83 = weight_56 = l_self_modules_final_conv_parameters_bias_ = None
        x_1 = torch.nn.functional.silu(x, inplace=True)
        x = None
        x_2 = torch.nn.functional.adaptive_avg_pool2d(x_1, 1)
        x_1 = None
        x_3 = x_2.flatten(1, -1)
        x_2 = None
        x_4 = torch.nn.functional.dropout(x_3, 0.0, False, False)
        x_3 = None
        x_5 = torch._C._nn.linear(
            x_4,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_4 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_5,)
