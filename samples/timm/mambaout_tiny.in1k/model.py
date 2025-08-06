import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_pre_logits_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_pre_logits_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_pre_logits_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_pre_logits_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_conv1_parameters_weight_ = (
            L_self_modules_stem_modules_conv1_parameters_weight_
        )
        l_self_modules_stem_modules_conv1_parameters_bias_ = (
            L_self_modules_stem_modules_conv1_parameters_bias_
        )
        l_x_ = L_x_
        l_self_modules_stem_modules_norm1_parameters_weight_ = (
            L_self_modules_stem_modules_norm1_parameters_weight_
        )
        l_self_modules_stem_modules_norm1_parameters_bias_ = (
            L_self_modules_stem_modules_norm1_parameters_bias_
        )
        l_self_modules_stem_modules_conv2_parameters_weight_ = (
            L_self_modules_stem_modules_conv2_parameters_weight_
        )
        l_self_modules_stem_modules_conv2_parameters_bias_ = (
            L_self_modules_stem_modules_conv2_parameters_bias_
        )
        l_self_modules_stem_modules_norm2_parameters_weight_ = (
            L_self_modules_stem_modules_norm2_parameters_weight_
        )
        l_self_modules_stem_modules_norm2_parameters_bias_ = (
            L_self_modules_stem_modules_norm2_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_ = L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_
        l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_ = L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_
        l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_ = L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_
        l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_ = L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc2_parameters_bias_
        l_self_modules_head_modules_norm_parameters_weight_ = (
            L_self_modules_head_modules_norm_parameters_weight_
        )
        l_self_modules_head_modules_norm_parameters_bias_ = (
            L_self_modules_head_modules_norm_parameters_bias_
        )
        l_self_modules_head_modules_pre_logits_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_pre_logits_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_pre_logits_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_pre_logits_modules_fc_parameters_bias_
        )
        l_self_modules_head_modules_pre_logits_modules_norm_parameters_weight_ = (
            L_self_modules_head_modules_pre_logits_modules_norm_parameters_weight_
        )
        l_self_modules_head_modules_pre_logits_modules_norm_parameters_bias_ = (
            L_self_modules_head_modules_pre_logits_modules_norm_parameters_bias_
        )
        l_self_modules_head_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_fc_parameters_bias_
        )
        x = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_conv1_parameters_weight_,
            l_self_modules_stem_modules_conv1_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = (
            l_self_modules_stem_modules_conv1_parameters_weight_
        ) = l_self_modules_stem_modules_conv1_parameters_bias_ = None
        x_1 = x.permute(0, 2, 3, 1)
        x = None
        x_2 = torch.nn.functional.layer_norm(
            x_1,
            (48,),
            l_self_modules_stem_modules_norm1_parameters_weight_,
            l_self_modules_stem_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_1 = (
            l_self_modules_stem_modules_norm1_parameters_weight_
        ) = l_self_modules_stem_modules_norm1_parameters_bias_ = None
        x_3 = x_2.permute(0, 3, 1, 2)
        x_2 = None
        x_4 = torch._C._nn.gelu(x_3, approximate="none")
        x_3 = None
        x_5 = torch.conv2d(
            x_4,
            l_self_modules_stem_modules_conv2_parameters_weight_,
            l_self_modules_stem_modules_conv2_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_4 = (
            l_self_modules_stem_modules_conv2_parameters_weight_
        ) = l_self_modules_stem_modules_conv2_parameters_bias_ = None
        x_6 = x_5.permute(0, 2, 3, 1)
        x_5 = None
        x_7 = torch.nn.functional.layer_norm(
            x_6,
            (96,),
            l_self_modules_stem_modules_norm2_parameters_weight_,
            l_self_modules_stem_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_6 = (
            l_self_modules_stem_modules_norm2_parameters_weight_
        ) = l_self_modules_stem_modules_norm2_parameters_bias_ = None
        x_8 = torch.nn.functional.layer_norm(
            x_7,
            (96,),
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_bias_ = (None)
        x_9 = torch._C._nn.linear(
            x_8,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc1_parameters_bias_,
        )
        x_8 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc1_parameters_bias_ = (None)
        split = torch.functional.split(x_9, (256, 160, 96), dim=-1)
        x_9 = None
        g = split[0]
        i = split[1]
        c = split[2]
        split = None
        c_1 = c.permute(0, 3, 1, 2)
        c = None
        c_2 = torch.conv2d(
            c_1,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            96,
        )
        c_1 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_parameters_bias_ = (None)
        c_3 = c_2.permute(0, 2, 3, 1)
        c_2 = None
        gelu_1 = torch._C._nn.gelu(g, approximate="none")
        g = None
        cat = torch.cat((i, c_3), dim=-1)
        i = c_3 = None
        mul = gelu_1 * cat
        gelu_1 = cat = None
        x_10 = torch._C._nn.linear(
            mul,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc2_parameters_bias_,
        )
        mul = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc2_parameters_bias_ = (None)
        input_1 = x_10 + x_7
        x_10 = x_7 = None
        x_11 = torch.nn.functional.layer_norm(
            input_1,
            (96,),
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_ = (None)
        x_12 = torch._C._nn.linear(
            x_11,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc1_parameters_bias_,
        )
        x_11 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc1_parameters_bias_ = (None)
        split_1 = torch.functional.split(x_12, (256, 160, 96), dim=-1)
        x_12 = None
        g_1 = split_1[0]
        i_1 = split_1[1]
        c_4 = split_1[2]
        split_1 = None
        c_5 = c_4.permute(0, 3, 1, 2)
        c_4 = None
        c_6 = torch.conv2d(
            c_5,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            96,
        )
        c_5 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_parameters_bias_ = (None)
        c_7 = c_6.permute(0, 2, 3, 1)
        c_6 = None
        gelu_2 = torch._C._nn.gelu(g_1, approximate="none")
        g_1 = None
        cat_1 = torch.cat((i_1, c_7), dim=-1)
        i_1 = c_7 = None
        mul_1 = gelu_2 * cat_1
        gelu_2 = cat_1 = None
        x_13 = torch._C._nn.linear(
            mul_1,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc2_parameters_bias_,
        )
        mul_1 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc2_parameters_bias_ = (None)
        input_2 = x_13 + input_1
        x_13 = input_1 = None
        x_14 = torch.nn.functional.layer_norm(
            input_2,
            (96,),
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_bias_ = (None)
        x_15 = torch._C._nn.linear(
            x_14,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc1_parameters_bias_,
        )
        x_14 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc1_parameters_bias_ = (None)
        split_2 = torch.functional.split(x_15, (256, 160, 96), dim=-1)
        x_15 = None
        g_2 = split_2[0]
        i_2 = split_2[1]
        c_8 = split_2[2]
        split_2 = None
        c_9 = c_8.permute(0, 3, 1, 2)
        c_8 = None
        c_10 = torch.conv2d(
            c_9,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            96,
        )
        c_9 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_parameters_bias_ = (None)
        c_11 = c_10.permute(0, 2, 3, 1)
        c_10 = None
        gelu_3 = torch._C._nn.gelu(g_2, approximate="none")
        g_2 = None
        cat_2 = torch.cat((i_2, c_11), dim=-1)
        i_2 = c_11 = None
        mul_2 = gelu_3 * cat_2
        gelu_3 = cat_2 = None
        x_16 = torch._C._nn.linear(
            mul_2,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc2_parameters_bias_,
        )
        mul_2 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc2_parameters_bias_ = (None)
        input_3 = x_16 + input_2
        x_16 = input_2 = None
        x_17 = input_3.permute(0, 3, 1, 2)
        input_3 = None
        x_18 = torch.conv2d(
            x_17,
            l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_17 = l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_ = l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_ = (None)
        x_19 = x_18.permute(0, 2, 3, 1)
        x_18 = None
        x_20 = torch.nn.functional.layer_norm(
            x_19,
            (192,),
            l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_,
            1e-06,
        )
        x_19 = l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_ = (None)
        x_21 = torch.nn.functional.layer_norm(
            x_20,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_ = (None)
        x_22 = torch._C._nn.linear(
            x_21,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc1_parameters_bias_,
        )
        x_21 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc1_parameters_bias_ = (None)
        split_3 = torch.functional.split(x_22, (512, 320, 192), dim=-1)
        x_22 = None
        g_3 = split_3[0]
        i_3 = split_3[1]
        c_12 = split_3[2]
        split_3 = None
        c_13 = c_12.permute(0, 3, 1, 2)
        c_12 = None
        c_14 = torch.conv2d(
            c_13,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        c_13 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_parameters_bias_ = (None)
        c_15 = c_14.permute(0, 2, 3, 1)
        c_14 = None
        gelu_4 = torch._C._nn.gelu(g_3, approximate="none")
        g_3 = None
        cat_3 = torch.cat((i_3, c_15), dim=-1)
        i_3 = c_15 = None
        mul_3 = gelu_4 * cat_3
        gelu_4 = cat_3 = None
        x_23 = torch._C._nn.linear(
            mul_3,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc2_parameters_bias_,
        )
        mul_3 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc2_parameters_bias_ = (None)
        input_4 = x_23 + x_20
        x_23 = x_20 = None
        x_24 = torch.nn.functional.layer_norm(
            input_4,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_ = (None)
        x_25 = torch._C._nn.linear(
            x_24,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc1_parameters_bias_,
        )
        x_24 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc1_parameters_bias_ = (None)
        split_4 = torch.functional.split(x_25, (512, 320, 192), dim=-1)
        x_25 = None
        g_4 = split_4[0]
        i_4 = split_4[1]
        c_16 = split_4[2]
        split_4 = None
        c_17 = c_16.permute(0, 3, 1, 2)
        c_16 = None
        c_18 = torch.conv2d(
            c_17,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        c_17 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_parameters_bias_ = (None)
        c_19 = c_18.permute(0, 2, 3, 1)
        c_18 = None
        gelu_5 = torch._C._nn.gelu(g_4, approximate="none")
        g_4 = None
        cat_4 = torch.cat((i_4, c_19), dim=-1)
        i_4 = c_19 = None
        mul_4 = gelu_5 * cat_4
        gelu_5 = cat_4 = None
        x_26 = torch._C._nn.linear(
            mul_4,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc2_parameters_bias_,
        )
        mul_4 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc2_parameters_bias_ = (None)
        input_5 = x_26 + input_4
        x_26 = input_4 = None
        x_27 = torch.nn.functional.layer_norm(
            input_5,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_bias_ = (None)
        x_28 = torch._C._nn.linear(
            x_27,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc1_parameters_bias_,
        )
        x_27 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc1_parameters_bias_ = (None)
        split_5 = torch.functional.split(x_28, (512, 320, 192), dim=-1)
        x_28 = None
        g_5 = split_5[0]
        i_5 = split_5[1]
        c_20 = split_5[2]
        split_5 = None
        c_21 = c_20.permute(0, 3, 1, 2)
        c_20 = None
        c_22 = torch.conv2d(
            c_21,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        c_21 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_parameters_bias_ = (None)
        c_23 = c_22.permute(0, 2, 3, 1)
        c_22 = None
        gelu_6 = torch._C._nn.gelu(g_5, approximate="none")
        g_5 = None
        cat_5 = torch.cat((i_5, c_23), dim=-1)
        i_5 = c_23 = None
        mul_5 = gelu_6 * cat_5
        gelu_6 = cat_5 = None
        x_29 = torch._C._nn.linear(
            mul_5,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc2_parameters_bias_,
        )
        mul_5 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc2_parameters_bias_ = (None)
        input_6 = x_29 + input_5
        x_29 = input_5 = None
        x_30 = input_6.permute(0, 3, 1, 2)
        input_6 = None
        x_31 = torch.conv2d(
            x_30,
            l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_30 = l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_ = (None)
        x_32 = x_31.permute(0, 2, 3, 1)
        x_31 = None
        x_33 = torch.nn.functional.layer_norm(
            x_32,
            (384,),
            l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_,
            1e-06,
        )
        x_32 = l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_ = (None)
        x_34 = torch.nn.functional.layer_norm(
            x_33,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_ = (None)
        x_35 = torch._C._nn.linear(
            x_34,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc1_parameters_bias_,
        )
        x_34 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc1_parameters_bias_ = (None)
        split_6 = torch.functional.split(x_35, (1024, 640, 384), dim=-1)
        x_35 = None
        g_6 = split_6[0]
        i_6 = split_6[1]
        c_24 = split_6[2]
        split_6 = None
        c_25 = c_24.permute(0, 3, 1, 2)
        c_24 = None
        c_26 = torch.conv2d(
            c_25,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_25 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_parameters_bias_ = (None)
        c_27 = c_26.permute(0, 2, 3, 1)
        c_26 = None
        gelu_7 = torch._C._nn.gelu(g_6, approximate="none")
        g_6 = None
        cat_6 = torch.cat((i_6, c_27), dim=-1)
        i_6 = c_27 = None
        mul_6 = gelu_7 * cat_6
        gelu_7 = cat_6 = None
        x_36 = torch._C._nn.linear(
            mul_6,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc2_parameters_bias_,
        )
        mul_6 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc2_parameters_bias_ = (None)
        input_7 = x_36 + x_33
        x_36 = x_33 = None
        x_37 = torch.nn.functional.layer_norm(
            input_7,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_ = (None)
        x_38 = torch._C._nn.linear(
            x_37,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc1_parameters_bias_,
        )
        x_37 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc1_parameters_bias_ = (None)
        split_7 = torch.functional.split(x_38, (1024, 640, 384), dim=-1)
        x_38 = None
        g_7 = split_7[0]
        i_7 = split_7[1]
        c_28 = split_7[2]
        split_7 = None
        c_29 = c_28.permute(0, 3, 1, 2)
        c_28 = None
        c_30 = torch.conv2d(
            c_29,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_29 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_parameters_bias_ = (None)
        c_31 = c_30.permute(0, 2, 3, 1)
        c_30 = None
        gelu_8 = torch._C._nn.gelu(g_7, approximate="none")
        g_7 = None
        cat_7 = torch.cat((i_7, c_31), dim=-1)
        i_7 = c_31 = None
        mul_7 = gelu_8 * cat_7
        gelu_8 = cat_7 = None
        x_39 = torch._C._nn.linear(
            mul_7,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc2_parameters_bias_,
        )
        mul_7 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc2_parameters_bias_ = (None)
        input_8 = x_39 + input_7
        x_39 = input_7 = None
        x_40 = torch.nn.functional.layer_norm(
            input_8,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_ = (None)
        x_41 = torch._C._nn.linear(
            x_40,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc1_parameters_bias_,
        )
        x_40 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc1_parameters_bias_ = (None)
        split_8 = torch.functional.split(x_41, (1024, 640, 384), dim=-1)
        x_41 = None
        g_8 = split_8[0]
        i_8 = split_8[1]
        c_32 = split_8[2]
        split_8 = None
        c_33 = c_32.permute(0, 3, 1, 2)
        c_32 = None
        c_34 = torch.conv2d(
            c_33,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_33 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_parameters_bias_ = (None)
        c_35 = c_34.permute(0, 2, 3, 1)
        c_34 = None
        gelu_9 = torch._C._nn.gelu(g_8, approximate="none")
        g_8 = None
        cat_8 = torch.cat((i_8, c_35), dim=-1)
        i_8 = c_35 = None
        mul_8 = gelu_9 * cat_8
        gelu_9 = cat_8 = None
        x_42 = torch._C._nn.linear(
            mul_8,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc2_parameters_bias_,
        )
        mul_8 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc2_parameters_bias_ = (None)
        input_9 = x_42 + input_8
        x_42 = input_8 = None
        x_43 = torch.nn.functional.layer_norm(
            input_9,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_ = (None)
        x_44 = torch._C._nn.linear(
            x_43,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc1_parameters_bias_,
        )
        x_43 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc1_parameters_bias_ = (None)
        split_9 = torch.functional.split(x_44, (1024, 640, 384), dim=-1)
        x_44 = None
        g_9 = split_9[0]
        i_9 = split_9[1]
        c_36 = split_9[2]
        split_9 = None
        c_37 = c_36.permute(0, 3, 1, 2)
        c_36 = None
        c_38 = torch.conv2d(
            c_37,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_37 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_parameters_bias_ = (None)
        c_39 = c_38.permute(0, 2, 3, 1)
        c_38 = None
        gelu_10 = torch._C._nn.gelu(g_9, approximate="none")
        g_9 = None
        cat_9 = torch.cat((i_9, c_39), dim=-1)
        i_9 = c_39 = None
        mul_9 = gelu_10 * cat_9
        gelu_10 = cat_9 = None
        x_45 = torch._C._nn.linear(
            mul_9,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc2_parameters_bias_,
        )
        mul_9 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc2_parameters_bias_ = (None)
        input_10 = x_45 + input_9
        x_45 = input_9 = None
        x_46 = torch.nn.functional.layer_norm(
            input_10,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_ = (None)
        x_47 = torch._C._nn.linear(
            x_46,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc1_parameters_bias_,
        )
        x_46 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc1_parameters_bias_ = (None)
        split_10 = torch.functional.split(x_47, (1024, 640, 384), dim=-1)
        x_47 = None
        g_10 = split_10[0]
        i_10 = split_10[1]
        c_40 = split_10[2]
        split_10 = None
        c_41 = c_40.permute(0, 3, 1, 2)
        c_40 = None
        c_42 = torch.conv2d(
            c_41,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_41 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_parameters_bias_ = (None)
        c_43 = c_42.permute(0, 2, 3, 1)
        c_42 = None
        gelu_11 = torch._C._nn.gelu(g_10, approximate="none")
        g_10 = None
        cat_10 = torch.cat((i_10, c_43), dim=-1)
        i_10 = c_43 = None
        mul_10 = gelu_11 * cat_10
        gelu_11 = cat_10 = None
        x_48 = torch._C._nn.linear(
            mul_10,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc2_parameters_bias_,
        )
        mul_10 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc2_parameters_bias_ = (None)
        input_11 = x_48 + input_10
        x_48 = input_10 = None
        x_49 = torch.nn.functional.layer_norm(
            input_11,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_ = (None)
        x_50 = torch._C._nn.linear(
            x_49,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc1_parameters_bias_,
        )
        x_49 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc1_parameters_bias_ = (None)
        split_11 = torch.functional.split(x_50, (1024, 640, 384), dim=-1)
        x_50 = None
        g_11 = split_11[0]
        i_11 = split_11[1]
        c_44 = split_11[2]
        split_11 = None
        c_45 = c_44.permute(0, 3, 1, 2)
        c_44 = None
        c_46 = torch.conv2d(
            c_45,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_45 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_parameters_bias_ = (None)
        c_47 = c_46.permute(0, 2, 3, 1)
        c_46 = None
        gelu_12 = torch._C._nn.gelu(g_11, approximate="none")
        g_11 = None
        cat_11 = torch.cat((i_11, c_47), dim=-1)
        i_11 = c_47 = None
        mul_11 = gelu_12 * cat_11
        gelu_12 = cat_11 = None
        x_51 = torch._C._nn.linear(
            mul_11,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc2_parameters_bias_,
        )
        mul_11 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc2_parameters_bias_ = (None)
        input_12 = x_51 + input_11
        x_51 = input_11 = None
        x_52 = torch.nn.functional.layer_norm(
            input_12,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_bias_ = (None)
        x_53 = torch._C._nn.linear(
            x_52,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc1_parameters_bias_,
        )
        x_52 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc1_parameters_bias_ = (None)
        split_12 = torch.functional.split(x_53, (1024, 640, 384), dim=-1)
        x_53 = None
        g_12 = split_12[0]
        i_12 = split_12[1]
        c_48 = split_12[2]
        split_12 = None
        c_49 = c_48.permute(0, 3, 1, 2)
        c_48 = None
        c_50 = torch.conv2d(
            c_49,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_49 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_parameters_bias_ = (None)
        c_51 = c_50.permute(0, 2, 3, 1)
        c_50 = None
        gelu_13 = torch._C._nn.gelu(g_12, approximate="none")
        g_12 = None
        cat_12 = torch.cat((i_12, c_51), dim=-1)
        i_12 = c_51 = None
        mul_12 = gelu_13 * cat_12
        gelu_13 = cat_12 = None
        x_54 = torch._C._nn.linear(
            mul_12,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc2_parameters_bias_,
        )
        mul_12 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc2_parameters_bias_ = (None)
        input_13 = x_54 + input_12
        x_54 = input_12 = None
        x_55 = torch.nn.functional.layer_norm(
            input_13,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_bias_ = (None)
        x_56 = torch._C._nn.linear(
            x_55,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc1_parameters_bias_,
        )
        x_55 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc1_parameters_bias_ = (None)
        split_13 = torch.functional.split(x_56, (1024, 640, 384), dim=-1)
        x_56 = None
        g_13 = split_13[0]
        i_13 = split_13[1]
        c_52 = split_13[2]
        split_13 = None
        c_53 = c_52.permute(0, 3, 1, 2)
        c_52 = None
        c_54 = torch.conv2d(
            c_53,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_53 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_parameters_bias_ = (None)
        c_55 = c_54.permute(0, 2, 3, 1)
        c_54 = None
        gelu_14 = torch._C._nn.gelu(g_13, approximate="none")
        g_13 = None
        cat_13 = torch.cat((i_13, c_55), dim=-1)
        i_13 = c_55 = None
        mul_13 = gelu_14 * cat_13
        gelu_14 = cat_13 = None
        x_57 = torch._C._nn.linear(
            mul_13,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc2_parameters_bias_,
        )
        mul_13 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc2_parameters_bias_ = (None)
        input_14 = x_57 + input_13
        x_57 = input_13 = None
        x_58 = torch.nn.functional.layer_norm(
            input_14,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_bias_ = (None)
        x_59 = torch._C._nn.linear(
            x_58,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc1_parameters_bias_,
        )
        x_58 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc1_parameters_bias_ = (None)
        split_14 = torch.functional.split(x_59, (1024, 640, 384), dim=-1)
        x_59 = None
        g_14 = split_14[0]
        i_14 = split_14[1]
        c_56 = split_14[2]
        split_14 = None
        c_57 = c_56.permute(0, 3, 1, 2)
        c_56 = None
        c_58 = torch.conv2d(
            c_57,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_57 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_parameters_bias_ = (None)
        c_59 = c_58.permute(0, 2, 3, 1)
        c_58 = None
        gelu_15 = torch._C._nn.gelu(g_14, approximate="none")
        g_14 = None
        cat_14 = torch.cat((i_14, c_59), dim=-1)
        i_14 = c_59 = None
        mul_14 = gelu_15 * cat_14
        gelu_15 = cat_14 = None
        x_60 = torch._C._nn.linear(
            mul_14,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc2_parameters_bias_,
        )
        mul_14 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc2_parameters_bias_ = (None)
        input_15 = x_60 + input_14
        x_60 = input_14 = None
        x_61 = input_15.permute(0, 3, 1, 2)
        input_15 = None
        x_62 = torch.conv2d(
            x_61,
            l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_61 = l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_ = (None)
        x_63 = x_62.permute(0, 2, 3, 1)
        x_62 = None
        x_64 = torch.nn.functional.layer_norm(
            x_63,
            (576,),
            l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_,
            1e-06,
        )
        x_63 = l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_ = (None)
        x_65 = torch.nn.functional.layer_norm(
            x_64,
            (576,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_ = (None)
        x_66 = torch._C._nn.linear(
            x_65,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc1_parameters_bias_,
        )
        x_65 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc1_parameters_bias_ = (None)
        split_15 = torch.functional.split(x_66, (1536, 960, 576), dim=-1)
        x_66 = None
        g_15 = split_15[0]
        i_15 = split_15[1]
        c_60 = split_15[2]
        split_15 = None
        c_61 = c_60.permute(0, 3, 1, 2)
        c_60 = None
        c_62 = torch.conv2d(
            c_61,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            576,
        )
        c_61 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_parameters_bias_ = (None)
        c_63 = c_62.permute(0, 2, 3, 1)
        c_62 = None
        gelu_16 = torch._C._nn.gelu(g_15, approximate="none")
        g_15 = None
        cat_15 = torch.cat((i_15, c_63), dim=-1)
        i_15 = c_63 = None
        mul_15 = gelu_16 * cat_15
        gelu_16 = cat_15 = None
        x_67 = torch._C._nn.linear(
            mul_15,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc2_parameters_bias_,
        )
        mul_15 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc2_parameters_bias_ = (None)
        input_16 = x_67 + x_64
        x_67 = x_64 = None
        x_68 = torch.nn.functional.layer_norm(
            input_16,
            (576,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_ = (None)
        x_69 = torch._C._nn.linear(
            x_68,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc1_parameters_bias_,
        )
        x_68 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc1_parameters_bias_ = (None)
        split_16 = torch.functional.split(x_69, (1536, 960, 576), dim=-1)
        x_69 = None
        g_16 = split_16[0]
        i_16 = split_16[1]
        c_64 = split_16[2]
        split_16 = None
        c_65 = c_64.permute(0, 3, 1, 2)
        c_64 = None
        c_66 = torch.conv2d(
            c_65,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            576,
        )
        c_65 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_parameters_bias_ = (None)
        c_67 = c_66.permute(0, 2, 3, 1)
        c_66 = None
        gelu_17 = torch._C._nn.gelu(g_16, approximate="none")
        g_16 = None
        cat_16 = torch.cat((i_16, c_67), dim=-1)
        i_16 = c_67 = None
        mul_16 = gelu_17 * cat_16
        gelu_17 = cat_16 = None
        x_70 = torch._C._nn.linear(
            mul_16,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc2_parameters_bias_,
        )
        mul_16 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc2_parameters_bias_ = (None)
        input_17 = x_70 + input_16
        x_70 = input_16 = None
        x_71 = torch.nn.functional.layer_norm(
            input_17,
            (576,),
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_bias_ = (None)
        x_72 = torch._C._nn.linear(
            x_71,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc1_parameters_bias_,
        )
        x_71 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc1_parameters_bias_ = (None)
        split_17 = torch.functional.split(x_72, (1536, 960, 576), dim=-1)
        x_72 = None
        g_17 = split_17[0]
        i_17 = split_17[1]
        c_68 = split_17[2]
        split_17 = None
        c_69 = c_68.permute(0, 3, 1, 2)
        c_68 = None
        c_70 = torch.conv2d(
            c_69,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            576,
        )
        c_69 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_parameters_bias_ = (None)
        c_71 = c_70.permute(0, 2, 3, 1)
        c_70 = None
        gelu_18 = torch._C._nn.gelu(g_17, approximate="none")
        g_17 = None
        cat_17 = torch.cat((i_17, c_71), dim=-1)
        i_17 = c_71 = None
        mul_17 = gelu_18 * cat_17
        gelu_18 = cat_17 = None
        x_73 = torch._C._nn.linear(
            mul_17,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc2_parameters_bias_,
        )
        mul_17 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc2_parameters_bias_ = (None)
        input_18 = x_73 + input_17
        x_73 = input_17 = None
        x_74 = input_18.mean((1, 2))
        input_18 = None
        x_75 = torch.nn.functional.layer_norm(
            x_74,
            (576,),
            l_self_modules_head_modules_norm_parameters_weight_,
            l_self_modules_head_modules_norm_parameters_bias_,
            1e-06,
        )
        x_74 = (
            l_self_modules_head_modules_norm_parameters_weight_
        ) = l_self_modules_head_modules_norm_parameters_bias_ = None
        input_19 = torch._C._nn.linear(
            x_75,
            l_self_modules_head_modules_pre_logits_modules_fc_parameters_weight_,
            l_self_modules_head_modules_pre_logits_modules_fc_parameters_bias_,
        )
        x_75 = (
            l_self_modules_head_modules_pre_logits_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_pre_logits_modules_fc_parameters_bias_ = None
        input_20 = torch._C._nn.gelu(input_19, approximate="none")
        input_19 = None
        x_76 = torch.nn.functional.layer_norm(
            input_20,
            (2304,),
            l_self_modules_head_modules_pre_logits_modules_norm_parameters_weight_,
            l_self_modules_head_modules_pre_logits_modules_norm_parameters_bias_,
            1e-06,
        )
        input_20 = (
            l_self_modules_head_modules_pre_logits_modules_norm_parameters_weight_
        ) = l_self_modules_head_modules_pre_logits_modules_norm_parameters_bias_ = None
        x_77 = torch.nn.functional.dropout(x_76, 0.0, False, False)
        x_76 = None
        x_78 = torch._C._nn.linear(
            x_77,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_77 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_78,)
