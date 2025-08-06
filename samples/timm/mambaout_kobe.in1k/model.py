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
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc2_parameters_bias_
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
            (24,),
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
            (48,),
            l_self_modules_stem_modules_norm2_parameters_weight_,
            l_self_modules_stem_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_6 = (
            l_self_modules_stem_modules_norm2_parameters_weight_
        ) = l_self_modules_stem_modules_norm2_parameters_bias_ = None
        x_8 = torch.nn.functional.layer_norm(
            x_7,
            (48,),
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
        split = torch.functional.split(x_9, (128, 80, 48), dim=-1)
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
            48,
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
            (48,),
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
        split_1 = torch.functional.split(x_12, (128, 80, 48), dim=-1)
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
            48,
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
            (48,),
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
        split_2 = torch.functional.split(x_15, (128, 80, 48), dim=-1)
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
            48,
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
            (96,),
            l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_,
            1e-06,
        )
        x_19 = l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_ = (None)
        x_21 = torch.nn.functional.layer_norm(
            x_20,
            (96,),
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
        split_3 = torch.functional.split(x_22, (256, 160, 96), dim=-1)
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
            96,
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
            (96,),
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
        split_4 = torch.functional.split(x_25, (256, 160, 96), dim=-1)
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
            96,
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
            (96,),
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
        split_5 = torch.functional.split(x_28, (256, 160, 96), dim=-1)
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
            96,
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
            (192,),
            l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_,
            1e-06,
        )
        x_32 = l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_ = (None)
        x_34 = torch.nn.functional.layer_norm(
            x_33,
            (192,),
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
        split_6 = torch.functional.split(x_35, (512, 320, 192), dim=-1)
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
            192,
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
            (192,),
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
        split_7 = torch.functional.split(x_38, (512, 320, 192), dim=-1)
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
            192,
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
            (192,),
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
        split_8 = torch.functional.split(x_41, (512, 320, 192), dim=-1)
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
            192,
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
            (192,),
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
        split_9 = torch.functional.split(x_44, (512, 320, 192), dim=-1)
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
            192,
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
            (192,),
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
        split_10 = torch.functional.split(x_47, (512, 320, 192), dim=-1)
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
            192,
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
            (192,),
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
        split_11 = torch.functional.split(x_50, (512, 320, 192), dim=-1)
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
            192,
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
            (192,),
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
        split_12 = torch.functional.split(x_53, (512, 320, 192), dim=-1)
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
            192,
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
            (192,),
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
        split_13 = torch.functional.split(x_56, (512, 320, 192), dim=-1)
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
            192,
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
            (192,),
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
        split_14 = torch.functional.split(x_59, (512, 320, 192), dim=-1)
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
            192,
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
        x_61 = torch.nn.functional.layer_norm(
            input_15,
            (192,),
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_bias_ = (None)
        x_62 = torch._C._nn.linear(
            x_61,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc1_parameters_bias_,
        )
        x_61 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc1_parameters_bias_ = (None)
        split_15 = torch.functional.split(x_62, (512, 320, 192), dim=-1)
        x_62 = None
        g_15 = split_15[0]
        i_15 = split_15[1]
        c_60 = split_15[2]
        split_15 = None
        c_61 = c_60.permute(0, 3, 1, 2)
        c_60 = None
        c_62 = torch.conv2d(
            c_61,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        c_61 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_parameters_bias_ = (None)
        c_63 = c_62.permute(0, 2, 3, 1)
        c_62 = None
        gelu_16 = torch._C._nn.gelu(g_15, approximate="none")
        g_15 = None
        cat_15 = torch.cat((i_15, c_63), dim=-1)
        i_15 = c_63 = None
        mul_15 = gelu_16 * cat_15
        gelu_16 = cat_15 = None
        x_63 = torch._C._nn.linear(
            mul_15,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc2_parameters_bias_,
        )
        mul_15 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc2_parameters_bias_ = (None)
        input_16 = x_63 + input_15
        x_63 = input_15 = None
        x_64 = torch.nn.functional.layer_norm(
            input_16,
            (192,),
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_bias_ = (None)
        x_65 = torch._C._nn.linear(
            x_64,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc1_parameters_bias_,
        )
        x_64 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc1_parameters_bias_ = (None)
        split_16 = torch.functional.split(x_65, (512, 320, 192), dim=-1)
        x_65 = None
        g_16 = split_16[0]
        i_16 = split_16[1]
        c_64 = split_16[2]
        split_16 = None
        c_65 = c_64.permute(0, 3, 1, 2)
        c_64 = None
        c_66 = torch.conv2d(
            c_65,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        c_65 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_parameters_bias_ = (None)
        c_67 = c_66.permute(0, 2, 3, 1)
        c_66 = None
        gelu_17 = torch._C._nn.gelu(g_16, approximate="none")
        g_16 = None
        cat_16 = torch.cat((i_16, c_67), dim=-1)
        i_16 = c_67 = None
        mul_16 = gelu_17 * cat_16
        gelu_17 = cat_16 = None
        x_66 = torch._C._nn.linear(
            mul_16,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc2_parameters_bias_,
        )
        mul_16 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc2_parameters_bias_ = (None)
        input_17 = x_66 + input_16
        x_66 = input_16 = None
        x_67 = torch.nn.functional.layer_norm(
            input_17,
            (192,),
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_bias_ = (None)
        x_68 = torch._C._nn.linear(
            x_67,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc1_parameters_bias_,
        )
        x_67 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc1_parameters_bias_ = (None)
        split_17 = torch.functional.split(x_68, (512, 320, 192), dim=-1)
        x_68 = None
        g_17 = split_17[0]
        i_17 = split_17[1]
        c_68 = split_17[2]
        split_17 = None
        c_69 = c_68.permute(0, 3, 1, 2)
        c_68 = None
        c_70 = torch.conv2d(
            c_69,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        c_69 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_parameters_bias_ = (None)
        c_71 = c_70.permute(0, 2, 3, 1)
        c_70 = None
        gelu_18 = torch._C._nn.gelu(g_17, approximate="none")
        g_17 = None
        cat_17 = torch.cat((i_17, c_71), dim=-1)
        i_17 = c_71 = None
        mul_17 = gelu_18 * cat_17
        gelu_18 = cat_17 = None
        x_69 = torch._C._nn.linear(
            mul_17,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc2_parameters_bias_,
        )
        mul_17 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc2_parameters_bias_ = (None)
        input_18 = x_69 + input_17
        x_69 = input_17 = None
        x_70 = torch.nn.functional.layer_norm(
            input_18,
            (192,),
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_bias_ = (None)
        x_71 = torch._C._nn.linear(
            x_70,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc1_parameters_bias_,
        )
        x_70 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc1_parameters_bias_ = (None)
        split_18 = torch.functional.split(x_71, (512, 320, 192), dim=-1)
        x_71 = None
        g_18 = split_18[0]
        i_18 = split_18[1]
        c_72 = split_18[2]
        split_18 = None
        c_73 = c_72.permute(0, 3, 1, 2)
        c_72 = None
        c_74 = torch.conv2d(
            c_73,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        c_73 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_parameters_bias_ = (None)
        c_75 = c_74.permute(0, 2, 3, 1)
        c_74 = None
        gelu_19 = torch._C._nn.gelu(g_18, approximate="none")
        g_18 = None
        cat_18 = torch.cat((i_18, c_75), dim=-1)
        i_18 = c_75 = None
        mul_18 = gelu_19 * cat_18
        gelu_19 = cat_18 = None
        x_72 = torch._C._nn.linear(
            mul_18,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc2_parameters_bias_,
        )
        mul_18 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc2_parameters_bias_ = (None)
        input_19 = x_72 + input_18
        x_72 = input_18 = None
        x_73 = torch.nn.functional.layer_norm(
            input_19,
            (192,),
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_bias_ = (None)
        x_74 = torch._C._nn.linear(
            x_73,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc1_parameters_bias_,
        )
        x_73 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc1_parameters_bias_ = (None)
        split_19 = torch.functional.split(x_74, (512, 320, 192), dim=-1)
        x_74 = None
        g_19 = split_19[0]
        i_19 = split_19[1]
        c_76 = split_19[2]
        split_19 = None
        c_77 = c_76.permute(0, 3, 1, 2)
        c_76 = None
        c_78 = torch.conv2d(
            c_77,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        c_77 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_parameters_bias_ = (None)
        c_79 = c_78.permute(0, 2, 3, 1)
        c_78 = None
        gelu_20 = torch._C._nn.gelu(g_19, approximate="none")
        g_19 = None
        cat_19 = torch.cat((i_19, c_79), dim=-1)
        i_19 = c_79 = None
        mul_19 = gelu_20 * cat_19
        gelu_20 = cat_19 = None
        x_75 = torch._C._nn.linear(
            mul_19,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc2_parameters_bias_,
        )
        mul_19 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc2_parameters_bias_ = (None)
        input_20 = x_75 + input_19
        x_75 = input_19 = None
        x_76 = torch.nn.functional.layer_norm(
            input_20,
            (192,),
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_bias_ = (None)
        x_77 = torch._C._nn.linear(
            x_76,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc1_parameters_bias_,
        )
        x_76 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc1_parameters_bias_ = (None)
        split_20 = torch.functional.split(x_77, (512, 320, 192), dim=-1)
        x_77 = None
        g_20 = split_20[0]
        i_20 = split_20[1]
        c_80 = split_20[2]
        split_20 = None
        c_81 = c_80.permute(0, 3, 1, 2)
        c_80 = None
        c_82 = torch.conv2d(
            c_81,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        c_81 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_parameters_bias_ = (None)
        c_83 = c_82.permute(0, 2, 3, 1)
        c_82 = None
        gelu_21 = torch._C._nn.gelu(g_20, approximate="none")
        g_20 = None
        cat_20 = torch.cat((i_20, c_83), dim=-1)
        i_20 = c_83 = None
        mul_20 = gelu_21 * cat_20
        gelu_21 = cat_20 = None
        x_78 = torch._C._nn.linear(
            mul_20,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc2_parameters_bias_,
        )
        mul_20 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc2_parameters_bias_ = (None)
        input_21 = x_78 + input_20
        x_78 = input_20 = None
        x_79 = input_21.permute(0, 3, 1, 2)
        input_21 = None
        x_80 = torch.conv2d(
            x_79,
            l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_79 = l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_ = (None)
        x_81 = x_80.permute(0, 2, 3, 1)
        x_80 = None
        x_82 = torch.nn.functional.layer_norm(
            x_81,
            (288,),
            l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_,
            1e-06,
        )
        x_81 = l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_ = (None)
        x_83 = torch.nn.functional.layer_norm(
            x_82,
            (288,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_ = (None)
        x_84 = torch._C._nn.linear(
            x_83,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc1_parameters_bias_,
        )
        x_83 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc1_parameters_bias_ = (None)
        split_21 = torch.functional.split(x_84, (768, 480, 288), dim=-1)
        x_84 = None
        g_21 = split_21[0]
        i_21 = split_21[1]
        c_84 = split_21[2]
        split_21 = None
        c_85 = c_84.permute(0, 3, 1, 2)
        c_84 = None
        c_86 = torch.conv2d(
            c_85,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            288,
        )
        c_85 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_parameters_bias_ = (None)
        c_87 = c_86.permute(0, 2, 3, 1)
        c_86 = None
        gelu_22 = torch._C._nn.gelu(g_21, approximate="none")
        g_21 = None
        cat_21 = torch.cat((i_21, c_87), dim=-1)
        i_21 = c_87 = None
        mul_21 = gelu_22 * cat_21
        gelu_22 = cat_21 = None
        x_85 = torch._C._nn.linear(
            mul_21,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc2_parameters_bias_,
        )
        mul_21 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc2_parameters_bias_ = (None)
        input_22 = x_85 + x_82
        x_85 = x_82 = None
        x_86 = torch.nn.functional.layer_norm(
            input_22,
            (288,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_ = (None)
        x_87 = torch._C._nn.linear(
            x_86,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc1_parameters_bias_,
        )
        x_86 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc1_parameters_bias_ = (None)
        split_22 = torch.functional.split(x_87, (768, 480, 288), dim=-1)
        x_87 = None
        g_22 = split_22[0]
        i_22 = split_22[1]
        c_88 = split_22[2]
        split_22 = None
        c_89 = c_88.permute(0, 3, 1, 2)
        c_88 = None
        c_90 = torch.conv2d(
            c_89,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            288,
        )
        c_89 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_parameters_bias_ = (None)
        c_91 = c_90.permute(0, 2, 3, 1)
        c_90 = None
        gelu_23 = torch._C._nn.gelu(g_22, approximate="none")
        g_22 = None
        cat_22 = torch.cat((i_22, c_91), dim=-1)
        i_22 = c_91 = None
        mul_22 = gelu_23 * cat_22
        gelu_23 = cat_22 = None
        x_88 = torch._C._nn.linear(
            mul_22,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc2_parameters_bias_,
        )
        mul_22 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc2_parameters_bias_ = (None)
        input_23 = x_88 + input_22
        x_88 = input_22 = None
        x_89 = torch.nn.functional.layer_norm(
            input_23,
            (288,),
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_bias_ = (None)
        x_90 = torch._C._nn.linear(
            x_89,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc1_parameters_bias_,
        )
        x_89 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc1_parameters_bias_ = (None)
        split_23 = torch.functional.split(x_90, (768, 480, 288), dim=-1)
        x_90 = None
        g_23 = split_23[0]
        i_23 = split_23[1]
        c_92 = split_23[2]
        split_23 = None
        c_93 = c_92.permute(0, 3, 1, 2)
        c_92 = None
        c_94 = torch.conv2d(
            c_93,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            288,
        )
        c_93 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_parameters_bias_ = (None)
        c_95 = c_94.permute(0, 2, 3, 1)
        c_94 = None
        gelu_24 = torch._C._nn.gelu(g_23, approximate="none")
        g_23 = None
        cat_23 = torch.cat((i_23, c_95), dim=-1)
        i_23 = c_95 = None
        mul_23 = gelu_24 * cat_23
        gelu_24 = cat_23 = None
        x_91 = torch._C._nn.linear(
            mul_23,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc2_parameters_bias_,
        )
        mul_23 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc2_parameters_bias_ = (None)
        input_24 = x_91 + input_23
        x_91 = input_23 = None
        x_92 = input_24.mean((1, 2))
        input_24 = None
        x_93 = torch.nn.functional.layer_norm(
            x_92,
            (288,),
            l_self_modules_head_modules_norm_parameters_weight_,
            l_self_modules_head_modules_norm_parameters_bias_,
            1e-06,
        )
        x_92 = (
            l_self_modules_head_modules_norm_parameters_weight_
        ) = l_self_modules_head_modules_norm_parameters_bias_ = None
        input_25 = torch._C._nn.linear(
            x_93,
            l_self_modules_head_modules_pre_logits_modules_fc_parameters_weight_,
            l_self_modules_head_modules_pre_logits_modules_fc_parameters_bias_,
        )
        x_93 = (
            l_self_modules_head_modules_pre_logits_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_pre_logits_modules_fc_parameters_bias_ = None
        input_26 = torch._C._nn.gelu(input_25, approximate="none")
        input_25 = None
        x_94 = torch.nn.functional.layer_norm(
            input_26,
            (1152,),
            l_self_modules_head_modules_pre_logits_modules_norm_parameters_weight_,
            l_self_modules_head_modules_pre_logits_modules_norm_parameters_bias_,
            1e-06,
        )
        input_26 = (
            l_self_modules_head_modules_pre_logits_modules_norm_parameters_weight_
        ) = l_self_modules_head_modules_pre_logits_modules_norm_parameters_bias_ = None
        x_95 = torch.nn.functional.dropout(x_94, 0.0, False, False)
        x_94 = None
        x_96 = torch._C._nn.linear(
            x_95,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_95 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_96,)
