import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
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
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_ls_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_pre_logits_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_pre_logits_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_ = L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_ = L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_
        l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_ = L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_
        l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_ = L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_ls_parameters_gamma_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_ls_parameters_gamma_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_ls_parameters_gamma_
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
        x_1 = torch._C._nn.gelu(x, approximate="none")
        x = None
        x_2 = torch.conv2d(
            x_1,
            l_self_modules_stem_modules_conv2_parameters_weight_,
            l_self_modules_stem_modules_conv2_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_1 = (
            l_self_modules_stem_modules_conv2_parameters_weight_
        ) = l_self_modules_stem_modules_conv2_parameters_bias_ = None
        x_3 = x_2.permute(0, 2, 3, 1)
        x_2 = None
        x_4 = torch.nn.functional.layer_norm(
            x_3,
            (96,),
            l_self_modules_stem_modules_norm2_parameters_weight_,
            l_self_modules_stem_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_3 = (
            l_self_modules_stem_modules_norm2_parameters_weight_
        ) = l_self_modules_stem_modules_norm2_parameters_bias_ = None
        x_5 = torch.nn.functional.layer_norm(
            x_4,
            (96,),
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_bias_ = (None)
        x_6 = torch._C._nn.linear(
            x_5,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc1_parameters_bias_,
        )
        x_5 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc1_parameters_bias_ = (None)
        split = torch.functional.split(x_6, (256, 160, 96), dim=-1)
        x_6 = None
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
        x_7 = torch._C._nn.linear(
            mul,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc2_parameters_bias_,
        )
        mul = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc2_parameters_bias_ = (None)
        x_8 = (
            x_7
            * l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_ls_parameters_gamma_
        )
        x_7 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_ls_parameters_gamma_ = (None)
        input_1 = x_8 + x_4
        x_8 = x_4 = None
        x_9 = torch.nn.functional.layer_norm(
            input_1,
            (96,),
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_ = (None)
        x_10 = torch._C._nn.linear(
            x_9,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc1_parameters_bias_,
        )
        x_9 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc1_parameters_bias_ = (None)
        split_1 = torch.functional.split(x_10, (256, 160, 96), dim=-1)
        x_10 = None
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
        mul_2 = gelu_2 * cat_1
        gelu_2 = cat_1 = None
        x_11 = torch._C._nn.linear(
            mul_2,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc2_parameters_bias_,
        )
        mul_2 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc2_parameters_bias_ = (None)
        x_12 = (
            x_11
            * l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_ls_parameters_gamma_
        )
        x_11 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_ls_parameters_gamma_ = (None)
        input_2 = x_12 + input_1
        x_12 = input_1 = None
        x_13 = torch.nn.functional.layer_norm(
            input_2,
            (96,),
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_bias_ = (None)
        x_14 = torch._C._nn.linear(
            x_13,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc1_parameters_bias_,
        )
        x_13 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc1_parameters_bias_ = (None)
        split_2 = torch.functional.split(x_14, (256, 160, 96), dim=-1)
        x_14 = None
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
        mul_4 = gelu_3 * cat_2
        gelu_3 = cat_2 = None
        x_15 = torch._C._nn.linear(
            mul_4,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc2_parameters_bias_,
        )
        mul_4 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc2_parameters_bias_ = (None)
        x_16 = (
            x_15
            * l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_ls_parameters_gamma_
        )
        x_15 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_ls_parameters_gamma_ = (None)
        input_3 = x_16 + input_2
        x_16 = input_2 = None
        x_17 = torch.nn.functional.layer_norm(
            input_3,
            (96,),
            l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_,
            1e-06,
        )
        input_3 = l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_ = (None)
        x_18 = x_17.permute(0, 3, 1, 2)
        x_17 = None
        x_19 = torch.conv2d(
            x_18,
            l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_18 = l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_ = l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_ = (None)
        x_20 = x_19.permute(0, 2, 3, 1)
        x_19 = None
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
        mul_6 = gelu_4 * cat_3
        gelu_4 = cat_3 = None
        x_23 = torch._C._nn.linear(
            mul_6,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc2_parameters_bias_,
        )
        mul_6 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc2_parameters_bias_ = (None)
        x_24 = (
            x_23
            * l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_ls_parameters_gamma_
        )
        x_23 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_ls_parameters_gamma_ = (None)
        input_4 = x_24 + x_20
        x_24 = x_20 = None
        x_25 = torch.nn.functional.layer_norm(
            input_4,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_ = (None)
        x_26 = torch._C._nn.linear(
            x_25,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc1_parameters_bias_,
        )
        x_25 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc1_parameters_bias_ = (None)
        split_4 = torch.functional.split(x_26, (512, 320, 192), dim=-1)
        x_26 = None
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
        mul_8 = gelu_5 * cat_4
        gelu_5 = cat_4 = None
        x_27 = torch._C._nn.linear(
            mul_8,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc2_parameters_bias_,
        )
        mul_8 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc2_parameters_bias_ = (None)
        x_28 = (
            x_27
            * l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_ls_parameters_gamma_
        )
        x_27 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_ls_parameters_gamma_ = (None)
        input_5 = x_28 + input_4
        x_28 = input_4 = None
        x_29 = torch.nn.functional.layer_norm(
            input_5,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_bias_ = (None)
        x_30 = torch._C._nn.linear(
            x_29,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc1_parameters_bias_,
        )
        x_29 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc1_parameters_bias_ = (None)
        split_5 = torch.functional.split(x_30, (512, 320, 192), dim=-1)
        x_30 = None
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
        mul_10 = gelu_6 * cat_5
        gelu_6 = cat_5 = None
        x_31 = torch._C._nn.linear(
            mul_10,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc2_parameters_bias_,
        )
        mul_10 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc2_parameters_bias_ = (None)
        x_32 = (
            x_31
            * l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_ls_parameters_gamma_
        )
        x_31 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_ls_parameters_gamma_ = (None)
        input_6 = x_32 + input_5
        x_32 = input_5 = None
        x_33 = torch.nn.functional.layer_norm(
            input_6,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm_parameters_bias_ = (None)
        x_34 = torch._C._nn.linear(
            x_33,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_fc1_parameters_bias_,
        )
        x_33 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_fc1_parameters_bias_ = (None)
        split_6 = torch.functional.split(x_34, (512, 320, 192), dim=-1)
        x_34 = None
        g_6 = split_6[0]
        i_6 = split_6[1]
        c_24 = split_6[2]
        split_6 = None
        c_25 = c_24.permute(0, 3, 1, 2)
        c_24 = None
        c_26 = torch.conv2d(
            c_25,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        c_25 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv_parameters_bias_ = (None)
        c_27 = c_26.permute(0, 2, 3, 1)
        c_26 = None
        gelu_7 = torch._C._nn.gelu(g_6, approximate="none")
        g_6 = None
        cat_6 = torch.cat((i_6, c_27), dim=-1)
        i_6 = c_27 = None
        mul_12 = gelu_7 * cat_6
        gelu_7 = cat_6 = None
        x_35 = torch._C._nn.linear(
            mul_12,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_fc2_parameters_bias_,
        )
        mul_12 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_fc2_parameters_bias_ = (None)
        x_36 = (
            x_35
            * l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_ls_parameters_gamma_
        )
        x_35 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_ls_parameters_gamma_ = (None)
        input_7 = x_36 + input_6
        x_36 = input_6 = None
        x_37 = torch.nn.functional.layer_norm(
            input_7,
            (192,),
            l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_,
            1e-06,
        )
        input_7 = l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_ = (None)
        x_38 = x_37.permute(0, 3, 1, 2)
        x_37 = None
        x_39 = torch.conv2d(
            x_38,
            l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_38 = l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_ = (None)
        x_40 = x_39.permute(0, 2, 3, 1)
        x_39 = None
        x_41 = torch.nn.functional.layer_norm(
            x_40,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_ = (None)
        x_42 = torch._C._nn.linear(
            x_41,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc1_parameters_bias_,
        )
        x_41 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc1_parameters_bias_ = (None)
        split_7 = torch.functional.split(x_42, (1024, 640, 384), dim=-1)
        x_42 = None
        g_7 = split_7[0]
        i_7 = split_7[1]
        c_28 = split_7[2]
        split_7 = None
        c_29 = c_28.permute(0, 3, 1, 2)
        c_28 = None
        c_30 = torch.conv2d(
            c_29,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_29 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_parameters_bias_ = (None)
        c_31 = c_30.permute(0, 2, 3, 1)
        c_30 = None
        gelu_8 = torch._C._nn.gelu(g_7, approximate="none")
        g_7 = None
        cat_7 = torch.cat((i_7, c_31), dim=-1)
        i_7 = c_31 = None
        mul_14 = gelu_8 * cat_7
        gelu_8 = cat_7 = None
        x_43 = torch._C._nn.linear(
            mul_14,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc2_parameters_bias_,
        )
        mul_14 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc2_parameters_bias_ = (None)
        x_44 = (
            x_43
            * l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_ls_parameters_gamma_
        )
        x_43 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_ls_parameters_gamma_ = (None)
        input_8 = x_44 + x_40
        x_44 = x_40 = None
        x_45 = torch.nn.functional.layer_norm(
            input_8,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_ = (None)
        x_46 = torch._C._nn.linear(
            x_45,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc1_parameters_bias_,
        )
        x_45 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc1_parameters_bias_ = (None)
        split_8 = torch.functional.split(x_46, (1024, 640, 384), dim=-1)
        x_46 = None
        g_8 = split_8[0]
        i_8 = split_8[1]
        c_32 = split_8[2]
        split_8 = None
        c_33 = c_32.permute(0, 3, 1, 2)
        c_32 = None
        c_34 = torch.conv2d(
            c_33,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_33 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_parameters_bias_ = (None)
        c_35 = c_34.permute(0, 2, 3, 1)
        c_34 = None
        gelu_9 = torch._C._nn.gelu(g_8, approximate="none")
        g_8 = None
        cat_8 = torch.cat((i_8, c_35), dim=-1)
        i_8 = c_35 = None
        mul_16 = gelu_9 * cat_8
        gelu_9 = cat_8 = None
        x_47 = torch._C._nn.linear(
            mul_16,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc2_parameters_bias_,
        )
        mul_16 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc2_parameters_bias_ = (None)
        x_48 = (
            x_47
            * l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_ls_parameters_gamma_
        )
        x_47 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_ls_parameters_gamma_ = (None)
        input_9 = x_48 + input_8
        x_48 = input_8 = None
        x_49 = torch.nn.functional.layer_norm(
            input_9,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_ = (None)
        x_50 = torch._C._nn.linear(
            x_49,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc1_parameters_bias_,
        )
        x_49 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc1_parameters_bias_ = (None)
        split_9 = torch.functional.split(x_50, (1024, 640, 384), dim=-1)
        x_50 = None
        g_9 = split_9[0]
        i_9 = split_9[1]
        c_36 = split_9[2]
        split_9 = None
        c_37 = c_36.permute(0, 3, 1, 2)
        c_36 = None
        c_38 = torch.conv2d(
            c_37,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_37 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_parameters_bias_ = (None)
        c_39 = c_38.permute(0, 2, 3, 1)
        c_38 = None
        gelu_10 = torch._C._nn.gelu(g_9, approximate="none")
        g_9 = None
        cat_9 = torch.cat((i_9, c_39), dim=-1)
        i_9 = c_39 = None
        mul_18 = gelu_10 * cat_9
        gelu_10 = cat_9 = None
        x_51 = torch._C._nn.linear(
            mul_18,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc2_parameters_bias_,
        )
        mul_18 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc2_parameters_bias_ = (None)
        x_52 = (
            x_51
            * l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_ls_parameters_gamma_
        )
        x_51 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_ls_parameters_gamma_ = (None)
        input_10 = x_52 + input_9
        x_52 = input_9 = None
        x_53 = torch.nn.functional.layer_norm(
            input_10,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_ = (None)
        x_54 = torch._C._nn.linear(
            x_53,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc1_parameters_bias_,
        )
        x_53 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc1_parameters_bias_ = (None)
        split_10 = torch.functional.split(x_54, (1024, 640, 384), dim=-1)
        x_54 = None
        g_10 = split_10[0]
        i_10 = split_10[1]
        c_40 = split_10[2]
        split_10 = None
        c_41 = c_40.permute(0, 3, 1, 2)
        c_40 = None
        c_42 = torch.conv2d(
            c_41,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_41 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_parameters_bias_ = (None)
        c_43 = c_42.permute(0, 2, 3, 1)
        c_42 = None
        gelu_11 = torch._C._nn.gelu(g_10, approximate="none")
        g_10 = None
        cat_10 = torch.cat((i_10, c_43), dim=-1)
        i_10 = c_43 = None
        mul_20 = gelu_11 * cat_10
        gelu_11 = cat_10 = None
        x_55 = torch._C._nn.linear(
            mul_20,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc2_parameters_bias_,
        )
        mul_20 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc2_parameters_bias_ = (None)
        x_56 = (
            x_55
            * l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_ls_parameters_gamma_
        )
        x_55 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_ls_parameters_gamma_ = (None)
        input_11 = x_56 + input_10
        x_56 = input_10 = None
        x_57 = torch.nn.functional.layer_norm(
            input_11,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_ = (None)
        x_58 = torch._C._nn.linear(
            x_57,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc1_parameters_bias_,
        )
        x_57 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc1_parameters_bias_ = (None)
        split_11 = torch.functional.split(x_58, (1024, 640, 384), dim=-1)
        x_58 = None
        g_11 = split_11[0]
        i_11 = split_11[1]
        c_44 = split_11[2]
        split_11 = None
        c_45 = c_44.permute(0, 3, 1, 2)
        c_44 = None
        c_46 = torch.conv2d(
            c_45,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_45 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_parameters_bias_ = (None)
        c_47 = c_46.permute(0, 2, 3, 1)
        c_46 = None
        gelu_12 = torch._C._nn.gelu(g_11, approximate="none")
        g_11 = None
        cat_11 = torch.cat((i_11, c_47), dim=-1)
        i_11 = c_47 = None
        mul_22 = gelu_12 * cat_11
        gelu_12 = cat_11 = None
        x_59 = torch._C._nn.linear(
            mul_22,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc2_parameters_bias_,
        )
        mul_22 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc2_parameters_bias_ = (None)
        x_60 = (
            x_59
            * l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_ls_parameters_gamma_
        )
        x_59 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_ls_parameters_gamma_ = (None)
        input_12 = x_60 + input_11
        x_60 = input_11 = None
        x_61 = torch.nn.functional.layer_norm(
            input_12,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_ = (None)
        x_62 = torch._C._nn.linear(
            x_61,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc1_parameters_bias_,
        )
        x_61 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc1_parameters_bias_ = (None)
        split_12 = torch.functional.split(x_62, (1024, 640, 384), dim=-1)
        x_62 = None
        g_12 = split_12[0]
        i_12 = split_12[1]
        c_48 = split_12[2]
        split_12 = None
        c_49 = c_48.permute(0, 3, 1, 2)
        c_48 = None
        c_50 = torch.conv2d(
            c_49,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_49 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_parameters_bias_ = (None)
        c_51 = c_50.permute(0, 2, 3, 1)
        c_50 = None
        gelu_13 = torch._C._nn.gelu(g_12, approximate="none")
        g_12 = None
        cat_12 = torch.cat((i_12, c_51), dim=-1)
        i_12 = c_51 = None
        mul_24 = gelu_13 * cat_12
        gelu_13 = cat_12 = None
        x_63 = torch._C._nn.linear(
            mul_24,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc2_parameters_bias_,
        )
        mul_24 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc2_parameters_bias_ = (None)
        x_64 = (
            x_63
            * l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_ls_parameters_gamma_
        )
        x_63 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_ls_parameters_gamma_ = (None)
        input_13 = x_64 + input_12
        x_64 = input_12 = None
        x_65 = torch.nn.functional.layer_norm(
            input_13,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_bias_ = (None)
        x_66 = torch._C._nn.linear(
            x_65,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc1_parameters_bias_,
        )
        x_65 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc1_parameters_bias_ = (None)
        split_13 = torch.functional.split(x_66, (1024, 640, 384), dim=-1)
        x_66 = None
        g_13 = split_13[0]
        i_13 = split_13[1]
        c_52 = split_13[2]
        split_13 = None
        c_53 = c_52.permute(0, 3, 1, 2)
        c_52 = None
        c_54 = torch.conv2d(
            c_53,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_53 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_parameters_bias_ = (None)
        c_55 = c_54.permute(0, 2, 3, 1)
        c_54 = None
        gelu_14 = torch._C._nn.gelu(g_13, approximate="none")
        g_13 = None
        cat_13 = torch.cat((i_13, c_55), dim=-1)
        i_13 = c_55 = None
        mul_26 = gelu_14 * cat_13
        gelu_14 = cat_13 = None
        x_67 = torch._C._nn.linear(
            mul_26,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc2_parameters_bias_,
        )
        mul_26 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc2_parameters_bias_ = (None)
        x_68 = (
            x_67
            * l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_ls_parameters_gamma_
        )
        x_67 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_ls_parameters_gamma_ = (None)
        input_14 = x_68 + input_13
        x_68 = input_13 = None
        x_69 = torch.nn.functional.layer_norm(
            input_14,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_bias_ = (None)
        x_70 = torch._C._nn.linear(
            x_69,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc1_parameters_bias_,
        )
        x_69 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc1_parameters_bias_ = (None)
        split_14 = torch.functional.split(x_70, (1024, 640, 384), dim=-1)
        x_70 = None
        g_14 = split_14[0]
        i_14 = split_14[1]
        c_56 = split_14[2]
        split_14 = None
        c_57 = c_56.permute(0, 3, 1, 2)
        c_56 = None
        c_58 = torch.conv2d(
            c_57,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_57 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_parameters_bias_ = (None)
        c_59 = c_58.permute(0, 2, 3, 1)
        c_58 = None
        gelu_15 = torch._C._nn.gelu(g_14, approximate="none")
        g_14 = None
        cat_14 = torch.cat((i_14, c_59), dim=-1)
        i_14 = c_59 = None
        mul_28 = gelu_15 * cat_14
        gelu_15 = cat_14 = None
        x_71 = torch._C._nn.linear(
            mul_28,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc2_parameters_bias_,
        )
        mul_28 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc2_parameters_bias_ = (None)
        x_72 = (
            x_71
            * l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_ls_parameters_gamma_
        )
        x_71 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_ls_parameters_gamma_ = (None)
        input_15 = x_72 + input_14
        x_72 = input_14 = None
        x_73 = torch.nn.functional.layer_norm(
            input_15,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_bias_ = (None)
        x_74 = torch._C._nn.linear(
            x_73,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc1_parameters_bias_,
        )
        x_73 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc1_parameters_bias_ = (None)
        split_15 = torch.functional.split(x_74, (1024, 640, 384), dim=-1)
        x_74 = None
        g_15 = split_15[0]
        i_15 = split_15[1]
        c_60 = split_15[2]
        split_15 = None
        c_61 = c_60.permute(0, 3, 1, 2)
        c_60 = None
        c_62 = torch.conv2d(
            c_61,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_61 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_parameters_bias_ = (None)
        c_63 = c_62.permute(0, 2, 3, 1)
        c_62 = None
        gelu_16 = torch._C._nn.gelu(g_15, approximate="none")
        g_15 = None
        cat_15 = torch.cat((i_15, c_63), dim=-1)
        i_15 = c_63 = None
        mul_30 = gelu_16 * cat_15
        gelu_16 = cat_15 = None
        x_75 = torch._C._nn.linear(
            mul_30,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc2_parameters_bias_,
        )
        mul_30 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc2_parameters_bias_ = (None)
        x_76 = (
            x_75
            * l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_ls_parameters_gamma_
        )
        x_75 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_ls_parameters_gamma_ = (None)
        input_16 = x_76 + input_15
        x_76 = input_15 = None
        x_77 = torch.nn.functional.layer_norm(
            input_16,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_bias_ = (None)
        x_78 = torch._C._nn.linear(
            x_77,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc1_parameters_bias_,
        )
        x_77 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc1_parameters_bias_ = (None)
        split_16 = torch.functional.split(x_78, (1024, 640, 384), dim=-1)
        x_78 = None
        g_16 = split_16[0]
        i_16 = split_16[1]
        c_64 = split_16[2]
        split_16 = None
        c_65 = c_64.permute(0, 3, 1, 2)
        c_64 = None
        c_66 = torch.conv2d(
            c_65,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_65 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_parameters_bias_ = (None)
        c_67 = c_66.permute(0, 2, 3, 1)
        c_66 = None
        gelu_17 = torch._C._nn.gelu(g_16, approximate="none")
        g_16 = None
        cat_16 = torch.cat((i_16, c_67), dim=-1)
        i_16 = c_67 = None
        mul_32 = gelu_17 * cat_16
        gelu_17 = cat_16 = None
        x_79 = torch._C._nn.linear(
            mul_32,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc2_parameters_bias_,
        )
        mul_32 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc2_parameters_bias_ = (None)
        x_80 = (
            x_79
            * l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_ls_parameters_gamma_
        )
        x_79 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_ls_parameters_gamma_ = (None)
        input_17 = x_80 + input_16
        x_80 = input_16 = None
        x_81 = torch.nn.functional.layer_norm(
            input_17,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_bias_ = (None)
        x_82 = torch._C._nn.linear(
            x_81,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc1_parameters_bias_,
        )
        x_81 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc1_parameters_bias_ = (None)
        split_17 = torch.functional.split(x_82, (1024, 640, 384), dim=-1)
        x_82 = None
        g_17 = split_17[0]
        i_17 = split_17[1]
        c_68 = split_17[2]
        split_17 = None
        c_69 = c_68.permute(0, 3, 1, 2)
        c_68 = None
        c_70 = torch.conv2d(
            c_69,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_69 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_parameters_bias_ = (None)
        c_71 = c_70.permute(0, 2, 3, 1)
        c_70 = None
        gelu_18 = torch._C._nn.gelu(g_17, approximate="none")
        g_17 = None
        cat_17 = torch.cat((i_17, c_71), dim=-1)
        i_17 = c_71 = None
        mul_34 = gelu_18 * cat_17
        gelu_18 = cat_17 = None
        x_83 = torch._C._nn.linear(
            mul_34,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc2_parameters_bias_,
        )
        mul_34 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc2_parameters_bias_ = (None)
        x_84 = (
            x_83
            * l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_ls_parameters_gamma_
        )
        x_83 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_ls_parameters_gamma_ = (None)
        input_18 = x_84 + input_17
        x_84 = input_17 = None
        x_85 = torch.nn.functional.layer_norm(
            input_18,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_bias_ = (None)
        x_86 = torch._C._nn.linear(
            x_85,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc1_parameters_bias_,
        )
        x_85 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc1_parameters_bias_ = (None)
        split_18 = torch.functional.split(x_86, (1024, 640, 384), dim=-1)
        x_86 = None
        g_18 = split_18[0]
        i_18 = split_18[1]
        c_72 = split_18[2]
        split_18 = None
        c_73 = c_72.permute(0, 3, 1, 2)
        c_72 = None
        c_74 = torch.conv2d(
            c_73,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_73 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_parameters_bias_ = (None)
        c_75 = c_74.permute(0, 2, 3, 1)
        c_74 = None
        gelu_19 = torch._C._nn.gelu(g_18, approximate="none")
        g_18 = None
        cat_18 = torch.cat((i_18, c_75), dim=-1)
        i_18 = c_75 = None
        mul_36 = gelu_19 * cat_18
        gelu_19 = cat_18 = None
        x_87 = torch._C._nn.linear(
            mul_36,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc2_parameters_bias_,
        )
        mul_36 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc2_parameters_bias_ = (None)
        x_88 = (
            x_87
            * l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_ls_parameters_gamma_
        )
        x_87 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_ls_parameters_gamma_ = (None)
        input_19 = x_88 + input_18
        x_88 = input_18 = None
        x_89 = torch.nn.functional.layer_norm(
            input_19,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_bias_ = (None)
        x_90 = torch._C._nn.linear(
            x_89,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc1_parameters_bias_,
        )
        x_89 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc1_parameters_bias_ = (None)
        split_19 = torch.functional.split(x_90, (1024, 640, 384), dim=-1)
        x_90 = None
        g_19 = split_19[0]
        i_19 = split_19[1]
        c_76 = split_19[2]
        split_19 = None
        c_77 = c_76.permute(0, 3, 1, 2)
        c_76 = None
        c_78 = torch.conv2d(
            c_77,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_77 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_parameters_bias_ = (None)
        c_79 = c_78.permute(0, 2, 3, 1)
        c_78 = None
        gelu_20 = torch._C._nn.gelu(g_19, approximate="none")
        g_19 = None
        cat_19 = torch.cat((i_19, c_79), dim=-1)
        i_19 = c_79 = None
        mul_38 = gelu_20 * cat_19
        gelu_20 = cat_19 = None
        x_91 = torch._C._nn.linear(
            mul_38,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc2_parameters_bias_,
        )
        mul_38 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc2_parameters_bias_ = (None)
        x_92 = (
            x_91
            * l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_ls_parameters_gamma_
        )
        x_91 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_ls_parameters_gamma_ = (None)
        input_20 = x_92 + input_19
        x_92 = input_19 = None
        x_93 = torch.nn.functional.layer_norm(
            input_20,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_bias_ = (None)
        x_94 = torch._C._nn.linear(
            x_93,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc1_parameters_bias_,
        )
        x_93 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc1_parameters_bias_ = (None)
        split_20 = torch.functional.split(x_94, (1024, 640, 384), dim=-1)
        x_94 = None
        g_20 = split_20[0]
        i_20 = split_20[1]
        c_80 = split_20[2]
        split_20 = None
        c_81 = c_80.permute(0, 3, 1, 2)
        c_80 = None
        c_82 = torch.conv2d(
            c_81,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_81 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_parameters_bias_ = (None)
        c_83 = c_82.permute(0, 2, 3, 1)
        c_82 = None
        gelu_21 = torch._C._nn.gelu(g_20, approximate="none")
        g_20 = None
        cat_20 = torch.cat((i_20, c_83), dim=-1)
        i_20 = c_83 = None
        mul_40 = gelu_21 * cat_20
        gelu_21 = cat_20 = None
        x_95 = torch._C._nn.linear(
            mul_40,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc2_parameters_bias_,
        )
        mul_40 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc2_parameters_bias_ = (None)
        x_96 = (
            x_95
            * l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_ls_parameters_gamma_
        )
        x_95 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_ls_parameters_gamma_ = (None)
        input_21 = x_96 + input_20
        x_96 = input_20 = None
        x_97 = torch.nn.functional.layer_norm(
            input_21,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_bias_ = (None)
        x_98 = torch._C._nn.linear(
            x_97,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc1_parameters_bias_,
        )
        x_97 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc1_parameters_bias_ = (None)
        split_21 = torch.functional.split(x_98, (1024, 640, 384), dim=-1)
        x_98 = None
        g_21 = split_21[0]
        i_21 = split_21[1]
        c_84 = split_21[2]
        split_21 = None
        c_85 = c_84.permute(0, 3, 1, 2)
        c_84 = None
        c_86 = torch.conv2d(
            c_85,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_85 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_parameters_bias_ = (None)
        c_87 = c_86.permute(0, 2, 3, 1)
        c_86 = None
        gelu_22 = torch._C._nn.gelu(g_21, approximate="none")
        g_21 = None
        cat_21 = torch.cat((i_21, c_87), dim=-1)
        i_21 = c_87 = None
        mul_42 = gelu_22 * cat_21
        gelu_22 = cat_21 = None
        x_99 = torch._C._nn.linear(
            mul_42,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc2_parameters_bias_,
        )
        mul_42 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc2_parameters_bias_ = (None)
        x_100 = (
            x_99
            * l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_ls_parameters_gamma_
        )
        x_99 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_ls_parameters_gamma_ = (None)
        input_22 = x_100 + input_21
        x_100 = input_21 = None
        x_101 = torch.nn.functional.layer_norm(
            input_22,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_bias_ = (None)
        x_102 = torch._C._nn.linear(
            x_101,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_fc1_parameters_bias_,
        )
        x_101 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_fc1_parameters_bias_ = (None)
        split_22 = torch.functional.split(x_102, (1024, 640, 384), dim=-1)
        x_102 = None
        g_22 = split_22[0]
        i_22 = split_22[1]
        c_88 = split_22[2]
        split_22 = None
        c_89 = c_88.permute(0, 3, 1, 2)
        c_88 = None
        c_90 = torch.conv2d(
            c_89,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_89 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_conv_parameters_bias_ = (None)
        c_91 = c_90.permute(0, 2, 3, 1)
        c_90 = None
        gelu_23 = torch._C._nn.gelu(g_22, approximate="none")
        g_22 = None
        cat_22 = torch.cat((i_22, c_91), dim=-1)
        i_22 = c_91 = None
        mul_44 = gelu_23 * cat_22
        gelu_23 = cat_22 = None
        x_103 = torch._C._nn.linear(
            mul_44,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_fc2_parameters_bias_,
        )
        mul_44 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_fc2_parameters_bias_ = (None)
        x_104 = (
            x_103
            * l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_ls_parameters_gamma_
        )
        x_103 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_ls_parameters_gamma_ = (None)
        input_23 = x_104 + input_22
        x_104 = input_22 = None
        x_105 = torch.nn.functional.layer_norm(
            input_23,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_bias_ = (None)
        x_106 = torch._C._nn.linear(
            x_105,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_fc1_parameters_bias_,
        )
        x_105 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_fc1_parameters_bias_ = (None)
        split_23 = torch.functional.split(x_106, (1024, 640, 384), dim=-1)
        x_106 = None
        g_23 = split_23[0]
        i_23 = split_23[1]
        c_92 = split_23[2]
        split_23 = None
        c_93 = c_92.permute(0, 3, 1, 2)
        c_92 = None
        c_94 = torch.conv2d(
            c_93,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_93 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_conv_parameters_bias_ = (None)
        c_95 = c_94.permute(0, 2, 3, 1)
        c_94 = None
        gelu_24 = torch._C._nn.gelu(g_23, approximate="none")
        g_23 = None
        cat_23 = torch.cat((i_23, c_95), dim=-1)
        i_23 = c_95 = None
        mul_46 = gelu_24 * cat_23
        gelu_24 = cat_23 = None
        x_107 = torch._C._nn.linear(
            mul_46,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_fc2_parameters_bias_,
        )
        mul_46 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_fc2_parameters_bias_ = (None)
        x_108 = (
            x_107
            * l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_ls_parameters_gamma_
        )
        x_107 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_ls_parameters_gamma_ = (None)
        input_24 = x_108 + input_23
        x_108 = input_23 = None
        x_109 = torch.nn.functional.layer_norm(
            input_24,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_bias_ = (None)
        x_110 = torch._C._nn.linear(
            x_109,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_fc1_parameters_bias_,
        )
        x_109 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_fc1_parameters_bias_ = (None)
        split_24 = torch.functional.split(x_110, (1024, 640, 384), dim=-1)
        x_110 = None
        g_24 = split_24[0]
        i_24 = split_24[1]
        c_96 = split_24[2]
        split_24 = None
        c_97 = c_96.permute(0, 3, 1, 2)
        c_96 = None
        c_98 = torch.conv2d(
            c_97,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_97 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_conv_parameters_bias_ = (None)
        c_99 = c_98.permute(0, 2, 3, 1)
        c_98 = None
        gelu_25 = torch._C._nn.gelu(g_24, approximate="none")
        g_24 = None
        cat_24 = torch.cat((i_24, c_99), dim=-1)
        i_24 = c_99 = None
        mul_48 = gelu_25 * cat_24
        gelu_25 = cat_24 = None
        x_111 = torch._C._nn.linear(
            mul_48,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_fc2_parameters_bias_,
        )
        mul_48 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_fc2_parameters_bias_ = (None)
        x_112 = (
            x_111
            * l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_ls_parameters_gamma_
        )
        x_111 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_ls_parameters_gamma_ = (None)
        input_25 = x_112 + input_24
        x_112 = input_24 = None
        x_113 = torch.nn.functional.layer_norm(
            input_25,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_bias_ = (None)
        x_114 = torch._C._nn.linear(
            x_113,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_fc1_parameters_bias_,
        )
        x_113 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_fc1_parameters_bias_ = (None)
        split_25 = torch.functional.split(x_114, (1024, 640, 384), dim=-1)
        x_114 = None
        g_25 = split_25[0]
        i_25 = split_25[1]
        c_100 = split_25[2]
        split_25 = None
        c_101 = c_100.permute(0, 3, 1, 2)
        c_100 = None
        c_102 = torch.conv2d(
            c_101,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_101 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_conv_parameters_bias_ = (None)
        c_103 = c_102.permute(0, 2, 3, 1)
        c_102 = None
        gelu_26 = torch._C._nn.gelu(g_25, approximate="none")
        g_25 = None
        cat_25 = torch.cat((i_25, c_103), dim=-1)
        i_25 = c_103 = None
        mul_50 = gelu_26 * cat_25
        gelu_26 = cat_25 = None
        x_115 = torch._C._nn.linear(
            mul_50,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_fc2_parameters_bias_,
        )
        mul_50 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_fc2_parameters_bias_ = (None)
        x_116 = (
            x_115
            * l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_ls_parameters_gamma_
        )
        x_115 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_ls_parameters_gamma_ = (None)
        input_26 = x_116 + input_25
        x_116 = input_25 = None
        x_117 = torch.nn.functional.layer_norm(
            input_26,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_bias_ = (None)
        x_118 = torch._C._nn.linear(
            x_117,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_fc1_parameters_bias_,
        )
        x_117 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_fc1_parameters_bias_ = (None)
        split_26 = torch.functional.split(x_118, (1024, 640, 384), dim=-1)
        x_118 = None
        g_26 = split_26[0]
        i_26 = split_26[1]
        c_104 = split_26[2]
        split_26 = None
        c_105 = c_104.permute(0, 3, 1, 2)
        c_104 = None
        c_106 = torch.conv2d(
            c_105,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_105 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_conv_parameters_bias_ = (None)
        c_107 = c_106.permute(0, 2, 3, 1)
        c_106 = None
        gelu_27 = torch._C._nn.gelu(g_26, approximate="none")
        g_26 = None
        cat_26 = torch.cat((i_26, c_107), dim=-1)
        i_26 = c_107 = None
        mul_52 = gelu_27 * cat_26
        gelu_27 = cat_26 = None
        x_119 = torch._C._nn.linear(
            mul_52,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_fc2_parameters_bias_,
        )
        mul_52 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_fc2_parameters_bias_ = (None)
        x_120 = (
            x_119
            * l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_ls_parameters_gamma_
        )
        x_119 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_ls_parameters_gamma_ = (None)
        input_27 = x_120 + input_26
        x_120 = input_26 = None
        x_121 = torch.nn.functional.layer_norm(
            input_27,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_bias_ = (None)
        x_122 = torch._C._nn.linear(
            x_121,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_fc1_parameters_bias_,
        )
        x_121 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_fc1_parameters_bias_ = (None)
        split_27 = torch.functional.split(x_122, (1024, 640, 384), dim=-1)
        x_122 = None
        g_27 = split_27[0]
        i_27 = split_27[1]
        c_108 = split_27[2]
        split_27 = None
        c_109 = c_108.permute(0, 3, 1, 2)
        c_108 = None
        c_110 = torch.conv2d(
            c_109,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_109 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_conv_parameters_bias_ = (None)
        c_111 = c_110.permute(0, 2, 3, 1)
        c_110 = None
        gelu_28 = torch._C._nn.gelu(g_27, approximate="none")
        g_27 = None
        cat_27 = torch.cat((i_27, c_111), dim=-1)
        i_27 = c_111 = None
        mul_54 = gelu_28 * cat_27
        gelu_28 = cat_27 = None
        x_123 = torch._C._nn.linear(
            mul_54,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_fc2_parameters_bias_,
        )
        mul_54 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_fc2_parameters_bias_ = (None)
        x_124 = (
            x_123
            * l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_ls_parameters_gamma_
        )
        x_123 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_ls_parameters_gamma_ = (None)
        input_28 = x_124 + input_27
        x_124 = input_27 = None
        x_125 = torch.nn.functional.layer_norm(
            input_28,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_bias_ = (None)
        x_126 = torch._C._nn.linear(
            x_125,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_fc1_parameters_bias_,
        )
        x_125 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_fc1_parameters_bias_ = (None)
        split_28 = torch.functional.split(x_126, (1024, 640, 384), dim=-1)
        x_126 = None
        g_28 = split_28[0]
        i_28 = split_28[1]
        c_112 = split_28[2]
        split_28 = None
        c_113 = c_112.permute(0, 3, 1, 2)
        c_112 = None
        c_114 = torch.conv2d(
            c_113,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_113 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_conv_parameters_bias_ = (None)
        c_115 = c_114.permute(0, 2, 3, 1)
        c_114 = None
        gelu_29 = torch._C._nn.gelu(g_28, approximate="none")
        g_28 = None
        cat_28 = torch.cat((i_28, c_115), dim=-1)
        i_28 = c_115 = None
        mul_56 = gelu_29 * cat_28
        gelu_29 = cat_28 = None
        x_127 = torch._C._nn.linear(
            mul_56,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_fc2_parameters_bias_,
        )
        mul_56 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_fc2_parameters_bias_ = (None)
        x_128 = (
            x_127
            * l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_ls_parameters_gamma_
        )
        x_127 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_ls_parameters_gamma_ = (None)
        input_29 = x_128 + input_28
        x_128 = input_28 = None
        x_129 = torch.nn.functional.layer_norm(
            input_29,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_bias_ = (None)
        x_130 = torch._C._nn.linear(
            x_129,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_fc1_parameters_bias_,
        )
        x_129 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_fc1_parameters_bias_ = (None)
        split_29 = torch.functional.split(x_130, (1024, 640, 384), dim=-1)
        x_130 = None
        g_29 = split_29[0]
        i_29 = split_29[1]
        c_116 = split_29[2]
        split_29 = None
        c_117 = c_116.permute(0, 3, 1, 2)
        c_116 = None
        c_118 = torch.conv2d(
            c_117,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_117 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_conv_parameters_bias_ = (None)
        c_119 = c_118.permute(0, 2, 3, 1)
        c_118 = None
        gelu_30 = torch._C._nn.gelu(g_29, approximate="none")
        g_29 = None
        cat_29 = torch.cat((i_29, c_119), dim=-1)
        i_29 = c_119 = None
        mul_58 = gelu_30 * cat_29
        gelu_30 = cat_29 = None
        x_131 = torch._C._nn.linear(
            mul_58,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_fc2_parameters_bias_,
        )
        mul_58 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_fc2_parameters_bias_ = (None)
        x_132 = (
            x_131
            * l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_ls_parameters_gamma_
        )
        x_131 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_ls_parameters_gamma_ = (None)
        input_30 = x_132 + input_29
        x_132 = input_29 = None
        x_133 = torch.nn.functional.layer_norm(
            input_30,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_bias_ = (None)
        x_134 = torch._C._nn.linear(
            x_133,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_fc1_parameters_bias_,
        )
        x_133 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_fc1_parameters_bias_ = (None)
        split_30 = torch.functional.split(x_134, (1024, 640, 384), dim=-1)
        x_134 = None
        g_30 = split_30[0]
        i_30 = split_30[1]
        c_120 = split_30[2]
        split_30 = None
        c_121 = c_120.permute(0, 3, 1, 2)
        c_120 = None
        c_122 = torch.conv2d(
            c_121,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_121 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_conv_parameters_bias_ = (None)
        c_123 = c_122.permute(0, 2, 3, 1)
        c_122 = None
        gelu_31 = torch._C._nn.gelu(g_30, approximate="none")
        g_30 = None
        cat_30 = torch.cat((i_30, c_123), dim=-1)
        i_30 = c_123 = None
        mul_60 = gelu_31 * cat_30
        gelu_31 = cat_30 = None
        x_135 = torch._C._nn.linear(
            mul_60,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_fc2_parameters_bias_,
        )
        mul_60 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_fc2_parameters_bias_ = (None)
        x_136 = (
            x_135
            * l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_ls_parameters_gamma_
        )
        x_135 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_ls_parameters_gamma_ = (None)
        input_31 = x_136 + input_30
        x_136 = input_30 = None
        x_137 = torch.nn.functional.layer_norm(
            input_31,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_bias_ = (None)
        x_138 = torch._C._nn.linear(
            x_137,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_fc1_parameters_bias_,
        )
        x_137 = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_fc1_parameters_bias_ = (None)
        split_31 = torch.functional.split(x_138, (1024, 640, 384), dim=-1)
        x_138 = None
        g_31 = split_31[0]
        i_31 = split_31[1]
        c_124 = split_31[2]
        split_31 = None
        c_125 = c_124.permute(0, 3, 1, 2)
        c_124 = None
        c_126 = torch.conv2d(
            c_125,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_125 = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_conv_parameters_bias_ = (None)
        c_127 = c_126.permute(0, 2, 3, 1)
        c_126 = None
        gelu_32 = torch._C._nn.gelu(g_31, approximate="none")
        g_31 = None
        cat_31 = torch.cat((i_31, c_127), dim=-1)
        i_31 = c_127 = None
        mul_62 = gelu_32 * cat_31
        gelu_32 = cat_31 = None
        x_139 = torch._C._nn.linear(
            mul_62,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_fc2_parameters_bias_,
        )
        mul_62 = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_fc2_parameters_bias_ = (None)
        x_140 = (
            x_139
            * l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_ls_parameters_gamma_
        )
        x_139 = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_ls_parameters_gamma_ = (None)
        input_32 = x_140 + input_31
        x_140 = input_31 = None
        x_141 = torch.nn.functional.layer_norm(
            input_32,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_bias_ = (None)
        x_142 = torch._C._nn.linear(
            x_141,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_fc1_parameters_bias_,
        )
        x_141 = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_fc1_parameters_bias_ = (None)
        split_32 = torch.functional.split(x_142, (1024, 640, 384), dim=-1)
        x_142 = None
        g_32 = split_32[0]
        i_32 = split_32[1]
        c_128 = split_32[2]
        split_32 = None
        c_129 = c_128.permute(0, 3, 1, 2)
        c_128 = None
        c_130 = torch.conv2d(
            c_129,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_129 = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_conv_parameters_bias_ = (None)
        c_131 = c_130.permute(0, 2, 3, 1)
        c_130 = None
        gelu_33 = torch._C._nn.gelu(g_32, approximate="none")
        g_32 = None
        cat_32 = torch.cat((i_32, c_131), dim=-1)
        i_32 = c_131 = None
        mul_64 = gelu_33 * cat_32
        gelu_33 = cat_32 = None
        x_143 = torch._C._nn.linear(
            mul_64,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_fc2_parameters_bias_,
        )
        mul_64 = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_fc2_parameters_bias_ = (None)
        x_144 = (
            x_143
            * l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_ls_parameters_gamma_
        )
        x_143 = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_ls_parameters_gamma_ = (None)
        input_33 = x_144 + input_32
        x_144 = input_32 = None
        x_145 = torch.nn.functional.layer_norm(
            input_33,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_bias_ = (None)
        x_146 = torch._C._nn.linear(
            x_145,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_fc1_parameters_bias_,
        )
        x_145 = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_fc1_parameters_bias_ = (None)
        split_33 = torch.functional.split(x_146, (1024, 640, 384), dim=-1)
        x_146 = None
        g_33 = split_33[0]
        i_33 = split_33[1]
        c_132 = split_33[2]
        split_33 = None
        c_133 = c_132.permute(0, 3, 1, 2)
        c_132 = None
        c_134 = torch.conv2d(
            c_133,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        c_133 = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_conv_parameters_bias_ = (None)
        c_135 = c_134.permute(0, 2, 3, 1)
        c_134 = None
        gelu_34 = torch._C._nn.gelu(g_33, approximate="none")
        g_33 = None
        cat_33 = torch.cat((i_33, c_135), dim=-1)
        i_33 = c_135 = None
        mul_66 = gelu_34 * cat_33
        gelu_34 = cat_33 = None
        x_147 = torch._C._nn.linear(
            mul_66,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_fc2_parameters_bias_,
        )
        mul_66 = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_fc2_parameters_bias_ = (None)
        x_148 = (
            x_147
            * l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_ls_parameters_gamma_
        )
        x_147 = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_ls_parameters_gamma_ = (None)
        input_34 = x_148 + input_33
        x_148 = input_33 = None
        x_149 = torch.nn.functional.layer_norm(
            input_34,
            (384,),
            l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_,
            1e-06,
        )
        input_34 = l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_ = (None)
        x_150 = x_149.permute(0, 3, 1, 2)
        x_149 = None
        x_151 = torch.conv2d(
            x_150,
            l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_150 = l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_ = (None)
        x_152 = x_151.permute(0, 2, 3, 1)
        x_151 = None
        x_153 = torch.nn.functional.layer_norm(
            x_152,
            (576,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_ = (None)
        x_154 = torch._C._nn.linear(
            x_153,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc1_parameters_bias_,
        )
        x_153 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc1_parameters_bias_ = (None)
        split_34 = torch.functional.split(x_154, (1536, 960, 576), dim=-1)
        x_154 = None
        g_34 = split_34[0]
        i_34 = split_34[1]
        c_136 = split_34[2]
        split_34 = None
        c_137 = c_136.permute(0, 3, 1, 2)
        c_136 = None
        c_138 = torch.conv2d(
            c_137,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            576,
        )
        c_137 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_parameters_bias_ = (None)
        c_139 = c_138.permute(0, 2, 3, 1)
        c_138 = None
        gelu_35 = torch._C._nn.gelu(g_34, approximate="none")
        g_34 = None
        cat_34 = torch.cat((i_34, c_139), dim=-1)
        i_34 = c_139 = None
        mul_68 = gelu_35 * cat_34
        gelu_35 = cat_34 = None
        x_155 = torch._C._nn.linear(
            mul_68,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc2_parameters_bias_,
        )
        mul_68 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc2_parameters_bias_ = (None)
        x_156 = (
            x_155
            * l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_ls_parameters_gamma_
        )
        x_155 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_ls_parameters_gamma_ = (None)
        input_35 = x_156 + x_152
        x_156 = x_152 = None
        x_157 = torch.nn.functional.layer_norm(
            input_35,
            (576,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_ = (None)
        x_158 = torch._C._nn.linear(
            x_157,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc1_parameters_bias_,
        )
        x_157 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc1_parameters_bias_ = (None)
        split_35 = torch.functional.split(x_158, (1536, 960, 576), dim=-1)
        x_158 = None
        g_35 = split_35[0]
        i_35 = split_35[1]
        c_140 = split_35[2]
        split_35 = None
        c_141 = c_140.permute(0, 3, 1, 2)
        c_140 = None
        c_142 = torch.conv2d(
            c_141,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            576,
        )
        c_141 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_parameters_bias_ = (None)
        c_143 = c_142.permute(0, 2, 3, 1)
        c_142 = None
        gelu_36 = torch._C._nn.gelu(g_35, approximate="none")
        g_35 = None
        cat_35 = torch.cat((i_35, c_143), dim=-1)
        i_35 = c_143 = None
        mul_70 = gelu_36 * cat_35
        gelu_36 = cat_35 = None
        x_159 = torch._C._nn.linear(
            mul_70,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc2_parameters_bias_,
        )
        mul_70 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc2_parameters_bias_ = (None)
        x_160 = (
            x_159
            * l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_ls_parameters_gamma_
        )
        x_159 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_ls_parameters_gamma_ = (None)
        input_36 = x_160 + input_35
        x_160 = input_35 = None
        x_161 = torch.nn.functional.layer_norm(
            input_36,
            (576,),
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_bias_ = (None)
        x_162 = torch._C._nn.linear(
            x_161,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc1_parameters_bias_,
        )
        x_161 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc1_parameters_bias_ = (None)
        split_36 = torch.functional.split(x_162, (1536, 960, 576), dim=-1)
        x_162 = None
        g_36 = split_36[0]
        i_36 = split_36[1]
        c_144 = split_36[2]
        split_36 = None
        c_145 = c_144.permute(0, 3, 1, 2)
        c_144 = None
        c_146 = torch.conv2d(
            c_145,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            576,
        )
        c_145 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_parameters_bias_ = (None)
        c_147 = c_146.permute(0, 2, 3, 1)
        c_146 = None
        gelu_37 = torch._C._nn.gelu(g_36, approximate="none")
        g_36 = None
        cat_36 = torch.cat((i_36, c_147), dim=-1)
        i_36 = c_147 = None
        mul_72 = gelu_37 * cat_36
        gelu_37 = cat_36 = None
        x_163 = torch._C._nn.linear(
            mul_72,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc2_parameters_bias_,
        )
        mul_72 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc2_parameters_bias_ = (None)
        x_164 = (
            x_163
            * l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_ls_parameters_gamma_
        )
        x_163 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_ls_parameters_gamma_ = (None)
        input_37 = x_164 + input_36
        x_164 = input_36 = None
        x_165 = input_37.mean(dim=(1, 2))
        input_37 = None
        x_166 = torch.nn.functional.layer_norm(
            x_165,
            (576,),
            l_self_modules_head_modules_norm_parameters_weight_,
            l_self_modules_head_modules_norm_parameters_bias_,
            1e-06,
        )
        x_165 = (
            l_self_modules_head_modules_norm_parameters_weight_
        ) = l_self_modules_head_modules_norm_parameters_bias_ = None
        input_38 = torch._C._nn.linear(
            x_166,
            l_self_modules_head_modules_pre_logits_modules_fc_parameters_weight_,
            l_self_modules_head_modules_pre_logits_modules_fc_parameters_bias_,
        )
        x_166 = (
            l_self_modules_head_modules_pre_logits_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_pre_logits_modules_fc_parameters_bias_ = None
        input_39 = torch._C._nn.gelu(input_38)
        input_38 = None
        x_167 = torch.nn.functional.dropout(input_39, 0.0, False, False)
        input_39 = None
        x_168 = torch._C._nn.linear(
            x_167,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_167 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_168,)
