import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_weight_ = L_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_weight_
        l_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_bias_ = L_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_bias_
        l_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_weight_ = L_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_weight_
        l_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_bias_ = L_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_weight_ = L_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_bias_ = L_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_bias_
        l_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_weight_ = L_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_bias_ = L_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_weight_ = L_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_weight_
        l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_bias_ = L_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_bias_
        l_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_weight_ = L_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_weight_
        l_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_bias_ = L_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_dw_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_dw_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_dw_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_dw_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_head_modules_norm_parameters_weight_ = (
            L_self_modules_head_modules_norm_parameters_weight_
        )
        l_self_modules_head_modules_norm_parameters_bias_ = (
            L_self_modules_head_modules_norm_parameters_bias_
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
        x_1 = torch.nn.functional.layer_norm(
            x,
            (128,),
            l_self_modules_stem_modules_1_parameters_weight_,
            l_self_modules_stem_modules_1_parameters_bias_,
            1e-06,
        )
        x = (
            l_self_modules_stem_modules_1_parameters_weight_
        ) = l_self_modules_stem_modules_1_parameters_bias_ = None
        x_2 = x_1.permute(0, 3, 1, 2)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_bias_ = (None)
        x_4 = x_3.permute(0, 2, 3, 1)
        x_3 = None
        x_5 = torch.nn.functional.layer_norm(
            x_4,
            (128,),
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_bias_,
            1e-06,
        )
        x_4 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_bias_ = (None)
        x_6 = torch._C._nn.linear(
            x_5,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_5 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_7 = torch._C._nn.gelu(x_6)
        x_6 = None
        x_8 = torch.nn.functional.dropout(x_7, 0.0, False, False)
        x_7 = None
        x_g = x_8.norm(p=2, dim=(1, 2), keepdim=True)
        mean = x_g.mean(dim=-1, keepdim=True)
        add = mean + 1e-06
        mean = None
        x_n = x_g / add
        x_g = add = None
        view = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_1 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul = x_8 * x_n
        x_n = None
        addcmul = torch.addcmul(view, view_1, mul)
        view = view_1 = mul = None
        x_9 = x_8 + addcmul
        x_8 = addcmul = None
        x_10 = torch._C._nn.linear(
            x_9,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_9 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_11 = torch.nn.functional.dropout(x_10, 0.0, False, False)
        x_10 = None
        x_12 = x_11.permute(0, 3, 1, 2)
        x_11 = None
        x_13 = x_12 + x_2
        x_12 = x_2 = None
        x_14 = torch.conv2d(
            x_13,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_bias_ = (None)
        x_15 = x_14.permute(0, 2, 3, 1)
        x_14 = None
        x_16 = torch.nn.functional.layer_norm(
            x_15,
            (128,),
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_,
            1e-06,
        )
        x_15 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_ = (None)
        x_17 = torch._C._nn.linear(
            x_16,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_16 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_18 = torch._C._nn.gelu(x_17)
        x_17 = None
        x_19 = torch.nn.functional.dropout(x_18, 0.0, False, False)
        x_18 = None
        x_g_1 = x_19.norm(p=2, dim=(1, 2), keepdim=True)
        mean_1 = x_g_1.mean(dim=-1, keepdim=True)
        add_3 = mean_1 + 1e-06
        mean_1 = None
        x_n_1 = x_g_1 / add_3
        x_g_1 = add_3 = None
        view_2 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_3 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_1 = x_19 * x_n_1
        x_n_1 = None
        addcmul_1 = torch.addcmul(view_2, view_3, mul_1)
        view_2 = view_3 = mul_1 = None
        x_20 = x_19 + addcmul_1
        x_19 = addcmul_1 = None
        x_21 = torch._C._nn.linear(
            x_20,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_20 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_22 = torch.nn.functional.dropout(x_21, 0.0, False, False)
        x_21 = None
        x_23 = x_22.permute(0, 3, 1, 2)
        x_22 = None
        x_24 = x_23 + x_13
        x_23 = x_13 = None
        x_25 = torch.conv2d(
            x_24,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_dw_parameters_bias_ = (None)
        x_26 = x_25.permute(0, 2, 3, 1)
        x_25 = None
        x_27 = torch.nn.functional.layer_norm(
            x_26,
            (128,),
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_bias_,
            1e-06,
        )
        x_26 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_bias_ = (None)
        x_28 = torch._C._nn.linear(
            x_27,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_27 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_29 = torch._C._nn.gelu(x_28)
        x_28 = None
        x_30 = torch.nn.functional.dropout(x_29, 0.0, False, False)
        x_29 = None
        x_g_2 = x_30.norm(p=2, dim=(1, 2), keepdim=True)
        mean_2 = x_g_2.mean(dim=-1, keepdim=True)
        add_6 = mean_2 + 1e-06
        mean_2 = None
        x_n_2 = x_g_2 / add_6
        x_g_2 = add_6 = None
        view_4 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_5 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_2 = x_30 * x_n_2
        x_n_2 = None
        addcmul_2 = torch.addcmul(view_4, view_5, mul_2)
        view_4 = view_5 = mul_2 = None
        x_31 = x_30 + addcmul_2
        x_30 = addcmul_2 = None
        x_32 = torch._C._nn.linear(
            x_31,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_31 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_33 = torch.nn.functional.dropout(x_32, 0.0, False, False)
        x_32 = None
        x_34 = x_33.permute(0, 3, 1, 2)
        x_33 = None
        x_35 = x_34 + x_24
        x_34 = x_24 = None
        x_36 = x_35.permute(0, 2, 3, 1)
        x_35 = None
        x_37 = torch.nn.functional.layer_norm(
            x_36,
            (128,),
            l_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_weight_,
            l_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_bias_,
            1e-06,
        )
        x_36 = l_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_weight_ = l_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_bias_ = (None)
        x_38 = x_37.permute(0, 3, 1, 2)
        x_37 = None
        input_2 = torch.conv2d(
            x_38,
            l_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_38 = l_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_weight_ = l_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_bias_ = (None)
        x_39 = torch.conv2d(
            input_2,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_bias_ = (None)
        x_40 = x_39.permute(0, 2, 3, 1)
        x_39 = None
        x_41 = torch.nn.functional.layer_norm(
            x_40,
            (256,),
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_,
            1e-06,
        )
        x_40 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_ = (None)
        x_42 = torch._C._nn.linear(
            x_41,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_41 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_43 = torch._C._nn.gelu(x_42)
        x_42 = None
        x_44 = torch.nn.functional.dropout(x_43, 0.0, False, False)
        x_43 = None
        x_g_3 = x_44.norm(p=2, dim=(1, 2), keepdim=True)
        mean_3 = x_g_3.mean(dim=-1, keepdim=True)
        add_9 = mean_3 + 1e-06
        mean_3 = None
        x_n_3 = x_g_3 / add_9
        x_g_3 = add_9 = None
        view_6 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_7 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_3 = x_44 * x_n_3
        x_n_3 = None
        addcmul_3 = torch.addcmul(view_6, view_7, mul_3)
        view_6 = view_7 = mul_3 = None
        x_45 = x_44 + addcmul_3
        x_44 = addcmul_3 = None
        x_46 = torch._C._nn.linear(
            x_45,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_45 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_47 = torch.nn.functional.dropout(x_46, 0.0, False, False)
        x_46 = None
        x_48 = x_47.permute(0, 3, 1, 2)
        x_47 = None
        x_49 = x_48 + input_2
        x_48 = input_2 = None
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_bias_ = (None)
        x_51 = x_50.permute(0, 2, 3, 1)
        x_50 = None
        x_52 = torch.nn.functional.layer_norm(
            x_51,
            (256,),
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_,
            1e-06,
        )
        x_51 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_ = (None)
        x_53 = torch._C._nn.linear(
            x_52,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_52 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_54 = torch._C._nn.gelu(x_53)
        x_53 = None
        x_55 = torch.nn.functional.dropout(x_54, 0.0, False, False)
        x_54 = None
        x_g_4 = x_55.norm(p=2, dim=(1, 2), keepdim=True)
        mean_4 = x_g_4.mean(dim=-1, keepdim=True)
        add_12 = mean_4 + 1e-06
        mean_4 = None
        x_n_4 = x_g_4 / add_12
        x_g_4 = add_12 = None
        view_8 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_9 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_4 = x_55 * x_n_4
        x_n_4 = None
        addcmul_4 = torch.addcmul(view_8, view_9, mul_4)
        view_8 = view_9 = mul_4 = None
        x_56 = x_55 + addcmul_4
        x_55 = addcmul_4 = None
        x_57 = torch._C._nn.linear(
            x_56,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_56 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_58 = torch.nn.functional.dropout(x_57, 0.0, False, False)
        x_57 = None
        x_59 = x_58.permute(0, 3, 1, 2)
        x_58 = None
        x_60 = x_59 + x_49
        x_59 = x_49 = None
        x_61 = torch.conv2d(
            x_60,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_dw_parameters_bias_ = (None)
        x_62 = x_61.permute(0, 2, 3, 1)
        x_61 = None
        x_63 = torch.nn.functional.layer_norm(
            x_62,
            (256,),
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_bias_,
            1e-06,
        )
        x_62 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_bias_ = (None)
        x_64 = torch._C._nn.linear(
            x_63,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_63 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_65 = torch._C._nn.gelu(x_64)
        x_64 = None
        x_66 = torch.nn.functional.dropout(x_65, 0.0, False, False)
        x_65 = None
        x_g_5 = x_66.norm(p=2, dim=(1, 2), keepdim=True)
        mean_5 = x_g_5.mean(dim=-1, keepdim=True)
        add_15 = mean_5 + 1e-06
        mean_5 = None
        x_n_5 = x_g_5 / add_15
        x_g_5 = add_15 = None
        view_10 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_11 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_5 = x_66 * x_n_5
        x_n_5 = None
        addcmul_5 = torch.addcmul(view_10, view_11, mul_5)
        view_10 = view_11 = mul_5 = None
        x_67 = x_66 + addcmul_5
        x_66 = addcmul_5 = None
        x_68 = torch._C._nn.linear(
            x_67,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_67 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_69 = torch.nn.functional.dropout(x_68, 0.0, False, False)
        x_68 = None
        x_70 = x_69.permute(0, 3, 1, 2)
        x_69 = None
        x_71 = x_70 + x_60
        x_70 = x_60 = None
        x_72 = x_71.permute(0, 2, 3, 1)
        x_71 = None
        x_73 = torch.nn.functional.layer_norm(
            x_72,
            (256,),
            l_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_weight_,
            l_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_bias_,
            1e-06,
        )
        x_72 = l_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_weight_ = l_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_bias_ = (None)
        x_74 = x_73.permute(0, 3, 1, 2)
        x_73 = None
        input_3 = torch.conv2d(
            x_74,
            l_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_74 = l_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_weight_ = l_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_bias_ = (None)
        x_75 = torch.conv2d(
            input_3,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_bias_ = (None)
        x_76 = x_75.permute(0, 2, 3, 1)
        x_75 = None
        x_77 = torch.nn.functional.layer_norm(
            x_76,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_,
            1e-06,
        )
        x_76 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_ = (None)
        x_78 = torch._C._nn.linear(
            x_77,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_77 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_79 = torch._C._nn.gelu(x_78)
        x_78 = None
        x_80 = torch.nn.functional.dropout(x_79, 0.0, False, False)
        x_79 = None
        x_g_6 = x_80.norm(p=2, dim=(1, 2), keepdim=True)
        mean_6 = x_g_6.mean(dim=-1, keepdim=True)
        add_18 = mean_6 + 1e-06
        mean_6 = None
        x_n_6 = x_g_6 / add_18
        x_g_6 = add_18 = None
        view_12 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_13 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_6 = x_80 * x_n_6
        x_n_6 = None
        addcmul_6 = torch.addcmul(view_12, view_13, mul_6)
        view_12 = view_13 = mul_6 = None
        x_81 = x_80 + addcmul_6
        x_80 = addcmul_6 = None
        x_82 = torch._C._nn.linear(
            x_81,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_81 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_83 = torch.nn.functional.dropout(x_82, 0.0, False, False)
        x_82 = None
        x_84 = x_83.permute(0, 3, 1, 2)
        x_83 = None
        x_85 = x_84 + input_3
        x_84 = input_3 = None
        x_86 = torch.conv2d(
            x_85,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_bias_ = (None)
        x_87 = x_86.permute(0, 2, 3, 1)
        x_86 = None
        x_88 = torch.nn.functional.layer_norm(
            x_87,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_,
            1e-06,
        )
        x_87 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_ = (None)
        x_89 = torch._C._nn.linear(
            x_88,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_88 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_90 = torch._C._nn.gelu(x_89)
        x_89 = None
        x_91 = torch.nn.functional.dropout(x_90, 0.0, False, False)
        x_90 = None
        x_g_7 = x_91.norm(p=2, dim=(1, 2), keepdim=True)
        mean_7 = x_g_7.mean(dim=-1, keepdim=True)
        add_21 = mean_7 + 1e-06
        mean_7 = None
        x_n_7 = x_g_7 / add_21
        x_g_7 = add_21 = None
        view_14 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_15 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_7 = x_91 * x_n_7
        x_n_7 = None
        addcmul_7 = torch.addcmul(view_14, view_15, mul_7)
        view_14 = view_15 = mul_7 = None
        x_92 = x_91 + addcmul_7
        x_91 = addcmul_7 = None
        x_93 = torch._C._nn.linear(
            x_92,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_92 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_94 = torch.nn.functional.dropout(x_93, 0.0, False, False)
        x_93 = None
        x_95 = x_94.permute(0, 3, 1, 2)
        x_94 = None
        x_96 = x_95 + x_85
        x_95 = x_85 = None
        x_97 = torch.conv2d(
            x_96,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_bias_ = (None)
        x_98 = x_97.permute(0, 2, 3, 1)
        x_97 = None
        x_99 = torch.nn.functional.layer_norm(
            x_98,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_,
            1e-06,
        )
        x_98 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_ = (None)
        x_100 = torch._C._nn.linear(
            x_99,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_99 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_101 = torch._C._nn.gelu(x_100)
        x_100 = None
        x_102 = torch.nn.functional.dropout(x_101, 0.0, False, False)
        x_101 = None
        x_g_8 = x_102.norm(p=2, dim=(1, 2), keepdim=True)
        mean_8 = x_g_8.mean(dim=-1, keepdim=True)
        add_24 = mean_8 + 1e-06
        mean_8 = None
        x_n_8 = x_g_8 / add_24
        x_g_8 = add_24 = None
        view_16 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_17 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_8 = x_102 * x_n_8
        x_n_8 = None
        addcmul_8 = torch.addcmul(view_16, view_17, mul_8)
        view_16 = view_17 = mul_8 = None
        x_103 = x_102 + addcmul_8
        x_102 = addcmul_8 = None
        x_104 = torch._C._nn.linear(
            x_103,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_103 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_105 = torch.nn.functional.dropout(x_104, 0.0, False, False)
        x_104 = None
        x_106 = x_105.permute(0, 3, 1, 2)
        x_105 = None
        x_107 = x_106 + x_96
        x_106 = x_96 = None
        x_108 = torch.conv2d(
            x_107,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_bias_ = (None)
        x_109 = x_108.permute(0, 2, 3, 1)
        x_108 = None
        x_110 = torch.nn.functional.layer_norm(
            x_109,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_,
            1e-06,
        )
        x_109 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_ = (None)
        x_111 = torch._C._nn.linear(
            x_110,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_110 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_112 = torch._C._nn.gelu(x_111)
        x_111 = None
        x_113 = torch.nn.functional.dropout(x_112, 0.0, False, False)
        x_112 = None
        x_g_9 = x_113.norm(p=2, dim=(1, 2), keepdim=True)
        mean_9 = x_g_9.mean(dim=-1, keepdim=True)
        add_27 = mean_9 + 1e-06
        mean_9 = None
        x_n_9 = x_g_9 / add_27
        x_g_9 = add_27 = None
        view_18 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_19 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_9 = x_113 * x_n_9
        x_n_9 = None
        addcmul_9 = torch.addcmul(view_18, view_19, mul_9)
        view_18 = view_19 = mul_9 = None
        x_114 = x_113 + addcmul_9
        x_113 = addcmul_9 = None
        x_115 = torch._C._nn.linear(
            x_114,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_114 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_116 = torch.nn.functional.dropout(x_115, 0.0, False, False)
        x_115 = None
        x_117 = x_116.permute(0, 3, 1, 2)
        x_116 = None
        x_118 = x_117 + x_107
        x_117 = x_107 = None
        x_119 = torch.conv2d(
            x_118,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_dw_parameters_bias_ = (None)
        x_120 = x_119.permute(0, 2, 3, 1)
        x_119 = None
        x_121 = torch.nn.functional.layer_norm(
            x_120,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_,
            1e-06,
        )
        x_120 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_ = (None)
        x_122 = torch._C._nn.linear(
            x_121,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_121 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_123 = torch._C._nn.gelu(x_122)
        x_122 = None
        x_124 = torch.nn.functional.dropout(x_123, 0.0, False, False)
        x_123 = None
        x_g_10 = x_124.norm(p=2, dim=(1, 2), keepdim=True)
        mean_10 = x_g_10.mean(dim=-1, keepdim=True)
        add_30 = mean_10 + 1e-06
        mean_10 = None
        x_n_10 = x_g_10 / add_30
        x_g_10 = add_30 = None
        view_20 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_21 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_10 = x_124 * x_n_10
        x_n_10 = None
        addcmul_10 = torch.addcmul(view_20, view_21, mul_10)
        view_20 = view_21 = mul_10 = None
        x_125 = x_124 + addcmul_10
        x_124 = addcmul_10 = None
        x_126 = torch._C._nn.linear(
            x_125,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_125 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_127 = torch.nn.functional.dropout(x_126, 0.0, False, False)
        x_126 = None
        x_128 = x_127.permute(0, 3, 1, 2)
        x_127 = None
        x_129 = x_128 + x_118
        x_128 = x_118 = None
        x_130 = torch.conv2d(
            x_129,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_dw_parameters_bias_ = (None)
        x_131 = x_130.permute(0, 2, 3, 1)
        x_130 = None
        x_132 = torch.nn.functional.layer_norm(
            x_131,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_,
            1e-06,
        )
        x_131 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_ = (None)
        x_133 = torch._C._nn.linear(
            x_132,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_132 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_134 = torch._C._nn.gelu(x_133)
        x_133 = None
        x_135 = torch.nn.functional.dropout(x_134, 0.0, False, False)
        x_134 = None
        x_g_11 = x_135.norm(p=2, dim=(1, 2), keepdim=True)
        mean_11 = x_g_11.mean(dim=-1, keepdim=True)
        add_33 = mean_11 + 1e-06
        mean_11 = None
        x_n_11 = x_g_11 / add_33
        x_g_11 = add_33 = None
        view_22 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_23 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_11 = x_135 * x_n_11
        x_n_11 = None
        addcmul_11 = torch.addcmul(view_22, view_23, mul_11)
        view_22 = view_23 = mul_11 = None
        x_136 = x_135 + addcmul_11
        x_135 = addcmul_11 = None
        x_137 = torch._C._nn.linear(
            x_136,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_136 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_138 = torch.nn.functional.dropout(x_137, 0.0, False, False)
        x_137 = None
        x_139 = x_138.permute(0, 3, 1, 2)
        x_138 = None
        x_140 = x_139 + x_129
        x_139 = x_129 = None
        x_141 = torch.conv2d(
            x_140,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_dw_parameters_bias_ = (None)
        x_142 = x_141.permute(0, 2, 3, 1)
        x_141 = None
        x_143 = torch.nn.functional.layer_norm(
            x_142,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_bias_,
            1e-06,
        )
        x_142 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_bias_ = (None)
        x_144 = torch._C._nn.linear(
            x_143,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_143 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_145 = torch._C._nn.gelu(x_144)
        x_144 = None
        x_146 = torch.nn.functional.dropout(x_145, 0.0, False, False)
        x_145 = None
        x_g_12 = x_146.norm(p=2, dim=(1, 2), keepdim=True)
        mean_12 = x_g_12.mean(dim=-1, keepdim=True)
        add_36 = mean_12 + 1e-06
        mean_12 = None
        x_n_12 = x_g_12 / add_36
        x_g_12 = add_36 = None
        view_24 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_25 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_12 = x_146 * x_n_12
        x_n_12 = None
        addcmul_12 = torch.addcmul(view_24, view_25, mul_12)
        view_24 = view_25 = mul_12 = None
        x_147 = x_146 + addcmul_12
        x_146 = addcmul_12 = None
        x_148 = torch._C._nn.linear(
            x_147,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_147 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_149 = torch.nn.functional.dropout(x_148, 0.0, False, False)
        x_148 = None
        x_150 = x_149.permute(0, 3, 1, 2)
        x_149 = None
        x_151 = x_150 + x_140
        x_150 = x_140 = None
        x_152 = torch.conv2d(
            x_151,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_dw_parameters_bias_ = (None)
        x_153 = x_152.permute(0, 2, 3, 1)
        x_152 = None
        x_154 = torch.nn.functional.layer_norm(
            x_153,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_bias_,
            1e-06,
        )
        x_153 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_bias_ = (None)
        x_155 = torch._C._nn.linear(
            x_154,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_154 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_156 = torch._C._nn.gelu(x_155)
        x_155 = None
        x_157 = torch.nn.functional.dropout(x_156, 0.0, False, False)
        x_156 = None
        x_g_13 = x_157.norm(p=2, dim=(1, 2), keepdim=True)
        mean_13 = x_g_13.mean(dim=-1, keepdim=True)
        add_39 = mean_13 + 1e-06
        mean_13 = None
        x_n_13 = x_g_13 / add_39
        x_g_13 = add_39 = None
        view_26 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_27 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_13 = x_157 * x_n_13
        x_n_13 = None
        addcmul_13 = torch.addcmul(view_26, view_27, mul_13)
        view_26 = view_27 = mul_13 = None
        x_158 = x_157 + addcmul_13
        x_157 = addcmul_13 = None
        x_159 = torch._C._nn.linear(
            x_158,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_158 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_160 = torch.nn.functional.dropout(x_159, 0.0, False, False)
        x_159 = None
        x_161 = x_160.permute(0, 3, 1, 2)
        x_160 = None
        x_162 = x_161 + x_151
        x_161 = x_151 = None
        x_163 = torch.conv2d(
            x_162,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_dw_parameters_bias_ = (None)
        x_164 = x_163.permute(0, 2, 3, 1)
        x_163 = None
        x_165 = torch.nn.functional.layer_norm(
            x_164,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_bias_,
            1e-06,
        )
        x_164 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_bias_ = (None)
        x_166 = torch._C._nn.linear(
            x_165,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_165 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_167 = torch._C._nn.gelu(x_166)
        x_166 = None
        x_168 = torch.nn.functional.dropout(x_167, 0.0, False, False)
        x_167 = None
        x_g_14 = x_168.norm(p=2, dim=(1, 2), keepdim=True)
        mean_14 = x_g_14.mean(dim=-1, keepdim=True)
        add_42 = mean_14 + 1e-06
        mean_14 = None
        x_n_14 = x_g_14 / add_42
        x_g_14 = add_42 = None
        view_28 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_29 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_14 = x_168 * x_n_14
        x_n_14 = None
        addcmul_14 = torch.addcmul(view_28, view_29, mul_14)
        view_28 = view_29 = mul_14 = None
        x_169 = x_168 + addcmul_14
        x_168 = addcmul_14 = None
        x_170 = torch._C._nn.linear(
            x_169,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_169 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_171 = torch.nn.functional.dropout(x_170, 0.0, False, False)
        x_170 = None
        x_172 = x_171.permute(0, 3, 1, 2)
        x_171 = None
        x_173 = x_172 + x_162
        x_172 = x_162 = None
        x_174 = torch.conv2d(
            x_173,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_dw_parameters_bias_ = (None)
        x_175 = x_174.permute(0, 2, 3, 1)
        x_174 = None
        x_176 = torch.nn.functional.layer_norm(
            x_175,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_bias_,
            1e-06,
        )
        x_175 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_bias_ = (None)
        x_177 = torch._C._nn.linear(
            x_176,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_176 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_178 = torch._C._nn.gelu(x_177)
        x_177 = None
        x_179 = torch.nn.functional.dropout(x_178, 0.0, False, False)
        x_178 = None
        x_g_15 = x_179.norm(p=2, dim=(1, 2), keepdim=True)
        mean_15 = x_g_15.mean(dim=-1, keepdim=True)
        add_45 = mean_15 + 1e-06
        mean_15 = None
        x_n_15 = x_g_15 / add_45
        x_g_15 = add_45 = None
        view_30 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_31 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_15 = x_179 * x_n_15
        x_n_15 = None
        addcmul_15 = torch.addcmul(view_30, view_31, mul_15)
        view_30 = view_31 = mul_15 = None
        x_180 = x_179 + addcmul_15
        x_179 = addcmul_15 = None
        x_181 = torch._C._nn.linear(
            x_180,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_180 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_182 = torch.nn.functional.dropout(x_181, 0.0, False, False)
        x_181 = None
        x_183 = x_182.permute(0, 3, 1, 2)
        x_182 = None
        x_184 = x_183 + x_173
        x_183 = x_173 = None
        x_185 = torch.conv2d(
            x_184,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_dw_parameters_bias_ = (None)
        x_186 = x_185.permute(0, 2, 3, 1)
        x_185 = None
        x_187 = torch.nn.functional.layer_norm(
            x_186,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_bias_,
            1e-06,
        )
        x_186 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_bias_ = (None)
        x_188 = torch._C._nn.linear(
            x_187,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_187 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_189 = torch._C._nn.gelu(x_188)
        x_188 = None
        x_190 = torch.nn.functional.dropout(x_189, 0.0, False, False)
        x_189 = None
        x_g_16 = x_190.norm(p=2, dim=(1, 2), keepdim=True)
        mean_16 = x_g_16.mean(dim=-1, keepdim=True)
        add_48 = mean_16 + 1e-06
        mean_16 = None
        x_n_16 = x_g_16 / add_48
        x_g_16 = add_48 = None
        view_32 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_33 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_16 = x_190 * x_n_16
        x_n_16 = None
        addcmul_16 = torch.addcmul(view_32, view_33, mul_16)
        view_32 = view_33 = mul_16 = None
        x_191 = x_190 + addcmul_16
        x_190 = addcmul_16 = None
        x_192 = torch._C._nn.linear(
            x_191,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_191 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_193 = torch.nn.functional.dropout(x_192, 0.0, False, False)
        x_192 = None
        x_194 = x_193.permute(0, 3, 1, 2)
        x_193 = None
        x_195 = x_194 + x_184
        x_194 = x_184 = None
        x_196 = torch.conv2d(
            x_195,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_dw_parameters_bias_ = (None)
        x_197 = x_196.permute(0, 2, 3, 1)
        x_196 = None
        x_198 = torch.nn.functional.layer_norm(
            x_197,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_bias_,
            1e-06,
        )
        x_197 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_bias_ = (None)
        x_199 = torch._C._nn.linear(
            x_198,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_198 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_200 = torch._C._nn.gelu(x_199)
        x_199 = None
        x_201 = torch.nn.functional.dropout(x_200, 0.0, False, False)
        x_200 = None
        x_g_17 = x_201.norm(p=2, dim=(1, 2), keepdim=True)
        mean_17 = x_g_17.mean(dim=-1, keepdim=True)
        add_51 = mean_17 + 1e-06
        mean_17 = None
        x_n_17 = x_g_17 / add_51
        x_g_17 = add_51 = None
        view_34 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_35 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_17 = x_201 * x_n_17
        x_n_17 = None
        addcmul_17 = torch.addcmul(view_34, view_35, mul_17)
        view_34 = view_35 = mul_17 = None
        x_202 = x_201 + addcmul_17
        x_201 = addcmul_17 = None
        x_203 = torch._C._nn.linear(
            x_202,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_202 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_204 = torch.nn.functional.dropout(x_203, 0.0, False, False)
        x_203 = None
        x_205 = x_204.permute(0, 3, 1, 2)
        x_204 = None
        x_206 = x_205 + x_195
        x_205 = x_195 = None
        x_207 = torch.conv2d(
            x_206,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_dw_parameters_bias_ = (None)
        x_208 = x_207.permute(0, 2, 3, 1)
        x_207 = None
        x_209 = torch.nn.functional.layer_norm(
            x_208,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_bias_,
            1e-06,
        )
        x_208 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_bias_ = (None)
        x_210 = torch._C._nn.linear(
            x_209,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_209 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_211 = torch._C._nn.gelu(x_210)
        x_210 = None
        x_212 = torch.nn.functional.dropout(x_211, 0.0, False, False)
        x_211 = None
        x_g_18 = x_212.norm(p=2, dim=(1, 2), keepdim=True)
        mean_18 = x_g_18.mean(dim=-1, keepdim=True)
        add_54 = mean_18 + 1e-06
        mean_18 = None
        x_n_18 = x_g_18 / add_54
        x_g_18 = add_54 = None
        view_36 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_37 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_18 = x_212 * x_n_18
        x_n_18 = None
        addcmul_18 = torch.addcmul(view_36, view_37, mul_18)
        view_36 = view_37 = mul_18 = None
        x_213 = x_212 + addcmul_18
        x_212 = addcmul_18 = None
        x_214 = torch._C._nn.linear(
            x_213,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_213 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_215 = torch.nn.functional.dropout(x_214, 0.0, False, False)
        x_214 = None
        x_216 = x_215.permute(0, 3, 1, 2)
        x_215 = None
        x_217 = x_216 + x_206
        x_216 = x_206 = None
        x_218 = torch.conv2d(
            x_217,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_dw_parameters_bias_ = (None)
        x_219 = x_218.permute(0, 2, 3, 1)
        x_218 = None
        x_220 = torch.nn.functional.layer_norm(
            x_219,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_bias_,
            1e-06,
        )
        x_219 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_bias_ = (None)
        x_221 = torch._C._nn.linear(
            x_220,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_220 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_222 = torch._C._nn.gelu(x_221)
        x_221 = None
        x_223 = torch.nn.functional.dropout(x_222, 0.0, False, False)
        x_222 = None
        x_g_19 = x_223.norm(p=2, dim=(1, 2), keepdim=True)
        mean_19 = x_g_19.mean(dim=-1, keepdim=True)
        add_57 = mean_19 + 1e-06
        mean_19 = None
        x_n_19 = x_g_19 / add_57
        x_g_19 = add_57 = None
        view_38 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_39 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_19 = x_223 * x_n_19
        x_n_19 = None
        addcmul_19 = torch.addcmul(view_38, view_39, mul_19)
        view_38 = view_39 = mul_19 = None
        x_224 = x_223 + addcmul_19
        x_223 = addcmul_19 = None
        x_225 = torch._C._nn.linear(
            x_224,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_224 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_226 = torch.nn.functional.dropout(x_225, 0.0, False, False)
        x_225 = None
        x_227 = x_226.permute(0, 3, 1, 2)
        x_226 = None
        x_228 = x_227 + x_217
        x_227 = x_217 = None
        x_229 = torch.conv2d(
            x_228,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_dw_parameters_bias_ = (None)
        x_230 = x_229.permute(0, 2, 3, 1)
        x_229 = None
        x_231 = torch.nn.functional.layer_norm(
            x_230,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_bias_,
            1e-06,
        )
        x_230 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_bias_ = (None)
        x_232 = torch._C._nn.linear(
            x_231,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_231 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_233 = torch._C._nn.gelu(x_232)
        x_232 = None
        x_234 = torch.nn.functional.dropout(x_233, 0.0, False, False)
        x_233 = None
        x_g_20 = x_234.norm(p=2, dim=(1, 2), keepdim=True)
        mean_20 = x_g_20.mean(dim=-1, keepdim=True)
        add_60 = mean_20 + 1e-06
        mean_20 = None
        x_n_20 = x_g_20 / add_60
        x_g_20 = add_60 = None
        view_40 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_41 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_20 = x_234 * x_n_20
        x_n_20 = None
        addcmul_20 = torch.addcmul(view_40, view_41, mul_20)
        view_40 = view_41 = mul_20 = None
        x_235 = x_234 + addcmul_20
        x_234 = addcmul_20 = None
        x_236 = torch._C._nn.linear(
            x_235,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_235 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_237 = torch.nn.functional.dropout(x_236, 0.0, False, False)
        x_236 = None
        x_238 = x_237.permute(0, 3, 1, 2)
        x_237 = None
        x_239 = x_238 + x_228
        x_238 = x_228 = None
        x_240 = torch.conv2d(
            x_239,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_conv_dw_parameters_bias_ = (None)
        x_241 = x_240.permute(0, 2, 3, 1)
        x_240 = None
        x_242 = torch.nn.functional.layer_norm(
            x_241,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_bias_,
            1e-06,
        )
        x_241 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_bias_ = (None)
        x_243 = torch._C._nn.linear(
            x_242,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_242 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_244 = torch._C._nn.gelu(x_243)
        x_243 = None
        x_245 = torch.nn.functional.dropout(x_244, 0.0, False, False)
        x_244 = None
        x_g_21 = x_245.norm(p=2, dim=(1, 2), keepdim=True)
        mean_21 = x_g_21.mean(dim=-1, keepdim=True)
        add_63 = mean_21 + 1e-06
        mean_21 = None
        x_n_21 = x_g_21 / add_63
        x_g_21 = add_63 = None
        view_42 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_43 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_21 = x_245 * x_n_21
        x_n_21 = None
        addcmul_21 = torch.addcmul(view_42, view_43, mul_21)
        view_42 = view_43 = mul_21 = None
        x_246 = x_245 + addcmul_21
        x_245 = addcmul_21 = None
        x_247 = torch._C._nn.linear(
            x_246,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_246 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_248 = torch.nn.functional.dropout(x_247, 0.0, False, False)
        x_247 = None
        x_249 = x_248.permute(0, 3, 1, 2)
        x_248 = None
        x_250 = x_249 + x_239
        x_249 = x_239 = None
        x_251 = torch.conv2d(
            x_250,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_conv_dw_parameters_bias_ = (None)
        x_252 = x_251.permute(0, 2, 3, 1)
        x_251 = None
        x_253 = torch.nn.functional.layer_norm(
            x_252,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_bias_,
            1e-06,
        )
        x_252 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_bias_ = (None)
        x_254 = torch._C._nn.linear(
            x_253,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_253 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_255 = torch._C._nn.gelu(x_254)
        x_254 = None
        x_256 = torch.nn.functional.dropout(x_255, 0.0, False, False)
        x_255 = None
        x_g_22 = x_256.norm(p=2, dim=(1, 2), keepdim=True)
        mean_22 = x_g_22.mean(dim=-1, keepdim=True)
        add_66 = mean_22 + 1e-06
        mean_22 = None
        x_n_22 = x_g_22 / add_66
        x_g_22 = add_66 = None
        view_44 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_45 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_22 = x_256 * x_n_22
        x_n_22 = None
        addcmul_22 = torch.addcmul(view_44, view_45, mul_22)
        view_44 = view_45 = mul_22 = None
        x_257 = x_256 + addcmul_22
        x_256 = addcmul_22 = None
        x_258 = torch._C._nn.linear(
            x_257,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_257 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_259 = torch.nn.functional.dropout(x_258, 0.0, False, False)
        x_258 = None
        x_260 = x_259.permute(0, 3, 1, 2)
        x_259 = None
        x_261 = x_260 + x_250
        x_260 = x_250 = None
        x_262 = torch.conv2d(
            x_261,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_conv_dw_parameters_bias_ = (None)
        x_263 = x_262.permute(0, 2, 3, 1)
        x_262 = None
        x_264 = torch.nn.functional.layer_norm(
            x_263,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_bias_,
            1e-06,
        )
        x_263 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_bias_ = (None)
        x_265 = torch._C._nn.linear(
            x_264,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_264 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_266 = torch._C._nn.gelu(x_265)
        x_265 = None
        x_267 = torch.nn.functional.dropout(x_266, 0.0, False, False)
        x_266 = None
        x_g_23 = x_267.norm(p=2, dim=(1, 2), keepdim=True)
        mean_23 = x_g_23.mean(dim=-1, keepdim=True)
        add_69 = mean_23 + 1e-06
        mean_23 = None
        x_n_23 = x_g_23 / add_69
        x_g_23 = add_69 = None
        view_46 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_47 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_23 = x_267 * x_n_23
        x_n_23 = None
        addcmul_23 = torch.addcmul(view_46, view_47, mul_23)
        view_46 = view_47 = mul_23 = None
        x_268 = x_267 + addcmul_23
        x_267 = addcmul_23 = None
        x_269 = torch._C._nn.linear(
            x_268,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_268 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_270 = torch.nn.functional.dropout(x_269, 0.0, False, False)
        x_269 = None
        x_271 = x_270.permute(0, 3, 1, 2)
        x_270 = None
        x_272 = x_271 + x_261
        x_271 = x_261 = None
        x_273 = torch.conv2d(
            x_272,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_conv_dw_parameters_bias_ = (None)
        x_274 = x_273.permute(0, 2, 3, 1)
        x_273 = None
        x_275 = torch.nn.functional.layer_norm(
            x_274,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_bias_,
            1e-06,
        )
        x_274 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_bias_ = (None)
        x_276 = torch._C._nn.linear(
            x_275,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_275 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_277 = torch._C._nn.gelu(x_276)
        x_276 = None
        x_278 = torch.nn.functional.dropout(x_277, 0.0, False, False)
        x_277 = None
        x_g_24 = x_278.norm(p=2, dim=(1, 2), keepdim=True)
        mean_24 = x_g_24.mean(dim=-1, keepdim=True)
        add_72 = mean_24 + 1e-06
        mean_24 = None
        x_n_24 = x_g_24 / add_72
        x_g_24 = add_72 = None
        view_48 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_49 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_24 = x_278 * x_n_24
        x_n_24 = None
        addcmul_24 = torch.addcmul(view_48, view_49, mul_24)
        view_48 = view_49 = mul_24 = None
        x_279 = x_278 + addcmul_24
        x_278 = addcmul_24 = None
        x_280 = torch._C._nn.linear(
            x_279,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_279 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_281 = torch.nn.functional.dropout(x_280, 0.0, False, False)
        x_280 = None
        x_282 = x_281.permute(0, 3, 1, 2)
        x_281 = None
        x_283 = x_282 + x_272
        x_282 = x_272 = None
        x_284 = torch.conv2d(
            x_283,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_conv_dw_parameters_bias_ = (None)
        x_285 = x_284.permute(0, 2, 3, 1)
        x_284 = None
        x_286 = torch.nn.functional.layer_norm(
            x_285,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_bias_,
            1e-06,
        )
        x_285 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_bias_ = (None)
        x_287 = torch._C._nn.linear(
            x_286,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_286 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_288 = torch._C._nn.gelu(x_287)
        x_287 = None
        x_289 = torch.nn.functional.dropout(x_288, 0.0, False, False)
        x_288 = None
        x_g_25 = x_289.norm(p=2, dim=(1, 2), keepdim=True)
        mean_25 = x_g_25.mean(dim=-1, keepdim=True)
        add_75 = mean_25 + 1e-06
        mean_25 = None
        x_n_25 = x_g_25 / add_75
        x_g_25 = add_75 = None
        view_50 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_51 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_25 = x_289 * x_n_25
        x_n_25 = None
        addcmul_25 = torch.addcmul(view_50, view_51, mul_25)
        view_50 = view_51 = mul_25 = None
        x_290 = x_289 + addcmul_25
        x_289 = addcmul_25 = None
        x_291 = torch._C._nn.linear(
            x_290,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_290 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_292 = torch.nn.functional.dropout(x_291, 0.0, False, False)
        x_291 = None
        x_293 = x_292.permute(0, 3, 1, 2)
        x_292 = None
        x_294 = x_293 + x_283
        x_293 = x_283 = None
        x_295 = torch.conv2d(
            x_294,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_conv_dw_parameters_bias_ = (None)
        x_296 = x_295.permute(0, 2, 3, 1)
        x_295 = None
        x_297 = torch.nn.functional.layer_norm(
            x_296,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_bias_,
            1e-06,
        )
        x_296 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_bias_ = (None)
        x_298 = torch._C._nn.linear(
            x_297,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_297 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_299 = torch._C._nn.gelu(x_298)
        x_298 = None
        x_300 = torch.nn.functional.dropout(x_299, 0.0, False, False)
        x_299 = None
        x_g_26 = x_300.norm(p=2, dim=(1, 2), keepdim=True)
        mean_26 = x_g_26.mean(dim=-1, keepdim=True)
        add_78 = mean_26 + 1e-06
        mean_26 = None
        x_n_26 = x_g_26 / add_78
        x_g_26 = add_78 = None
        view_52 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_53 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_26 = x_300 * x_n_26
        x_n_26 = None
        addcmul_26 = torch.addcmul(view_52, view_53, mul_26)
        view_52 = view_53 = mul_26 = None
        x_301 = x_300 + addcmul_26
        x_300 = addcmul_26 = None
        x_302 = torch._C._nn.linear(
            x_301,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_301 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_303 = torch.nn.functional.dropout(x_302, 0.0, False, False)
        x_302 = None
        x_304 = x_303.permute(0, 3, 1, 2)
        x_303 = None
        x_305 = x_304 + x_294
        x_304 = x_294 = None
        x_306 = torch.conv2d(
            x_305,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_conv_dw_parameters_bias_ = (None)
        x_307 = x_306.permute(0, 2, 3, 1)
        x_306 = None
        x_308 = torch.nn.functional.layer_norm(
            x_307,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_bias_,
            1e-06,
        )
        x_307 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_bias_ = (None)
        x_309 = torch._C._nn.linear(
            x_308,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_308 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_310 = torch._C._nn.gelu(x_309)
        x_309 = None
        x_311 = torch.nn.functional.dropout(x_310, 0.0, False, False)
        x_310 = None
        x_g_27 = x_311.norm(p=2, dim=(1, 2), keepdim=True)
        mean_27 = x_g_27.mean(dim=-1, keepdim=True)
        add_81 = mean_27 + 1e-06
        mean_27 = None
        x_n_27 = x_g_27 / add_81
        x_g_27 = add_81 = None
        view_54 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_55 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_27 = x_311 * x_n_27
        x_n_27 = None
        addcmul_27 = torch.addcmul(view_54, view_55, mul_27)
        view_54 = view_55 = mul_27 = None
        x_312 = x_311 + addcmul_27
        x_311 = addcmul_27 = None
        x_313 = torch._C._nn.linear(
            x_312,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_312 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_314 = torch.nn.functional.dropout(x_313, 0.0, False, False)
        x_313 = None
        x_315 = x_314.permute(0, 3, 1, 2)
        x_314 = None
        x_316 = x_315 + x_305
        x_315 = x_305 = None
        x_317 = torch.conv2d(
            x_316,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_conv_dw_parameters_bias_ = (None)
        x_318 = x_317.permute(0, 2, 3, 1)
        x_317 = None
        x_319 = torch.nn.functional.layer_norm(
            x_318,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_bias_,
            1e-06,
        )
        x_318 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_bias_ = (None)
        x_320 = torch._C._nn.linear(
            x_319,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_319 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_321 = torch._C._nn.gelu(x_320)
        x_320 = None
        x_322 = torch.nn.functional.dropout(x_321, 0.0, False, False)
        x_321 = None
        x_g_28 = x_322.norm(p=2, dim=(1, 2), keepdim=True)
        mean_28 = x_g_28.mean(dim=-1, keepdim=True)
        add_84 = mean_28 + 1e-06
        mean_28 = None
        x_n_28 = x_g_28 / add_84
        x_g_28 = add_84 = None
        view_56 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_57 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_28 = x_322 * x_n_28
        x_n_28 = None
        addcmul_28 = torch.addcmul(view_56, view_57, mul_28)
        view_56 = view_57 = mul_28 = None
        x_323 = x_322 + addcmul_28
        x_322 = addcmul_28 = None
        x_324 = torch._C._nn.linear(
            x_323,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_323 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_325 = torch.nn.functional.dropout(x_324, 0.0, False, False)
        x_324 = None
        x_326 = x_325.permute(0, 3, 1, 2)
        x_325 = None
        x_327 = x_326 + x_316
        x_326 = x_316 = None
        x_328 = torch.conv2d(
            x_327,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_conv_dw_parameters_bias_ = (None)
        x_329 = x_328.permute(0, 2, 3, 1)
        x_328 = None
        x_330 = torch.nn.functional.layer_norm(
            x_329,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_bias_,
            1e-06,
        )
        x_329 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_bias_ = (None)
        x_331 = torch._C._nn.linear(
            x_330,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_330 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_332 = torch._C._nn.gelu(x_331)
        x_331 = None
        x_333 = torch.nn.functional.dropout(x_332, 0.0, False, False)
        x_332 = None
        x_g_29 = x_333.norm(p=2, dim=(1, 2), keepdim=True)
        mean_29 = x_g_29.mean(dim=-1, keepdim=True)
        add_87 = mean_29 + 1e-06
        mean_29 = None
        x_n_29 = x_g_29 / add_87
        x_g_29 = add_87 = None
        view_58 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_59 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_29 = x_333 * x_n_29
        x_n_29 = None
        addcmul_29 = torch.addcmul(view_58, view_59, mul_29)
        view_58 = view_59 = mul_29 = None
        x_334 = x_333 + addcmul_29
        x_333 = addcmul_29 = None
        x_335 = torch._C._nn.linear(
            x_334,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_334 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_336 = torch.nn.functional.dropout(x_335, 0.0, False, False)
        x_335 = None
        x_337 = x_336.permute(0, 3, 1, 2)
        x_336 = None
        x_338 = x_337 + x_327
        x_337 = x_327 = None
        x_339 = torch.conv2d(
            x_338,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_conv_dw_parameters_bias_ = (None)
        x_340 = x_339.permute(0, 2, 3, 1)
        x_339 = None
        x_341 = torch.nn.functional.layer_norm(
            x_340,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_bias_,
            1e-06,
        )
        x_340 = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_bias_ = (None)
        x_342 = torch._C._nn.linear(
            x_341,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_341 = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_343 = torch._C._nn.gelu(x_342)
        x_342 = None
        x_344 = torch.nn.functional.dropout(x_343, 0.0, False, False)
        x_343 = None
        x_g_30 = x_344.norm(p=2, dim=(1, 2), keepdim=True)
        mean_30 = x_g_30.mean(dim=-1, keepdim=True)
        add_90 = mean_30 + 1e-06
        mean_30 = None
        x_n_30 = x_g_30 / add_90
        x_g_30 = add_90 = None
        view_60 = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_61 = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_30 = x_344 * x_n_30
        x_n_30 = None
        addcmul_30 = torch.addcmul(view_60, view_61, mul_30)
        view_60 = view_61 = mul_30 = None
        x_345 = x_344 + addcmul_30
        x_344 = addcmul_30 = None
        x_346 = torch._C._nn.linear(
            x_345,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_345 = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_347 = torch.nn.functional.dropout(x_346, 0.0, False, False)
        x_346 = None
        x_348 = x_347.permute(0, 3, 1, 2)
        x_347 = None
        x_349 = x_348 + x_338
        x_348 = x_338 = None
        x_350 = torch.conv2d(
            x_349,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_conv_dw_parameters_bias_ = (None)
        x_351 = x_350.permute(0, 2, 3, 1)
        x_350 = None
        x_352 = torch.nn.functional.layer_norm(
            x_351,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_bias_,
            1e-06,
        )
        x_351 = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_bias_ = (None)
        x_353 = torch._C._nn.linear(
            x_352,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_352 = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_354 = torch._C._nn.gelu(x_353)
        x_353 = None
        x_355 = torch.nn.functional.dropout(x_354, 0.0, False, False)
        x_354 = None
        x_g_31 = x_355.norm(p=2, dim=(1, 2), keepdim=True)
        mean_31 = x_g_31.mean(dim=-1, keepdim=True)
        add_93 = mean_31 + 1e-06
        mean_31 = None
        x_n_31 = x_g_31 / add_93
        x_g_31 = add_93 = None
        view_62 = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_63 = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_31 = x_355 * x_n_31
        x_n_31 = None
        addcmul_31 = torch.addcmul(view_62, view_63, mul_31)
        view_62 = view_63 = mul_31 = None
        x_356 = x_355 + addcmul_31
        x_355 = addcmul_31 = None
        x_357 = torch._C._nn.linear(
            x_356,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_356 = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_358 = torch.nn.functional.dropout(x_357, 0.0, False, False)
        x_357 = None
        x_359 = x_358.permute(0, 3, 1, 2)
        x_358 = None
        x_360 = x_359 + x_349
        x_359 = x_349 = None
        x_361 = torch.conv2d(
            x_360,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_conv_dw_parameters_bias_ = (None)
        x_362 = x_361.permute(0, 2, 3, 1)
        x_361 = None
        x_363 = torch.nn.functional.layer_norm(
            x_362,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_bias_,
            1e-06,
        )
        x_362 = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_bias_ = (None)
        x_364 = torch._C._nn.linear(
            x_363,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_363 = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_365 = torch._C._nn.gelu(x_364)
        x_364 = None
        x_366 = torch.nn.functional.dropout(x_365, 0.0, False, False)
        x_365 = None
        x_g_32 = x_366.norm(p=2, dim=(1, 2), keepdim=True)
        mean_32 = x_g_32.mean(dim=-1, keepdim=True)
        add_96 = mean_32 + 1e-06
        mean_32 = None
        x_n_32 = x_g_32 / add_96
        x_g_32 = add_96 = None
        view_64 = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_65 = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_32 = x_366 * x_n_32
        x_n_32 = None
        addcmul_32 = torch.addcmul(view_64, view_65, mul_32)
        view_64 = view_65 = mul_32 = None
        x_367 = x_366 + addcmul_32
        x_366 = addcmul_32 = None
        x_368 = torch._C._nn.linear(
            x_367,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_367 = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_369 = torch.nn.functional.dropout(x_368, 0.0, False, False)
        x_368 = None
        x_370 = x_369.permute(0, 3, 1, 2)
        x_369 = None
        x_371 = x_370 + x_360
        x_370 = x_360 = None
        x_372 = x_371.permute(0, 2, 3, 1)
        x_371 = None
        x_373 = torch.nn.functional.layer_norm(
            x_372,
            (512,),
            l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_bias_,
            1e-06,
        )
        x_372 = l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_bias_ = (None)
        x_374 = x_373.permute(0, 3, 1, 2)
        x_373 = None
        input_4 = torch.conv2d(
            x_374,
            l_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_374 = l_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_bias_ = (None)
        x_375 = torch.conv2d(
            input_4,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_bias_ = (None)
        x_376 = x_375.permute(0, 2, 3, 1)
        x_375 = None
        x_377 = torch.nn.functional.layer_norm(
            x_376,
            (1024,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_,
            1e-06,
        )
        x_376 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_ = (None)
        x_378 = torch._C._nn.linear(
            x_377,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_377 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_379 = torch._C._nn.gelu(x_378)
        x_378 = None
        x_380 = torch.nn.functional.dropout(x_379, 0.0, False, False)
        x_379 = None
        x_g_33 = x_380.norm(p=2, dim=(1, 2), keepdim=True)
        mean_33 = x_g_33.mean(dim=-1, keepdim=True)
        add_99 = mean_33 + 1e-06
        mean_33 = None
        x_n_33 = x_g_33 / add_99
        x_g_33 = add_99 = None
        view_66 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_67 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_33 = x_380 * x_n_33
        x_n_33 = None
        addcmul_33 = torch.addcmul(view_66, view_67, mul_33)
        view_66 = view_67 = mul_33 = None
        x_381 = x_380 + addcmul_33
        x_380 = addcmul_33 = None
        x_382 = torch._C._nn.linear(
            x_381,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_381 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_383 = torch.nn.functional.dropout(x_382, 0.0, False, False)
        x_382 = None
        x_384 = x_383.permute(0, 3, 1, 2)
        x_383 = None
        x_385 = x_384 + input_4
        x_384 = input_4 = None
        x_386 = torch.conv2d(
            x_385,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_bias_ = (None)
        x_387 = x_386.permute(0, 2, 3, 1)
        x_386 = None
        x_388 = torch.nn.functional.layer_norm(
            x_387,
            (1024,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_,
            1e-06,
        )
        x_387 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_ = (None)
        x_389 = torch._C._nn.linear(
            x_388,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_388 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_390 = torch._C._nn.gelu(x_389)
        x_389 = None
        x_391 = torch.nn.functional.dropout(x_390, 0.0, False, False)
        x_390 = None
        x_g_34 = x_391.norm(p=2, dim=(1, 2), keepdim=True)
        mean_34 = x_g_34.mean(dim=-1, keepdim=True)
        add_102 = mean_34 + 1e-06
        mean_34 = None
        x_n_34 = x_g_34 / add_102
        x_g_34 = add_102 = None
        view_68 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_69 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_34 = x_391 * x_n_34
        x_n_34 = None
        addcmul_34 = torch.addcmul(view_68, view_69, mul_34)
        view_68 = view_69 = mul_34 = None
        x_392 = x_391 + addcmul_34
        x_391 = addcmul_34 = None
        x_393 = torch._C._nn.linear(
            x_392,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_392 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_394 = torch.nn.functional.dropout(x_393, 0.0, False, False)
        x_393 = None
        x_395 = x_394.permute(0, 3, 1, 2)
        x_394 = None
        x_396 = x_395 + x_385
        x_395 = x_385 = None
        x_397 = torch.conv2d(
            x_396,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_dw_parameters_bias_ = (None)
        x_398 = x_397.permute(0, 2, 3, 1)
        x_397 = None
        x_399 = torch.nn.functional.layer_norm(
            x_398,
            (1024,),
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_bias_,
            1e-06,
        )
        x_398 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_bias_ = (None)
        x_400 = torch._C._nn.linear(
            x_399,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_399 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_401 = torch._C._nn.gelu(x_400)
        x_400 = None
        x_402 = torch.nn.functional.dropout(x_401, 0.0, False, False)
        x_401 = None
        x_g_35 = x_402.norm(p=2, dim=(1, 2), keepdim=True)
        mean_35 = x_g_35.mean(dim=-1, keepdim=True)
        add_105 = mean_35 + 1e-06
        mean_35 = None
        x_n_35 = x_g_35 / add_105
        x_g_35 = add_105 = None
        view_70 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_71 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_35 = x_402 * x_n_35
        x_n_35 = None
        addcmul_35 = torch.addcmul(view_70, view_71, mul_35)
        view_70 = view_71 = mul_35 = None
        x_403 = x_402 + addcmul_35
        x_402 = addcmul_35 = None
        x_404 = torch._C._nn.linear(
            x_403,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_403 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_405 = torch.nn.functional.dropout(x_404, 0.0, False, False)
        x_404 = None
        x_406 = x_405.permute(0, 3, 1, 2)
        x_405 = None
        x_407 = x_406 + x_396
        x_406 = x_396 = None
        x_408 = torch.nn.functional.adaptive_avg_pool2d(x_407, 1)
        x_407 = None
        x_409 = x_408.permute(0, 2, 3, 1)
        x_408 = None
        x_410 = torch.nn.functional.layer_norm(
            x_409,
            (1024,),
            l_self_modules_head_modules_norm_parameters_weight_,
            l_self_modules_head_modules_norm_parameters_bias_,
            1e-06,
        )
        x_409 = (
            l_self_modules_head_modules_norm_parameters_weight_
        ) = l_self_modules_head_modules_norm_parameters_bias_ = None
        x_411 = x_410.permute(0, 3, 1, 2)
        x_410 = None
        x_412 = x_411.flatten(1, -1)
        x_411 = None
        x_413 = torch.nn.functional.dropout(x_412, 0.0, False, False)
        x_412 = None
        x_414 = torch._C._nn.linear(
            x_413,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_413 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_414,)
