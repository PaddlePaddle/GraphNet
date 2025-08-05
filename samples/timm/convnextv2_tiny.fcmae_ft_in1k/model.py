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
            (96,),
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
            96,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_bias_ = (None)
        x_4 = x_3.permute(0, 2, 3, 1)
        x_3 = None
        x_5 = torch.nn.functional.layer_norm(
            x_4,
            (96,),
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
            96,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_bias_ = (None)
        x_15 = x_14.permute(0, 2, 3, 1)
        x_14 = None
        x_16 = torch.nn.functional.layer_norm(
            x_15,
            (96,),
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
            96,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_dw_parameters_bias_ = (None)
        x_26 = x_25.permute(0, 2, 3, 1)
        x_25 = None
        x_27 = torch.nn.functional.layer_norm(
            x_26,
            (96,),
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
            (96,),
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
            192,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_bias_ = (None)
        x_40 = x_39.permute(0, 2, 3, 1)
        x_39 = None
        x_41 = torch.nn.functional.layer_norm(
            x_40,
            (192,),
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
            192,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_bias_ = (None)
        x_51 = x_50.permute(0, 2, 3, 1)
        x_50 = None
        x_52 = torch.nn.functional.layer_norm(
            x_51,
            (192,),
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
            192,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_dw_parameters_bias_ = (None)
        x_62 = x_61.permute(0, 2, 3, 1)
        x_61 = None
        x_63 = torch.nn.functional.layer_norm(
            x_62,
            (192,),
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
            (192,),
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
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_bias_ = (None)
        x_76 = x_75.permute(0, 2, 3, 1)
        x_75 = None
        x_77 = torch.nn.functional.layer_norm(
            x_76,
            (384,),
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
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_bias_ = (None)
        x_87 = x_86.permute(0, 2, 3, 1)
        x_86 = None
        x_88 = torch.nn.functional.layer_norm(
            x_87,
            (384,),
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
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_bias_ = (None)
        x_98 = x_97.permute(0, 2, 3, 1)
        x_97 = None
        x_99 = torch.nn.functional.layer_norm(
            x_98,
            (384,),
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
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_bias_ = (None)
        x_109 = x_108.permute(0, 2, 3, 1)
        x_108 = None
        x_110 = torch.nn.functional.layer_norm(
            x_109,
            (384,),
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
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_dw_parameters_bias_ = (None)
        x_120 = x_119.permute(0, 2, 3, 1)
        x_119 = None
        x_121 = torch.nn.functional.layer_norm(
            x_120,
            (384,),
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
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_dw_parameters_bias_ = (None)
        x_131 = x_130.permute(0, 2, 3, 1)
        x_130 = None
        x_132 = torch.nn.functional.layer_norm(
            x_131,
            (384,),
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
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_dw_parameters_bias_ = (None)
        x_142 = x_141.permute(0, 2, 3, 1)
        x_141 = None
        x_143 = torch.nn.functional.layer_norm(
            x_142,
            (384,),
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
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_dw_parameters_bias_ = (None)
        x_153 = x_152.permute(0, 2, 3, 1)
        x_152 = None
        x_154 = torch.nn.functional.layer_norm(
            x_153,
            (384,),
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
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_dw_parameters_bias_ = (None)
        x_164 = x_163.permute(0, 2, 3, 1)
        x_163 = None
        x_165 = torch.nn.functional.layer_norm(
            x_164,
            (384,),
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
        x_174 = x_173.permute(0, 2, 3, 1)
        x_173 = None
        x_175 = torch.nn.functional.layer_norm(
            x_174,
            (384,),
            l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_bias_,
            1e-06,
        )
        x_174 = l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_bias_ = (None)
        x_176 = x_175.permute(0, 3, 1, 2)
        x_175 = None
        input_4 = torch.conv2d(
            x_176,
            l_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_176 = l_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_bias_ = (None)
        x_177 = torch.conv2d(
            input_4,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_bias_ = (None)
        x_178 = x_177.permute(0, 2, 3, 1)
        x_177 = None
        x_179 = torch.nn.functional.layer_norm(
            x_178,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_,
            1e-06,
        )
        x_178 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_ = (None)
        x_180 = torch._C._nn.linear(
            x_179,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_179 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_181 = torch._C._nn.gelu(x_180)
        x_180 = None
        x_182 = torch.nn.functional.dropout(x_181, 0.0, False, False)
        x_181 = None
        x_g_15 = x_182.norm(p=2, dim=(1, 2), keepdim=True)
        mean_15 = x_g_15.mean(dim=-1, keepdim=True)
        add_45 = mean_15 + 1e-06
        mean_15 = None
        x_n_15 = x_g_15 / add_45
        x_g_15 = add_45 = None
        view_30 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_31 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_15 = x_182 * x_n_15
        x_n_15 = None
        addcmul_15 = torch.addcmul(view_30, view_31, mul_15)
        view_30 = view_31 = mul_15 = None
        x_183 = x_182 + addcmul_15
        x_182 = addcmul_15 = None
        x_184 = torch._C._nn.linear(
            x_183,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_183 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_185 = torch.nn.functional.dropout(x_184, 0.0, False, False)
        x_184 = None
        x_186 = x_185.permute(0, 3, 1, 2)
        x_185 = None
        x_187 = x_186 + input_4
        x_186 = input_4 = None
        x_188 = torch.conv2d(
            x_187,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_bias_ = (None)
        x_189 = x_188.permute(0, 2, 3, 1)
        x_188 = None
        x_190 = torch.nn.functional.layer_norm(
            x_189,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_,
            1e-06,
        )
        x_189 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_ = (None)
        x_191 = torch._C._nn.linear(
            x_190,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_190 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_192 = torch._C._nn.gelu(x_191)
        x_191 = None
        x_193 = torch.nn.functional.dropout(x_192, 0.0, False, False)
        x_192 = None
        x_g_16 = x_193.norm(p=2, dim=(1, 2), keepdim=True)
        mean_16 = x_g_16.mean(dim=-1, keepdim=True)
        add_48 = mean_16 + 1e-06
        mean_16 = None
        x_n_16 = x_g_16 / add_48
        x_g_16 = add_48 = None
        view_32 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_33 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_16 = x_193 * x_n_16
        x_n_16 = None
        addcmul_16 = torch.addcmul(view_32, view_33, mul_16)
        view_32 = view_33 = mul_16 = None
        x_194 = x_193 + addcmul_16
        x_193 = addcmul_16 = None
        x_195 = torch._C._nn.linear(
            x_194,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_194 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_196 = torch.nn.functional.dropout(x_195, 0.0, False, False)
        x_195 = None
        x_197 = x_196.permute(0, 3, 1, 2)
        x_196 = None
        x_198 = x_197 + x_187
        x_197 = x_187 = None
        x_199 = torch.conv2d(
            x_198,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_dw_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_dw_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_dw_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_dw_parameters_bias_ = (None)
        x_200 = x_199.permute(0, 2, 3, 1)
        x_199 = None
        x_201 = torch.nn.functional.layer_norm(
            x_200,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_bias_,
            1e-06,
        )
        x_200 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_bias_ = (None)
        x_202 = torch._C._nn.linear(
            x_201,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_201 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_203 = torch._C._nn.gelu(x_202)
        x_202 = None
        x_204 = torch.nn.functional.dropout(x_203, 0.0, False, False)
        x_203 = None
        x_g_17 = x_204.norm(p=2, dim=(1, 2), keepdim=True)
        mean_17 = x_g_17.mean(dim=-1, keepdim=True)
        add_51 = mean_17 + 1e-06
        mean_17 = None
        x_n_17 = x_g_17 / add_51
        x_g_17 = add_51 = None
        view_34 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_35 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_17 = x_204 * x_n_17
        x_n_17 = None
        addcmul_17 = torch.addcmul(view_34, view_35, mul_17)
        view_34 = view_35 = mul_17 = None
        x_205 = x_204 + addcmul_17
        x_204 = addcmul_17 = None
        x_206 = torch._C._nn.linear(
            x_205,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_205 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_207 = torch.nn.functional.dropout(x_206, 0.0, False, False)
        x_206 = None
        x_208 = x_207.permute(0, 3, 1, 2)
        x_207 = None
        x_209 = x_208 + x_198
        x_208 = x_198 = None
        x_210 = torch.nn.functional.adaptive_avg_pool2d(x_209, 1)
        x_209 = None
        x_211 = x_210.permute(0, 2, 3, 1)
        x_210 = None
        x_212 = torch.nn.functional.layer_norm(
            x_211,
            (768,),
            l_self_modules_head_modules_norm_parameters_weight_,
            l_self_modules_head_modules_norm_parameters_bias_,
            1e-06,
        )
        x_211 = (
            l_self_modules_head_modules_norm_parameters_weight_
        ) = l_self_modules_head_modules_norm_parameters_bias_ = None
        x_213 = x_212.permute(0, 3, 1, 2)
        x_212 = None
        x_214 = x_213.flatten(1, -1)
        x_213 = None
        x_215 = torch.nn.functional.dropout(x_214, 0.0, False, False)
        x_214 = None
        x_216 = torch._C._nn.linear(
            x_215,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_215 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_216,)
