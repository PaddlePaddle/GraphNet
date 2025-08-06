import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_conv_parameters_weight_
        )
        l_self_modules_stem_modules_conv_parameters_bias_ = (
            L_self_modules_stem_modules_conv_parameters_bias_
        )
        l_x_ = L_x_
        l_self_modules_stem_modules_norm_parameters_weight_ = (
            L_self_modules_stem_modules_norm_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_ = L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_ = L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_
        l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_ = L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
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
        x = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_conv_parameters_weight_,
            l_self_modules_stem_modules_conv_parameters_bias_,
            (4, 4),
            (2, 2),
            (1, 1),
            1,
        )
        l_x_ = (
            l_self_modules_stem_modules_conv_parameters_weight_
        ) = l_self_modules_stem_modules_conv_parameters_bias_ = None
        x_1 = x.permute(0, 2, 3, 1)
        x = None
        x_2 = torch.nn.functional.layer_norm(
            x_1, (96,), l_self_modules_stem_modules_norm_parameters_weight_, None, 1e-06
        )
        x_1 = l_self_modules_stem_modules_norm_parameters_weight_ = None
        x_3 = x_2.permute(0, 3, 1, 2)
        x_2 = None
        sym_sum = torch.sym_sum([-3, s1])
        s1 = None
        floordiv = sym_sum // 4
        sym_sum_1 = torch.sym_sum([1, floordiv])
        floordiv = sym_sum_1 = None
        group_norm = torch.nn.functional.group_norm(
            x_3,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (
            None
        )
        y = torch._C._nn.avg_pool2d(group_norm, 3, 1, 1, False, False, None)
        sub = y - group_norm
        y = group_norm = None
        x_4 = x_3 + sub
        x_3 = sub = None
        group_norm_1 = torch.nn.functional.group_norm(
            x_4,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (
            None
        )
        x_5 = torch.conv2d(
            group_norm_1,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_1 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu = torch.nn.functional.relu(x_5, inplace=False)
        x_5 = None
        pow_1 = relu**2
        relu = None
        mul = (
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
            * pow_1
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = (
            pow_1
        ) = None
        x_6 = (
            mul
            + l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        )
        mul = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = (None)
        x_7 = torch.nn.functional.dropout(x_6, 0.0, False, False)
        x_6 = None
        x_8 = torch.conv2d(
            x_7,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_7 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_9 = torch.nn.functional.dropout(x_8, 0.0, False, False)
        x_8 = None
        x_10 = x_4 + x_9
        x_4 = x_9 = None
        group_norm_2 = torch.nn.functional.group_norm(
            x_10,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (
            None
        )
        y_1 = torch._C._nn.avg_pool2d(group_norm_2, 3, 1, 1, False, False, None)
        sub_1 = y_1 - group_norm_2
        y_1 = group_norm_2 = None
        x_11 = x_10 + sub_1
        x_10 = sub_1 = None
        group_norm_3 = torch.nn.functional.group_norm(
            x_11,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (
            None
        )
        x_12 = torch.conv2d(
            group_norm_3,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_3 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_1 = torch.nn.functional.relu(x_12, inplace=False)
        x_12 = None
        pow_2 = relu_1**2
        relu_1 = None
        mul_1 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
            * pow_2
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = (
            pow_2
        ) = None
        x_13 = (
            mul_1
            + l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        )
        mul_1 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = (None)
        x_14 = torch.nn.functional.dropout(x_13, 0.0, False, False)
        x_13 = None
        x_15 = torch.conv2d(
            x_14,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_14 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_16 = torch.nn.functional.dropout(x_15, 0.0, False, False)
        x_15 = None
        x_17 = x_11 + x_16
        x_11 = x_16 = None
        group_norm_4 = torch.nn.functional.group_norm(
            x_17,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (
            None
        )
        y_2 = torch._C._nn.avg_pool2d(group_norm_4, 3, 1, 1, False, False, None)
        sub_2 = y_2 - group_norm_4
        y_2 = group_norm_4 = None
        x_18 = x_17 + sub_2
        x_17 = sub_2 = None
        group_norm_5 = torch.nn.functional.group_norm(
            x_18,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (
            None
        )
        x_19 = torch.conv2d(
            group_norm_5,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_5 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_2 = torch.nn.functional.relu(x_19, inplace=False)
        x_19 = None
        pow_3 = relu_2**2
        relu_2 = None
        mul_2 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_
            * pow_3
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_ = (
            pow_3
        ) = None
        x_20 = (
            mul_2
            + l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_
        )
        mul_2 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_ = (None)
        x_21 = torch.nn.functional.dropout(x_20, 0.0, False, False)
        x_20 = None
        x_22 = torch.conv2d(
            x_21,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_21 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_23 = torch.nn.functional.dropout(x_22, 0.0, False, False)
        x_22 = None
        x_24 = x_18 + x_23
        x_18 = x_23 = None
        group_norm_6 = torch.nn.functional.group_norm(
            x_24,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm1_parameters_weight_ = (
            None
        )
        y_3 = torch._C._nn.avg_pool2d(group_norm_6, 3, 1, 1, False, False, None)
        sub_3 = y_3 - group_norm_6
        y_3 = group_norm_6 = None
        x_25 = x_24 + sub_3
        x_24 = sub_3 = None
        group_norm_7 = torch.nn.functional.group_norm(
            x_25,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm2_parameters_weight_ = (
            None
        )
        x_26 = torch.conv2d(
            group_norm_7,
            l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_7 = l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_3 = torch.nn.functional.relu(x_26, inplace=False)
        x_26 = None
        pow_4 = relu_3**2
        relu_3 = None
        mul_3 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_
            * pow_4
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_ = (
            pow_4
        ) = None
        x_27 = (
            mul_3
            + l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_
        )
        mul_3 = l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_ = (None)
        x_28 = torch.nn.functional.dropout(x_27, 0.0, False, False)
        x_27 = None
        x_29 = torch.conv2d(
            x_28,
            l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_28 = l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_30 = torch.nn.functional.dropout(x_29, 0.0, False, False)
        x_29 = None
        x_31 = x_25 + x_30
        x_25 = x_30 = None
        group_norm_8 = torch.nn.functional.group_norm(
            x_31,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm1_parameters_weight_ = (
            None
        )
        y_4 = torch._C._nn.avg_pool2d(group_norm_8, 3, 1, 1, False, False, None)
        sub_4 = y_4 - group_norm_8
        y_4 = group_norm_8 = None
        x_32 = x_31 + sub_4
        x_31 = sub_4 = None
        group_norm_9 = torch.nn.functional.group_norm(
            x_32,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm2_parameters_weight_ = (
            None
        )
        x_33 = torch.conv2d(
            group_norm_9,
            l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_9 = l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_4 = torch.nn.functional.relu(x_33, inplace=False)
        x_33 = None
        pow_5 = relu_4**2
        relu_4 = None
        mul_4 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_
            * pow_5
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_ = (
            pow_5
        ) = None
        x_34 = (
            mul_4
            + l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_
        )
        mul_4 = l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_ = (None)
        x_35 = torch.nn.functional.dropout(x_34, 0.0, False, False)
        x_34 = None
        x_36 = torch.conv2d(
            x_35,
            l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_35 = l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_37 = torch.nn.functional.dropout(x_36, 0.0, False, False)
        x_36 = None
        x_38 = x_32 + x_37
        x_32 = x_37 = None
        group_norm_10 = torch.nn.functional.group_norm(
            x_38,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm1_parameters_weight_ = (
            None
        )
        y_5 = torch._C._nn.avg_pool2d(group_norm_10, 3, 1, 1, False, False, None)
        sub_5 = y_5 - group_norm_10
        y_5 = group_norm_10 = None
        x_39 = x_38 + sub_5
        x_38 = sub_5 = None
        group_norm_11 = torch.nn.functional.group_norm(
            x_39,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm2_parameters_weight_ = (
            None
        )
        x_40 = torch.conv2d(
            group_norm_11,
            l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_11 = l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_5 = torch.nn.functional.relu(x_40, inplace=False)
        x_40 = None
        pow_6 = relu_5**2
        relu_5 = None
        mul_5 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_
            * pow_6
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_ = (
            pow_6
        ) = None
        x_41 = (
            mul_5
            + l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_
        )
        mul_5 = l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_ = (None)
        x_42 = torch.nn.functional.dropout(x_41, 0.0, False, False)
        x_41 = None
        x_43 = torch.conv2d(
            x_42,
            l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_42 = l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_44 = torch.nn.functional.dropout(x_43, 0.0, False, False)
        x_43 = None
        x_45 = x_39 + x_44
        x_39 = x_44 = None
        x_46 = x_45.permute(0, 2, 3, 1)
        x_45 = None
        x_47 = torch.nn.functional.layer_norm(
            x_46,
            (96,),
            l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_,
            None,
            1e-06,
        )
        x_46 = l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_ = (None)
        x_48 = x_47.permute(0, 3, 1, 2)
        x_47 = None
        x_49 = torch.conv2d(
            x_48,
            l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_48 = l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_ = l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_ = (None)
        floordiv_1 = sym_sum // 8
        sym_sum_2 = torch.sym_sum([1, floordiv_1])
        floordiv_1 = sym_sum_2 = None
        group_norm_12 = torch.nn.functional.group_norm(
            x_49,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (
            None
        )
        y_6 = torch._C._nn.avg_pool2d(group_norm_12, 3, 1, 1, False, False, None)
        sub_6 = y_6 - group_norm_12
        y_6 = group_norm_12 = None
        x_50 = x_49 + sub_6
        x_49 = sub_6 = None
        group_norm_13 = torch.nn.functional.group_norm(
            x_50,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (
            None
        )
        x_51 = torch.conv2d(
            group_norm_13,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_13 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_6 = torch.nn.functional.relu(x_51, inplace=False)
        x_51 = None
        pow_7 = relu_6**2
        relu_6 = None
        mul_6 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
            * pow_7
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = (
            pow_7
        ) = None
        x_52 = (
            mul_6
            + l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        )
        mul_6 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = (None)
        x_53 = torch.nn.functional.dropout(x_52, 0.0, False, False)
        x_52 = None
        x_54 = torch.conv2d(
            x_53,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_53 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_55 = torch.nn.functional.dropout(x_54, 0.0, False, False)
        x_54 = None
        x_56 = x_50 + x_55
        x_50 = x_55 = None
        group_norm_14 = torch.nn.functional.group_norm(
            x_56,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (
            None
        )
        y_7 = torch._C._nn.avg_pool2d(group_norm_14, 3, 1, 1, False, False, None)
        sub_7 = y_7 - group_norm_14
        y_7 = group_norm_14 = None
        x_57 = x_56 + sub_7
        x_56 = sub_7 = None
        group_norm_15 = torch.nn.functional.group_norm(
            x_57,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (
            None
        )
        x_58 = torch.conv2d(
            group_norm_15,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_15 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_7 = torch.nn.functional.relu(x_58, inplace=False)
        x_58 = None
        pow_8 = relu_7**2
        relu_7 = None
        mul_7 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
            * pow_8
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = (
            pow_8
        ) = None
        x_59 = (
            mul_7
            + l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        )
        mul_7 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = (None)
        x_60 = torch.nn.functional.dropout(x_59, 0.0, False, False)
        x_59 = None
        x_61 = torch.conv2d(
            x_60,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_60 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_62 = torch.nn.functional.dropout(x_61, 0.0, False, False)
        x_61 = None
        x_63 = x_57 + x_62
        x_57 = x_62 = None
        group_norm_16 = torch.nn.functional.group_norm(
            x_63,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (
            None
        )
        y_8 = torch._C._nn.avg_pool2d(group_norm_16, 3, 1, 1, False, False, None)
        sub_8 = y_8 - group_norm_16
        y_8 = group_norm_16 = None
        x_64 = x_63 + sub_8
        x_63 = sub_8 = None
        group_norm_17 = torch.nn.functional.group_norm(
            x_64,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (
            None
        )
        x_65 = torch.conv2d(
            group_norm_17,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_17 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_8 = torch.nn.functional.relu(x_65, inplace=False)
        x_65 = None
        pow_9 = relu_8**2
        relu_8 = None
        mul_8 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_
            * pow_9
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_ = (
            pow_9
        ) = None
        x_66 = (
            mul_8
            + l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_
        )
        mul_8 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_ = (None)
        x_67 = torch.nn.functional.dropout(x_66, 0.0, False, False)
        x_66 = None
        x_68 = torch.conv2d(
            x_67,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_67 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_69 = torch.nn.functional.dropout(x_68, 0.0, False, False)
        x_68 = None
        x_70 = x_64 + x_69
        x_64 = x_69 = None
        group_norm_18 = torch.nn.functional.group_norm(
            x_70,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_ = (
            None
        )
        y_9 = torch._C._nn.avg_pool2d(group_norm_18, 3, 1, 1, False, False, None)
        sub_9 = y_9 - group_norm_18
        y_9 = group_norm_18 = None
        x_71 = x_70 + sub_9
        x_70 = sub_9 = None
        group_norm_19 = torch.nn.functional.group_norm(
            x_71,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_ = (
            None
        )
        x_72 = torch.conv2d(
            group_norm_19,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_19 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_9 = torch.nn.functional.relu(x_72, inplace=False)
        x_72 = None
        pow_10 = relu_9**2
        relu_9 = None
        mul_9 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_
            * pow_10
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_ = (
            pow_10
        ) = None
        x_73 = (
            mul_9
            + l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_
        )
        mul_9 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_ = (None)
        x_74 = torch.nn.functional.dropout(x_73, 0.0, False, False)
        x_73 = None
        x_75 = torch.conv2d(
            x_74,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_74 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_76 = torch.nn.functional.dropout(x_75, 0.0, False, False)
        x_75 = None
        x_77 = x_71 + x_76
        x_71 = x_76 = None
        group_norm_20 = torch.nn.functional.group_norm(
            x_77,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_weight_ = (
            None
        )
        y_10 = torch._C._nn.avg_pool2d(group_norm_20, 3, 1, 1, False, False, None)
        sub_10 = y_10 - group_norm_20
        y_10 = group_norm_20 = None
        x_78 = x_77 + sub_10
        x_77 = sub_10 = None
        group_norm_21 = torch.nn.functional.group_norm(
            x_78,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_weight_ = (
            None
        )
        x_79 = torch.conv2d(
            group_norm_21,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_21 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_10 = torch.nn.functional.relu(x_79, inplace=False)
        x_79 = None
        pow_11 = relu_10**2
        relu_10 = None
        mul_10 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_
            * pow_11
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_ = (
            pow_11
        ) = None
        x_80 = (
            mul_10
            + l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_
        )
        mul_10 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_ = (None)
        x_81 = torch.nn.functional.dropout(x_80, 0.0, False, False)
        x_80 = None
        x_82 = torch.conv2d(
            x_81,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_81 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_83 = torch.nn.functional.dropout(x_82, 0.0, False, False)
        x_82 = None
        x_84 = x_78 + x_83
        x_78 = x_83 = None
        group_norm_22 = torch.nn.functional.group_norm(
            x_84,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_weight_ = (
            None
        )
        y_11 = torch._C._nn.avg_pool2d(group_norm_22, 3, 1, 1, False, False, None)
        sub_11 = y_11 - group_norm_22
        y_11 = group_norm_22 = None
        x_85 = x_84 + sub_11
        x_84 = sub_11 = None
        group_norm_23 = torch.nn.functional.group_norm(
            x_85,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_weight_ = (
            None
        )
        x_86 = torch.conv2d(
            group_norm_23,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_23 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_11 = torch.nn.functional.relu(x_86, inplace=False)
        x_86 = None
        pow_12 = relu_11**2
        relu_11 = None
        mul_11 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_
            * pow_12
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_ = (
            pow_12
        ) = None
        x_87 = (
            mul_11
            + l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_
        )
        mul_11 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_ = (None)
        x_88 = torch.nn.functional.dropout(x_87, 0.0, False, False)
        x_87 = None
        x_89 = torch.conv2d(
            x_88,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_88 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_90 = torch.nn.functional.dropout(x_89, 0.0, False, False)
        x_89 = None
        x_91 = x_85 + x_90
        x_85 = x_90 = None
        x_92 = x_91.permute(0, 2, 3, 1)
        x_91 = None
        x_93 = torch.nn.functional.layer_norm(
            x_92,
            (192,),
            l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_,
            None,
            1e-06,
        )
        x_92 = l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_ = (None)
        x_94 = x_93.permute(0, 3, 1, 2)
        x_93 = None
        x_95 = torch.conv2d(
            x_94,
            l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_94 = l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_ = (None)
        floordiv_2 = sym_sum // 16
        sym_sum_3 = torch.sym_sum([1, floordiv_2])
        floordiv_2 = sym_sum_3 = None
        view = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_12 = x_95 * view
        view = None
        group_norm_24 = torch.nn.functional.group_norm(
            x_95,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_95 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (None)
        y_12 = torch._C._nn.avg_pool2d(group_norm_24, 3, 1, 1, False, False, None)
        sub_12 = y_12 - group_norm_24
        y_12 = group_norm_24 = None
        x_96 = mul_12 + sub_12
        mul_12 = sub_12 = None
        view_1 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_13 = x_96 * view_1
        view_1 = None
        group_norm_25 = torch.nn.functional.group_norm(
            x_96,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_96 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (None)
        x_97 = torch.conv2d(
            group_norm_25,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_25 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_12 = torch.nn.functional.relu(x_97, inplace=False)
        x_97 = None
        pow_13 = relu_12**2
        relu_12 = None
        mul_14 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
            * pow_13
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = (
            pow_13
        ) = None
        x_98 = (
            mul_14
            + l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        )
        mul_14 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = (None)
        x_99 = torch.nn.functional.dropout(x_98, 0.0, False, False)
        x_98 = None
        x_100 = torch.conv2d(
            x_99,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_99 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_101 = torch.nn.functional.dropout(x_100, 0.0, False, False)
        x_100 = None
        x_102 = mul_13 + x_101
        mul_13 = x_101 = None
        view_2 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_15 = x_102 * view_2
        view_2 = None
        group_norm_26 = torch.nn.functional.group_norm(
            x_102,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_102 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (None)
        y_13 = torch._C._nn.avg_pool2d(group_norm_26, 3, 1, 1, False, False, None)
        sub_13 = y_13 - group_norm_26
        y_13 = group_norm_26 = None
        x_103 = mul_15 + sub_13
        mul_15 = sub_13 = None
        view_3 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_16 = x_103 * view_3
        view_3 = None
        group_norm_27 = torch.nn.functional.group_norm(
            x_103,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_103 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (None)
        x_104 = torch.conv2d(
            group_norm_27,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_27 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_13 = torch.nn.functional.relu(x_104, inplace=False)
        x_104 = None
        pow_14 = relu_13**2
        relu_13 = None
        mul_17 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
            * pow_14
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = (
            pow_14
        ) = None
        x_105 = (
            mul_17
            + l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        )
        mul_17 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = (None)
        x_106 = torch.nn.functional.dropout(x_105, 0.0, False, False)
        x_105 = None
        x_107 = torch.conv2d(
            x_106,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_106 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_108 = torch.nn.functional.dropout(x_107, 0.0, False, False)
        x_107 = None
        x_109 = mul_16 + x_108
        mul_16 = x_108 = None
        view_4 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_18 = x_109 * view_4
        view_4 = None
        group_norm_28 = torch.nn.functional.group_norm(
            x_109,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_109 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (None)
        y_14 = torch._C._nn.avg_pool2d(group_norm_28, 3, 1, 1, False, False, None)
        sub_14 = y_14 - group_norm_28
        y_14 = group_norm_28 = None
        x_110 = mul_18 + sub_14
        mul_18 = sub_14 = None
        view_5 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_19 = x_110 * view_5
        view_5 = None
        group_norm_29 = torch.nn.functional.group_norm(
            x_110,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_110 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (None)
        x_111 = torch.conv2d(
            group_norm_29,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_29 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_14 = torch.nn.functional.relu(x_111, inplace=False)
        x_111 = None
        pow_15 = relu_14**2
        relu_14 = None
        mul_20 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_
            * pow_15
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_ = (
            pow_15
        ) = None
        x_112 = (
            mul_20
            + l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_
        )
        mul_20 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_ = (None)
        x_113 = torch.nn.functional.dropout(x_112, 0.0, False, False)
        x_112 = None
        x_114 = torch.conv2d(
            x_113,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_113 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_115 = torch.nn.functional.dropout(x_114, 0.0, False, False)
        x_114 = None
        x_116 = mul_19 + x_115
        mul_19 = x_115 = None
        view_6 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_21 = x_116 * view_6
        view_6 = None
        group_norm_30 = torch.nn.functional.group_norm(
            x_116,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_116 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_ = (None)
        y_15 = torch._C._nn.avg_pool2d(group_norm_30, 3, 1, 1, False, False, None)
        sub_15 = y_15 - group_norm_30
        y_15 = group_norm_30 = None
        x_117 = mul_21 + sub_15
        mul_21 = sub_15 = None
        view_7 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_22 = x_117 * view_7
        view_7 = None
        group_norm_31 = torch.nn.functional.group_norm(
            x_117,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_117 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_ = (None)
        x_118 = torch.conv2d(
            group_norm_31,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_31 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_15 = torch.nn.functional.relu(x_118, inplace=False)
        x_118 = None
        pow_16 = relu_15**2
        relu_15 = None
        mul_23 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_
            * pow_16
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_ = (
            pow_16
        ) = None
        x_119 = (
            mul_23
            + l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_
        )
        mul_23 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_ = (None)
        x_120 = torch.nn.functional.dropout(x_119, 0.0, False, False)
        x_119 = None
        x_121 = torch.conv2d(
            x_120,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_120 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_122 = torch.nn.functional.dropout(x_121, 0.0, False, False)
        x_121 = None
        x_123 = mul_22 + x_122
        mul_22 = x_122 = None
        view_8 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_24 = x_123 * view_8
        view_8 = None
        group_norm_32 = torch.nn.functional.group_norm(
            x_123,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_123 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_ = (None)
        y_16 = torch._C._nn.avg_pool2d(group_norm_32, 3, 1, 1, False, False, None)
        sub_16 = y_16 - group_norm_32
        y_16 = group_norm_32 = None
        x_124 = mul_24 + sub_16
        mul_24 = sub_16 = None
        view_9 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_25 = x_124 * view_9
        view_9 = None
        group_norm_33 = torch.nn.functional.group_norm(
            x_124,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_124 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_ = (None)
        x_125 = torch.conv2d(
            group_norm_33,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_33 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_16 = torch.nn.functional.relu(x_125, inplace=False)
        x_125 = None
        pow_17 = relu_16**2
        relu_16 = None
        mul_26 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_
            * pow_17
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_ = (
            pow_17
        ) = None
        x_126 = (
            mul_26
            + l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_
        )
        mul_26 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_ = (None)
        x_127 = torch.nn.functional.dropout(x_126, 0.0, False, False)
        x_126 = None
        x_128 = torch.conv2d(
            x_127,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_127 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_129 = torch.nn.functional.dropout(x_128, 0.0, False, False)
        x_128 = None
        x_130 = mul_25 + x_129
        mul_25 = x_129 = None
        view_10 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_27 = x_130 * view_10
        view_10 = None
        group_norm_34 = torch.nn.functional.group_norm(
            x_130,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_130 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_ = (None)
        y_17 = torch._C._nn.avg_pool2d(group_norm_34, 3, 1, 1, False, False, None)
        sub_17 = y_17 - group_norm_34
        y_17 = group_norm_34 = None
        x_131 = mul_27 + sub_17
        mul_27 = sub_17 = None
        view_11 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_28 = x_131 * view_11
        view_11 = None
        group_norm_35 = torch.nn.functional.group_norm(
            x_131,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_131 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_ = (None)
        x_132 = torch.conv2d(
            group_norm_35,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_35 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_17 = torch.nn.functional.relu(x_132, inplace=False)
        x_132 = None
        pow_18 = relu_17**2
        relu_17 = None
        mul_29 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_
            * pow_18
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_ = (
            pow_18
        ) = None
        x_133 = (
            mul_29
            + l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_
        )
        mul_29 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_ = (None)
        x_134 = torch.nn.functional.dropout(x_133, 0.0, False, False)
        x_133 = None
        x_135 = torch.conv2d(
            x_134,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_134 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_136 = torch.nn.functional.dropout(x_135, 0.0, False, False)
        x_135 = None
        x_137 = mul_28 + x_136
        mul_28 = x_136 = None
        view_12 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_30 = x_137 * view_12
        view_12 = None
        group_norm_36 = torch.nn.functional.group_norm(
            x_137,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_137 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_ = (None)
        y_18 = torch._C._nn.avg_pool2d(group_norm_36, 3, 1, 1, False, False, None)
        sub_18 = y_18 - group_norm_36
        y_18 = group_norm_36 = None
        x_138 = mul_30 + sub_18
        mul_30 = sub_18 = None
        view_13 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_31 = x_138 * view_13
        view_13 = None
        group_norm_37 = torch.nn.functional.group_norm(
            x_138,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_138 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_ = (None)
        x_139 = torch.conv2d(
            group_norm_37,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_37 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_18 = torch.nn.functional.relu(x_139, inplace=False)
        x_139 = None
        pow_19 = relu_18**2
        relu_18 = None
        mul_32 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_scale_
            * pow_19
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_scale_ = (
            pow_19
        ) = None
        x_140 = (
            mul_32
            + l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_bias_
        )
        mul_32 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_bias_ = (None)
        x_141 = torch.nn.functional.dropout(x_140, 0.0, False, False)
        x_140 = None
        x_142 = torch.conv2d(
            x_141,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_141 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_143 = torch.nn.functional.dropout(x_142, 0.0, False, False)
        x_142 = None
        x_144 = mul_31 + x_143
        mul_31 = x_143 = None
        view_14 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_33 = x_144 * view_14
        view_14 = None
        group_norm_38 = torch.nn.functional.group_norm(
            x_144,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_144 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_ = (None)
        y_19 = torch._C._nn.avg_pool2d(group_norm_38, 3, 1, 1, False, False, None)
        sub_19 = y_19 - group_norm_38
        y_19 = group_norm_38 = None
        x_145 = mul_33 + sub_19
        mul_33 = sub_19 = None
        view_15 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_34 = x_145 * view_15
        view_15 = None
        group_norm_39 = torch.nn.functional.group_norm(
            x_145,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_145 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_ = (None)
        x_146 = torch.conv2d(
            group_norm_39,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_39 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_19 = torch.nn.functional.relu(x_146, inplace=False)
        x_146 = None
        pow_20 = relu_19**2
        relu_19 = None
        mul_35 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_scale_
            * pow_20
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_scale_ = (
            pow_20
        ) = None
        x_147 = (
            mul_35
            + l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_bias_
        )
        mul_35 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_bias_ = (None)
        x_148 = torch.nn.functional.dropout(x_147, 0.0, False, False)
        x_147 = None
        x_149 = torch.conv2d(
            x_148,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_148 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_150 = torch.nn.functional.dropout(x_149, 0.0, False, False)
        x_149 = None
        x_151 = mul_34 + x_150
        mul_34 = x_150 = None
        view_16 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_36 = x_151 * view_16
        view_16 = None
        group_norm_40 = torch.nn.functional.group_norm(
            x_151,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_151 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_ = (None)
        y_20 = torch._C._nn.avg_pool2d(group_norm_40, 3, 1, 1, False, False, None)
        sub_20 = y_20 - group_norm_40
        y_20 = group_norm_40 = None
        x_152 = mul_36 + sub_20
        mul_36 = sub_20 = None
        view_17 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_37 = x_152 * view_17
        view_17 = None
        group_norm_41 = torch.nn.functional.group_norm(
            x_152,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_152 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_ = (None)
        x_153 = torch.conv2d(
            group_norm_41,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_41 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_20 = torch.nn.functional.relu(x_153, inplace=False)
        x_153 = None
        pow_21 = relu_20**2
        relu_20 = None
        mul_38 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_scale_
            * pow_21
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_scale_ = (
            pow_21
        ) = None
        x_154 = (
            mul_38
            + l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_bias_
        )
        mul_38 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_bias_ = (None)
        x_155 = torch.nn.functional.dropout(x_154, 0.0, False, False)
        x_154 = None
        x_156 = torch.conv2d(
            x_155,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_155 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_157 = torch.nn.functional.dropout(x_156, 0.0, False, False)
        x_156 = None
        x_158 = mul_37 + x_157
        mul_37 = x_157 = None
        view_18 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_39 = x_158 * view_18
        view_18 = None
        group_norm_42 = torch.nn.functional.group_norm(
            x_158,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_158 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_ = (None)
        y_21 = torch._C._nn.avg_pool2d(group_norm_42, 3, 1, 1, False, False, None)
        sub_21 = y_21 - group_norm_42
        y_21 = group_norm_42 = None
        x_159 = mul_39 + sub_21
        mul_39 = sub_21 = None
        view_19 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_40 = x_159 * view_19
        view_19 = None
        group_norm_43 = torch.nn.functional.group_norm(
            x_159,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_159 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_ = (None)
        x_160 = torch.conv2d(
            group_norm_43,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_43 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_21 = torch.nn.functional.relu(x_160, inplace=False)
        x_160 = None
        pow_22 = relu_21**2
        relu_21 = None
        mul_41 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_scale_
            * pow_22
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_scale_ = (
            pow_22
        ) = None
        x_161 = (
            mul_41
            + l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_bias_
        )
        mul_41 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_bias_ = (None)
        x_162 = torch.nn.functional.dropout(x_161, 0.0, False, False)
        x_161 = None
        x_163 = torch.conv2d(
            x_162,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_162 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_164 = torch.nn.functional.dropout(x_163, 0.0, False, False)
        x_163 = None
        x_165 = mul_40 + x_164
        mul_40 = x_164 = None
        view_20 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_42 = x_165 * view_20
        view_20 = None
        group_norm_44 = torch.nn.functional.group_norm(
            x_165,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_165 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_ = (None)
        y_22 = torch._C._nn.avg_pool2d(group_norm_44, 3, 1, 1, False, False, None)
        sub_22 = y_22 - group_norm_44
        y_22 = group_norm_44 = None
        x_166 = mul_42 + sub_22
        mul_42 = sub_22 = None
        view_21 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_43 = x_166 * view_21
        view_21 = None
        group_norm_45 = torch.nn.functional.group_norm(
            x_166,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_166 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_ = (None)
        x_167 = torch.conv2d(
            group_norm_45,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_45 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_22 = torch.nn.functional.relu(x_167, inplace=False)
        x_167 = None
        pow_23 = relu_22**2
        relu_22 = None
        mul_44 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_scale_
            * pow_23
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_scale_ = (
            pow_23
        ) = None
        x_168 = (
            mul_44
            + l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_bias_
        )
        mul_44 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_bias_ = (None)
        x_169 = torch.nn.functional.dropout(x_168, 0.0, False, False)
        x_168 = None
        x_170 = torch.conv2d(
            x_169,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_169 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_171 = torch.nn.functional.dropout(x_170, 0.0, False, False)
        x_170 = None
        x_172 = mul_43 + x_171
        mul_43 = x_171 = None
        view_22 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_45 = x_172 * view_22
        view_22 = None
        group_norm_46 = torch.nn.functional.group_norm(
            x_172,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_172 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_ = (None)
        y_23 = torch._C._nn.avg_pool2d(group_norm_46, 3, 1, 1, False, False, None)
        sub_23 = y_23 - group_norm_46
        y_23 = group_norm_46 = None
        x_173 = mul_45 + sub_23
        mul_45 = sub_23 = None
        view_23 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_46 = x_173 * view_23
        view_23 = None
        group_norm_47 = torch.nn.functional.group_norm(
            x_173,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_173 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_ = (None)
        x_174 = torch.conv2d(
            group_norm_47,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_47 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_23 = torch.nn.functional.relu(x_174, inplace=False)
        x_174 = None
        pow_24 = relu_23**2
        relu_23 = None
        mul_47 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_scale_
            * pow_24
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_scale_ = (
            pow_24
        ) = None
        x_175 = (
            mul_47
            + l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_bias_
        )
        mul_47 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_bias_ = (None)
        x_176 = torch.nn.functional.dropout(x_175, 0.0, False, False)
        x_175 = None
        x_177 = torch.conv2d(
            x_176,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_176 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_178 = torch.nn.functional.dropout(x_177, 0.0, False, False)
        x_177 = None
        x_179 = mul_46 + x_178
        mul_46 = x_178 = None
        view_24 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_48 = x_179 * view_24
        view_24 = None
        group_norm_48 = torch.nn.functional.group_norm(
            x_179,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_179 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_ = (None)
        y_24 = torch._C._nn.avg_pool2d(group_norm_48, 3, 1, 1, False, False, None)
        sub_24 = y_24 - group_norm_48
        y_24 = group_norm_48 = None
        x_180 = mul_48 + sub_24
        mul_48 = sub_24 = None
        view_25 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_49 = x_180 * view_25
        view_25 = None
        group_norm_49 = torch.nn.functional.group_norm(
            x_180,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_180 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_ = (None)
        x_181 = torch.conv2d(
            group_norm_49,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_49 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_24 = torch.nn.functional.relu(x_181, inplace=False)
        x_181 = None
        pow_25 = relu_24**2
        relu_24 = None
        mul_50 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_scale_
            * pow_25
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_scale_ = (
            pow_25
        ) = None
        x_182 = (
            mul_50
            + l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_bias_
        )
        mul_50 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_bias_ = (None)
        x_183 = torch.nn.functional.dropout(x_182, 0.0, False, False)
        x_182 = None
        x_184 = torch.conv2d(
            x_183,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_183 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_185 = torch.nn.functional.dropout(x_184, 0.0, False, False)
        x_184 = None
        x_186 = mul_49 + x_185
        mul_49 = x_185 = None
        view_26 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_51 = x_186 * view_26
        view_26 = None
        group_norm_50 = torch.nn.functional.group_norm(
            x_186,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_186 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_ = (None)
        y_25 = torch._C._nn.avg_pool2d(group_norm_50, 3, 1, 1, False, False, None)
        sub_25 = y_25 - group_norm_50
        y_25 = group_norm_50 = None
        x_187 = mul_51 + sub_25
        mul_51 = sub_25 = None
        view_27 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_52 = x_187 * view_27
        view_27 = None
        group_norm_51 = torch.nn.functional.group_norm(
            x_187,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_187 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_ = (None)
        x_188 = torch.conv2d(
            group_norm_51,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_51 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_25 = torch.nn.functional.relu(x_188, inplace=False)
        x_188 = None
        pow_26 = relu_25**2
        relu_25 = None
        mul_53 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_scale_
            * pow_26
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_scale_ = (
            pow_26
        ) = None
        x_189 = (
            mul_53
            + l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_bias_
        )
        mul_53 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_bias_ = (None)
        x_190 = torch.nn.functional.dropout(x_189, 0.0, False, False)
        x_189 = None
        x_191 = torch.conv2d(
            x_190,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_190 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_192 = torch.nn.functional.dropout(x_191, 0.0, False, False)
        x_191 = None
        x_193 = mul_52 + x_192
        mul_52 = x_192 = None
        view_28 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_54 = x_193 * view_28
        view_28 = None
        group_norm_52 = torch.nn.functional.group_norm(
            x_193,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_193 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_ = (None)
        y_26 = torch._C._nn.avg_pool2d(group_norm_52, 3, 1, 1, False, False, None)
        sub_26 = y_26 - group_norm_52
        y_26 = group_norm_52 = None
        x_194 = mul_54 + sub_26
        mul_54 = sub_26 = None
        view_29 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_55 = x_194 * view_29
        view_29 = None
        group_norm_53 = torch.nn.functional.group_norm(
            x_194,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_194 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_ = (None)
        x_195 = torch.conv2d(
            group_norm_53,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_53 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_26 = torch.nn.functional.relu(x_195, inplace=False)
        x_195 = None
        pow_27 = relu_26**2
        relu_26 = None
        mul_56 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_scale_
            * pow_27
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_scale_ = (
            pow_27
        ) = None
        x_196 = (
            mul_56
            + l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_bias_
        )
        mul_56 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_bias_ = (None)
        x_197 = torch.nn.functional.dropout(x_196, 0.0, False, False)
        x_196 = None
        x_198 = torch.conv2d(
            x_197,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_197 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_199 = torch.nn.functional.dropout(x_198, 0.0, False, False)
        x_198 = None
        x_200 = mul_55 + x_199
        mul_55 = x_199 = None
        view_30 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_57 = x_200 * view_30
        view_30 = None
        group_norm_54 = torch.nn.functional.group_norm(
            x_200,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_200 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_ = (None)
        y_27 = torch._C._nn.avg_pool2d(group_norm_54, 3, 1, 1, False, False, None)
        sub_27 = y_27 - group_norm_54
        y_27 = group_norm_54 = None
        x_201 = mul_57 + sub_27
        mul_57 = sub_27 = None
        view_31 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_58 = x_201 * view_31
        view_31 = None
        group_norm_55 = torch.nn.functional.group_norm(
            x_201,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_201 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_ = (None)
        x_202 = torch.conv2d(
            group_norm_55,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_55 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_27 = torch.nn.functional.relu(x_202, inplace=False)
        x_202 = None
        pow_28 = relu_27**2
        relu_27 = None
        mul_59 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_scale_
            * pow_28
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_scale_ = (
            pow_28
        ) = None
        x_203 = (
            mul_59
            + l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_bias_
        )
        mul_59 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_bias_ = (None)
        x_204 = torch.nn.functional.dropout(x_203, 0.0, False, False)
        x_203 = None
        x_205 = torch.conv2d(
            x_204,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_204 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_206 = torch.nn.functional.dropout(x_205, 0.0, False, False)
        x_205 = None
        x_207 = mul_58 + x_206
        mul_58 = x_206 = None
        view_32 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_60 = x_207 * view_32
        view_32 = None
        group_norm_56 = torch.nn.functional.group_norm(
            x_207,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_207 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_ = (None)
        y_28 = torch._C._nn.avg_pool2d(group_norm_56, 3, 1, 1, False, False, None)
        sub_28 = y_28 - group_norm_56
        y_28 = group_norm_56 = None
        x_208 = mul_60 + sub_28
        mul_60 = sub_28 = None
        view_33 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_61 = x_208 * view_33
        view_33 = None
        group_norm_57 = torch.nn.functional.group_norm(
            x_208,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_208 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_ = (None)
        x_209 = torch.conv2d(
            group_norm_57,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_57 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_28 = torch.nn.functional.relu(x_209, inplace=False)
        x_209 = None
        pow_29 = relu_28**2
        relu_28 = None
        mul_62 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_scale_
            * pow_29
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_scale_ = (
            pow_29
        ) = None
        x_210 = (
            mul_62
            + l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_bias_
        )
        mul_62 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_bias_ = (None)
        x_211 = torch.nn.functional.dropout(x_210, 0.0, False, False)
        x_210 = None
        x_212 = torch.conv2d(
            x_211,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_211 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_213 = torch.nn.functional.dropout(x_212, 0.0, False, False)
        x_212 = None
        x_214 = mul_61 + x_213
        mul_61 = x_213 = None
        view_34 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_63 = x_214 * view_34
        view_34 = None
        group_norm_58 = torch.nn.functional.group_norm(
            x_214,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_214 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_ = (None)
        y_29 = torch._C._nn.avg_pool2d(group_norm_58, 3, 1, 1, False, False, None)
        sub_29 = y_29 - group_norm_58
        y_29 = group_norm_58 = None
        x_215 = mul_63 + sub_29
        mul_63 = sub_29 = None
        view_35 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_64 = x_215 * view_35
        view_35 = None
        group_norm_59 = torch.nn.functional.group_norm(
            x_215,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_215 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_ = (None)
        x_216 = torch.conv2d(
            group_norm_59,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_59 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_29 = torch.nn.functional.relu(x_216, inplace=False)
        x_216 = None
        pow_30 = relu_29**2
        relu_29 = None
        mul_65 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_act_parameters_scale_
            * pow_30
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_act_parameters_scale_ = (
            pow_30
        ) = None
        x_217 = (
            mul_65
            + l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_act_parameters_bias_
        )
        mul_65 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_act_parameters_bias_ = (None)
        x_218 = torch.nn.functional.dropout(x_217, 0.0, False, False)
        x_217 = None
        x_219 = torch.conv2d(
            x_218,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_218 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_220 = torch.nn.functional.dropout(x_219, 0.0, False, False)
        x_219 = None
        x_221 = mul_64 + x_220
        mul_64 = x_220 = None
        x_222 = x_221.permute(0, 2, 3, 1)
        x_221 = None
        x_223 = torch.nn.functional.layer_norm(
            x_222,
            (384,),
            l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_,
            None,
            1e-06,
        )
        x_222 = l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_ = (None)
        x_224 = x_223.permute(0, 3, 1, 2)
        x_223 = None
        x_225 = torch.conv2d(
            x_224,
            l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_224 = l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_ = (None)
        floordiv_3 = sym_sum // 32
        sym_sum = None
        sym_sum_4 = torch.sym_sum([1, floordiv_3])
        floordiv_3 = sym_sum_4 = None
        view_36 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale1_parameters_scale_.view(
            (768, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_66 = x_225 * view_36
        view_36 = None
        group_norm_60 = torch.nn.functional.group_norm(
            x_225,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_225 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (None)
        y_30 = torch._C._nn.avg_pool2d(group_norm_60, 3, 1, 1, False, False, None)
        sub_30 = y_30 - group_norm_60
        y_30 = group_norm_60 = None
        x_226 = mul_66 + sub_30
        mul_66 = sub_30 = None
        view_37 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale2_parameters_scale_.view(
            (768, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_67 = x_226 * view_37
        view_37 = None
        group_norm_61 = torch.nn.functional.group_norm(
            x_226,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_226 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (None)
        x_227 = torch.conv2d(
            group_norm_61,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_61 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_30 = torch.nn.functional.relu(x_227, inplace=False)
        x_227 = None
        pow_31 = relu_30**2
        relu_30 = None
        mul_68 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
            * pow_31
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = (
            pow_31
        ) = None
        x_228 = (
            mul_68
            + l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        )
        mul_68 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = (None)
        x_229 = torch.nn.functional.dropout(x_228, 0.0, False, False)
        x_228 = None
        x_230 = torch.conv2d(
            x_229,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_229 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_231 = torch.nn.functional.dropout(x_230, 0.0, False, False)
        x_230 = None
        x_232 = mul_67 + x_231
        mul_67 = x_231 = None
        view_38 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale1_parameters_scale_.view(
            (768, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_69 = x_232 * view_38
        view_38 = None
        group_norm_62 = torch.nn.functional.group_norm(
            x_232,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_232 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (None)
        y_31 = torch._C._nn.avg_pool2d(group_norm_62, 3, 1, 1, False, False, None)
        sub_31 = y_31 - group_norm_62
        y_31 = group_norm_62 = None
        x_233 = mul_69 + sub_31
        mul_69 = sub_31 = None
        view_39 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale2_parameters_scale_.view(
            (768, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_70 = x_233 * view_39
        view_39 = None
        group_norm_63 = torch.nn.functional.group_norm(
            x_233,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_233 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (None)
        x_234 = torch.conv2d(
            group_norm_63,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_63 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_31 = torch.nn.functional.relu(x_234, inplace=False)
        x_234 = None
        pow_32 = relu_31**2
        relu_31 = None
        mul_71 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
            * pow_32
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = (
            pow_32
        ) = None
        x_235 = (
            mul_71
            + l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        )
        mul_71 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = (None)
        x_236 = torch.nn.functional.dropout(x_235, 0.0, False, False)
        x_235 = None
        x_237 = torch.conv2d(
            x_236,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_236 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_238 = torch.nn.functional.dropout(x_237, 0.0, False, False)
        x_237 = None
        x_239 = mul_70 + x_238
        mul_70 = x_238 = None
        view_40 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale1_parameters_scale_.view(
            (768, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_72 = x_239 * view_40
        view_40 = None
        group_norm_64 = torch.nn.functional.group_norm(
            x_239,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_239 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (None)
        y_32 = torch._C._nn.avg_pool2d(group_norm_64, 3, 1, 1, False, False, None)
        sub_32 = y_32 - group_norm_64
        y_32 = group_norm_64 = None
        x_240 = mul_72 + sub_32
        mul_72 = sub_32 = None
        view_41 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale2_parameters_scale_.view(
            (768, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_73 = x_240 * view_41
        view_41 = None
        group_norm_65 = torch.nn.functional.group_norm(
            x_240,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_240 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (None)
        x_241 = torch.conv2d(
            group_norm_65,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_65 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_32 = torch.nn.functional.relu(x_241, inplace=False)
        x_241 = None
        pow_33 = relu_32**2
        relu_32 = None
        mul_74 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_
            * pow_33
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_ = (
            pow_33
        ) = None
        x_242 = (
            mul_74
            + l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_
        )
        mul_74 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_ = (None)
        x_243 = torch.nn.functional.dropout(x_242, 0.0, False, False)
        x_242 = None
        x_244 = torch.conv2d(
            x_243,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_243 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_245 = torch.nn.functional.dropout(x_244, 0.0, False, False)
        x_244 = None
        x_246 = mul_73 + x_245
        mul_73 = x_245 = None
        view_42 = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_res_scale1_parameters_scale_.view(
            (768, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_75 = x_246 * view_42
        view_42 = None
        group_norm_66 = torch.nn.functional.group_norm(
            x_246,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_246 = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm1_parameters_weight_ = (None)
        y_33 = torch._C._nn.avg_pool2d(group_norm_66, 3, 1, 1, False, False, None)
        sub_33 = y_33 - group_norm_66
        y_33 = group_norm_66 = None
        x_247 = mul_75 + sub_33
        mul_75 = sub_33 = None
        view_43 = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_res_scale2_parameters_scale_.view(
            (768, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_76 = x_247 * view_43
        view_43 = None
        group_norm_67 = torch.nn.functional.group_norm(
            x_247,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_247 = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm2_parameters_weight_ = (None)
        x_248 = torch.conv2d(
            group_norm_67,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_67 = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_33 = torch.nn.functional.relu(x_248, inplace=False)
        x_248 = None
        pow_34 = relu_33**2
        relu_33 = None
        mul_77 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_
            * pow_34
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_ = (
            pow_34
        ) = None
        x_249 = (
            mul_77
            + l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_
        )
        mul_77 = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_ = (None)
        x_250 = torch.nn.functional.dropout(x_249, 0.0, False, False)
        x_249 = None
        x_251 = torch.conv2d(
            x_250,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_250 = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_252 = torch.nn.functional.dropout(x_251, 0.0, False, False)
        x_251 = None
        x_253 = mul_76 + x_252
        mul_76 = x_252 = None
        view_44 = l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_res_scale1_parameters_scale_.view(
            (768, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_78 = x_253 * view_44
        view_44 = None
        group_norm_68 = torch.nn.functional.group_norm(
            x_253,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_253 = l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm1_parameters_weight_ = (None)
        y_34 = torch._C._nn.avg_pool2d(group_norm_68, 3, 1, 1, False, False, None)
        sub_34 = y_34 - group_norm_68
        y_34 = group_norm_68 = None
        x_254 = mul_78 + sub_34
        mul_78 = sub_34 = None
        view_45 = l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_res_scale2_parameters_scale_.view(
            (768, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_79 = x_254 * view_45
        view_45 = None
        group_norm_69 = torch.nn.functional.group_norm(
            x_254,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_254 = l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm2_parameters_weight_ = (None)
        x_255 = torch.conv2d(
            group_norm_69,
            l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_69 = l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_34 = torch.nn.functional.relu(x_255, inplace=False)
        x_255 = None
        pow_35 = relu_34**2
        relu_34 = None
        mul_80 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_
            * pow_35
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_ = (
            pow_35
        ) = None
        x_256 = (
            mul_80
            + l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_
        )
        mul_80 = l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_ = (None)
        x_257 = torch.nn.functional.dropout(x_256, 0.0, False, False)
        x_256 = None
        x_258 = torch.conv2d(
            x_257,
            l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_257 = l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_259 = torch.nn.functional.dropout(x_258, 0.0, False, False)
        x_258 = None
        x_260 = mul_79 + x_259
        mul_79 = x_259 = None
        view_46 = l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_res_scale1_parameters_scale_.view(
            (768, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_81 = x_260 * view_46
        view_46 = None
        group_norm_70 = torch.nn.functional.group_norm(
            x_260,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_260 = l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm1_parameters_weight_ = (None)
        y_35 = torch._C._nn.avg_pool2d(group_norm_70, 3, 1, 1, False, False, None)
        sub_35 = y_35 - group_norm_70
        y_35 = group_norm_70 = None
        x_261 = mul_81 + sub_35
        mul_81 = sub_35 = None
        view_47 = l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_res_scale2_parameters_scale_.view(
            (768, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_82 = x_261 * view_47
        view_47 = None
        group_norm_71 = torch.nn.functional.group_norm(
            x_261,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_261 = l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm2_parameters_weight_ = (None)
        x_262 = torch.conv2d(
            group_norm_71,
            l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_71 = l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_35 = torch.nn.functional.relu(x_262, inplace=False)
        x_262 = None
        pow_36 = relu_35**2
        relu_35 = None
        mul_83 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_
            * pow_36
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_ = (
            pow_36
        ) = None
        x_263 = (
            mul_83
            + l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_
        )
        mul_83 = l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_ = (None)
        x_264 = torch.nn.functional.dropout(x_263, 0.0, False, False)
        x_263 = None
        x_265 = torch.conv2d(
            x_264,
            l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_264 = l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_266 = torch.nn.functional.dropout(x_265, 0.0, False, False)
        x_265 = None
        x_267 = mul_82 + x_266
        mul_82 = x_266 = None
        x_268 = torch.nn.functional.adaptive_avg_pool2d(x_267, 1)
        x_267 = None
        x_269 = x_268.permute(0, 2, 3, 1)
        x_268 = None
        x_270 = torch.nn.functional.layer_norm(
            x_269,
            (768,),
            l_self_modules_head_modules_norm_parameters_weight_,
            l_self_modules_head_modules_norm_parameters_bias_,
            1e-06,
        )
        x_269 = (
            l_self_modules_head_modules_norm_parameters_weight_
        ) = l_self_modules_head_modules_norm_parameters_bias_ = None
        x_271 = x_270.permute(0, 3, 1, 2)
        x_270 = None
        x_272 = x_271.flatten(1, -1)
        x_271 = None
        x_273 = torch._C._nn.linear(
            x_272,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_272 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_273,)
