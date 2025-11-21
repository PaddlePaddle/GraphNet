import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_ = L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_pwconv2_parameters_weight_
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
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_pwconv1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_pwconv1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_pwconv2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_pwconv2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_head_modules_norm_parameters_weight_ = (
            L_self_modules_head_modules_norm_parameters_weight_
        )
        l_self_modules_head_modules_norm_parameters_bias_ = (
            L_self_modules_head_modules_norm_parameters_bias_
        )
        l_self_modules_head_modules_fc_modules_fc1_parameters_weight_ = (
            L_self_modules_head_modules_fc_modules_fc1_parameters_weight_
        )
        l_self_modules_head_modules_fc_modules_fc1_parameters_bias_ = (
            L_self_modules_head_modules_fc_modules_fc1_parameters_bias_
        )
        l_self_modules_head_modules_fc_modules_norm_parameters_weight_ = (
            L_self_modules_head_modules_fc_modules_norm_parameters_weight_
        )
        l_self_modules_head_modules_fc_modules_norm_parameters_bias_ = (
            L_self_modules_head_modules_fc_modules_norm_parameters_bias_
        )
        l_self_modules_head_modules_fc_modules_fc2_parameters_weight_ = (
            L_self_modules_head_modules_fc_modules_fc2_parameters_weight_
        )
        l_self_modules_head_modules_fc_modules_fc2_parameters_bias_ = (
            L_self_modules_head_modules_fc_modules_fc2_parameters_bias_
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
        x_4 = x_3.permute(0, 2, 3, 1)
        x_5 = torch.nn.functional.layer_norm(
            x_4,
            (96,),
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_4 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (None)
        x_6 = x_5.permute(0, 3, 1, 2)
        x_5 = None
        x_7 = torch.conv2d(
            x_6,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_6 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu = torch.nn.functional.relu(x_7, inplace=False)
        x_7 = None
        pow_1 = relu**2
        relu = None
        mul = (
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_scale_
            * pow_1
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_1
        ) = None
        x_8 = (
            mul
            + l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_9 = torch.conv2d(
            x_8,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        x_8 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_10 = torch.conv2d(
            x_9,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_9 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_11 = x_3 + x_10
        x_3 = x_10 = None
        x_12 = x_11.permute(0, 2, 3, 1)
        x_13 = torch.nn.functional.layer_norm(
            x_12,
            (96,),
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_12 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (None)
        x_14 = x_13.permute(0, 3, 1, 2)
        x_13 = None
        x_15 = torch.conv2d(
            x_14,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_14 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_1 = torch.nn.functional.relu(x_15, inplace=False)
        x_15 = None
        pow_2 = relu_1**2
        relu_1 = None
        mul_1 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
            * pow_2
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = (
            pow_2
        ) = None
        x_16 = (
            mul_1
            + l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        )
        mul_1 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = (None)
        x_17 = torch.nn.functional.dropout(x_16, 0.0, False, False)
        x_16 = None
        x_18 = torch.conv2d(
            x_17,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_17 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_19 = torch.nn.functional.dropout(x_18, 0.0, False, False)
        x_18 = None
        x_20 = x_11 + x_19
        x_11 = x_19 = None
        x_21 = x_20.permute(0, 2, 3, 1)
        x_22 = torch.nn.functional.layer_norm(
            x_21,
            (96,),
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_21 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (None)
        x_23 = x_22.permute(0, 3, 1, 2)
        x_22 = None
        x_24 = torch.conv2d(
            x_23,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_23 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_2 = torch.nn.functional.relu(x_24, inplace=False)
        x_24 = None
        pow_3 = relu_2**2
        relu_2 = None
        mul_2 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_scale_
            * pow_3
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_3
        ) = None
        x_25 = (
            mul_2
            + l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_2 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_26 = torch.conv2d(
            x_25,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        x_25 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_27 = torch.conv2d(
            x_26,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_26 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_28 = x_20 + x_27
        x_20 = x_27 = None
        x_29 = x_28.permute(0, 2, 3, 1)
        x_30 = torch.nn.functional.layer_norm(
            x_29,
            (96,),
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_29 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (None)
        x_31 = x_30.permute(0, 3, 1, 2)
        x_30 = None
        x_32 = torch.conv2d(
            x_31,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_31 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_3 = torch.nn.functional.relu(x_32, inplace=False)
        x_32 = None
        pow_4 = relu_3**2
        relu_3 = None
        mul_3 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
            * pow_4
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = (
            pow_4
        ) = None
        x_33 = (
            mul_3
            + l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        )
        mul_3 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = (None)
        x_34 = torch.nn.functional.dropout(x_33, 0.0, False, False)
        x_33 = None
        x_35 = torch.conv2d(
            x_34,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_34 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_36 = torch.nn.functional.dropout(x_35, 0.0, False, False)
        x_35 = None
        x_37 = x_28 + x_36
        x_28 = x_36 = None
        x_38 = x_37.permute(0, 2, 3, 1)
        x_39 = torch.nn.functional.layer_norm(
            x_38,
            (96,),
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_38 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (None)
        x_40 = x_39.permute(0, 3, 1, 2)
        x_39 = None
        x_41 = torch.conv2d(
            x_40,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_40 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_4 = torch.nn.functional.relu(x_41, inplace=False)
        x_41 = None
        pow_5 = relu_4**2
        relu_4 = None
        mul_4 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_scale_
            * pow_5
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_5
        ) = None
        x_42 = (
            mul_4
            + l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_4 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_43 = torch.conv2d(
            x_42,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        x_42 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_44 = torch.conv2d(
            x_43,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_43 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_45 = x_37 + x_44
        x_37 = x_44 = None
        x_46 = x_45.permute(0, 2, 3, 1)
        x_47 = torch.nn.functional.layer_norm(
            x_46,
            (96,),
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_46 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (None)
        x_48 = x_47.permute(0, 3, 1, 2)
        x_47 = None
        x_49 = torch.conv2d(
            x_48,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_48 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_5 = torch.nn.functional.relu(x_49, inplace=False)
        x_49 = None
        pow_6 = relu_5**2
        relu_5 = None
        mul_5 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_
            * pow_6
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_ = (
            pow_6
        ) = None
        x_50 = (
            mul_5
            + l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_
        )
        mul_5 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_ = (None)
        x_51 = torch.nn.functional.dropout(x_50, 0.0, False, False)
        x_50 = None
        x_52 = torch.conv2d(
            x_51,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_51 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_53 = torch.nn.functional.dropout(x_52, 0.0, False, False)
        x_52 = None
        x_54 = x_45 + x_53
        x_45 = x_53 = None
        x_55 = x_54.permute(0, 2, 3, 1)
        x_54 = None
        x_56 = torch.nn.functional.layer_norm(
            x_55,
            (96,),
            l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_,
            None,
            1e-06,
        )
        x_55 = l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_ = (None)
        x_57 = x_56.permute(0, 3, 1, 2)
        x_56 = None
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_57 = l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_ = l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_ = (None)
        x_59 = x_58.permute(0, 2, 3, 1)
        x_60 = torch.nn.functional.layer_norm(
            x_59,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_59 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (None)
        x_61 = x_60.permute(0, 3, 1, 2)
        x_60 = None
        x_62 = torch.conv2d(
            x_61,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_61 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_6 = torch.nn.functional.relu(x_62, inplace=False)
        x_62 = None
        pow_7 = relu_6**2
        relu_6 = None
        mul_6 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_scale_
            * pow_7
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_7
        ) = None
        x_63 = (
            mul_6
            + l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_6 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_64 = torch.conv2d(
            x_63,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        x_63 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_65 = torch.conv2d(
            x_64,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_64 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_66 = x_58 + x_65
        x_58 = x_65 = None
        x_67 = x_66.permute(0, 2, 3, 1)
        x_68 = torch.nn.functional.layer_norm(
            x_67,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_67 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (None)
        x_69 = x_68.permute(0, 3, 1, 2)
        x_68 = None
        x_70 = torch.conv2d(
            x_69,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_69 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_7 = torch.nn.functional.relu(x_70, inplace=False)
        x_70 = None
        pow_8 = relu_7**2
        relu_7 = None
        mul_7 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
            * pow_8
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = (
            pow_8
        ) = None
        x_71 = (
            mul_7
            + l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        )
        mul_7 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = (None)
        x_72 = torch.nn.functional.dropout(x_71, 0.0, False, False)
        x_71 = None
        x_73 = torch.conv2d(
            x_72,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_72 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_74 = torch.nn.functional.dropout(x_73, 0.0, False, False)
        x_73 = None
        x_75 = x_66 + x_74
        x_66 = x_74 = None
        x_76 = x_75.permute(0, 2, 3, 1)
        x_77 = torch.nn.functional.layer_norm(
            x_76,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_76 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (None)
        x_78 = x_77.permute(0, 3, 1, 2)
        x_77 = None
        x_79 = torch.conv2d(
            x_78,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_78 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_8 = torch.nn.functional.relu(x_79, inplace=False)
        x_79 = None
        pow_9 = relu_8**2
        relu_8 = None
        mul_8 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_scale_
            * pow_9
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_9
        ) = None
        x_80 = (
            mul_8
            + l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_8 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_81 = torch.conv2d(
            x_80,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        x_80 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_82 = torch.conv2d(
            x_81,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_81 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_83 = x_75 + x_82
        x_75 = x_82 = None
        x_84 = x_83.permute(0, 2, 3, 1)
        x_85 = torch.nn.functional.layer_norm(
            x_84,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_84 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (None)
        x_86 = x_85.permute(0, 3, 1, 2)
        x_85 = None
        x_87 = torch.conv2d(
            x_86,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_86 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_9 = torch.nn.functional.relu(x_87, inplace=False)
        x_87 = None
        pow_10 = relu_9**2
        relu_9 = None
        mul_9 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
            * pow_10
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = (
            pow_10
        ) = None
        x_88 = (
            mul_9
            + l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        )
        mul_9 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = (None)
        x_89 = torch.nn.functional.dropout(x_88, 0.0, False, False)
        x_88 = None
        x_90 = torch.conv2d(
            x_89,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_89 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_91 = torch.nn.functional.dropout(x_90, 0.0, False, False)
        x_90 = None
        x_92 = x_83 + x_91
        x_83 = x_91 = None
        x_93 = x_92.permute(0, 2, 3, 1)
        x_94 = torch.nn.functional.layer_norm(
            x_93,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_93 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (None)
        x_95 = x_94.permute(0, 3, 1, 2)
        x_94 = None
        x_96 = torch.conv2d(
            x_95,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_95 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_10 = torch.nn.functional.relu(x_96, inplace=False)
        x_96 = None
        pow_11 = relu_10**2
        relu_10 = None
        mul_10 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_scale_
            * pow_11
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_11
        ) = None
        x_97 = (
            mul_10
            + l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_10 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_98 = torch.conv2d(
            x_97,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        x_97 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_99 = torch.conv2d(
            x_98,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_98 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_100 = x_92 + x_99
        x_92 = x_99 = None
        x_101 = x_100.permute(0, 2, 3, 1)
        x_102 = torch.nn.functional.layer_norm(
            x_101,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_101 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (None)
        x_103 = x_102.permute(0, 3, 1, 2)
        x_102 = None
        x_104 = torch.conv2d(
            x_103,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_103 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_11 = torch.nn.functional.relu(x_104, inplace=False)
        x_104 = None
        pow_12 = relu_11**2
        relu_11 = None
        mul_11 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_
            * pow_12
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_ = (
            pow_12
        ) = None
        x_105 = (
            mul_11
            + l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_
        )
        mul_11 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_ = (None)
        x_106 = torch.nn.functional.dropout(x_105, 0.0, False, False)
        x_105 = None
        x_107 = torch.conv2d(
            x_106,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_106 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_108 = torch.nn.functional.dropout(x_107, 0.0, False, False)
        x_107 = None
        x_109 = x_100 + x_108
        x_100 = x_108 = None
        x_110 = x_109.permute(0, 2, 3, 1)
        x_111 = torch.nn.functional.layer_norm(
            x_110,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_110 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_ = (None)
        x_112 = x_111.permute(0, 3, 1, 2)
        x_111 = None
        x_113 = torch.conv2d(
            x_112,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_112 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_12 = torch.nn.functional.relu(x_113, inplace=False)
        x_113 = None
        pow_13 = relu_12**2
        relu_12 = None
        mul_12 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_token_mixer_modules_act1_parameters_scale_
            * pow_13
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_13
        ) = None
        x_114 = (
            mul_12
            + l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_12 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_115 = torch.conv2d(
            x_114,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        x_114 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_116 = torch.conv2d(
            x_115,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_115 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_117 = x_109 + x_116
        x_109 = x_116 = None
        x_118 = x_117.permute(0, 2, 3, 1)
        x_119 = torch.nn.functional.layer_norm(
            x_118,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_118 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_ = (None)
        x_120 = x_119.permute(0, 3, 1, 2)
        x_119 = None
        x_121 = torch.conv2d(
            x_120,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_120 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_13 = torch.nn.functional.relu(x_121, inplace=False)
        x_121 = None
        pow_14 = relu_13**2
        relu_13 = None
        mul_13 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_
            * pow_14
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_ = (
            pow_14
        ) = None
        x_122 = (
            mul_13
            + l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_
        )
        mul_13 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_ = (None)
        x_123 = torch.nn.functional.dropout(x_122, 0.0, False, False)
        x_122 = None
        x_124 = torch.conv2d(
            x_123,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_123 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_125 = torch.nn.functional.dropout(x_124, 0.0, False, False)
        x_124 = None
        x_126 = x_117 + x_125
        x_117 = x_125 = None
        x_127 = x_126.permute(0, 2, 3, 1)
        x_128 = torch.nn.functional.layer_norm(
            x_127,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_127 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_weight_ = (None)
        x_129 = x_128.permute(0, 3, 1, 2)
        x_128 = None
        x_130 = torch.conv2d(
            x_129,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_129 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_14 = torch.nn.functional.relu(x_130, inplace=False)
        x_130 = None
        pow_15 = relu_14**2
        relu_14 = None
        mul_14 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_token_mixer_modules_act1_parameters_scale_
            * pow_15
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_15
        ) = None
        x_131 = (
            mul_14
            + l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_14 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_132 = torch.conv2d(
            x_131,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        x_131 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_133 = torch.conv2d(
            x_132,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_132 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_134 = x_126 + x_133
        x_126 = x_133 = None
        x_135 = x_134.permute(0, 2, 3, 1)
        x_136 = torch.nn.functional.layer_norm(
            x_135,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_135 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_weight_ = (None)
        x_137 = x_136.permute(0, 3, 1, 2)
        x_136 = None
        x_138 = torch.conv2d(
            x_137,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_137 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_15 = torch.nn.functional.relu(x_138, inplace=False)
        x_138 = None
        pow_16 = relu_15**2
        relu_15 = None
        mul_15 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_
            * pow_16
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_ = (
            pow_16
        ) = None
        x_139 = (
            mul_15
            + l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_
        )
        mul_15 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_ = (None)
        x_140 = torch.nn.functional.dropout(x_139, 0.0, False, False)
        x_139 = None
        x_141 = torch.conv2d(
            x_140,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_140 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_142 = torch.nn.functional.dropout(x_141, 0.0, False, False)
        x_141 = None
        x_143 = x_134 + x_142
        x_134 = x_142 = None
        x_144 = x_143.permute(0, 2, 3, 1)
        x_145 = torch.nn.functional.layer_norm(
            x_144,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_144 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_weight_ = (None)
        x_146 = x_145.permute(0, 3, 1, 2)
        x_145 = None
        x_147 = torch.conv2d(
            x_146,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_146 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_16 = torch.nn.functional.relu(x_147, inplace=False)
        x_147 = None
        pow_17 = relu_16**2
        relu_16 = None
        mul_16 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_token_mixer_modules_act1_parameters_scale_
            * pow_17
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_17
        ) = None
        x_148 = (
            mul_16
            + l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_16 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_149 = torch.conv2d(
            x_148,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        x_148 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_150 = torch.conv2d(
            x_149,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_149 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_151 = x_143 + x_150
        x_143 = x_150 = None
        x_152 = x_151.permute(0, 2, 3, 1)
        x_153 = torch.nn.functional.layer_norm(
            x_152,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_152 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_weight_ = (None)
        x_154 = x_153.permute(0, 3, 1, 2)
        x_153 = None
        x_155 = torch.conv2d(
            x_154,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_154 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_17 = torch.nn.functional.relu(x_155, inplace=False)
        x_155 = None
        pow_18 = relu_17**2
        relu_17 = None
        mul_17 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_
            * pow_18
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_ = (
            pow_18
        ) = None
        x_156 = (
            mul_17
            + l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_
        )
        mul_17 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_ = (None)
        x_157 = torch.nn.functional.dropout(x_156, 0.0, False, False)
        x_156 = None
        x_158 = torch.conv2d(
            x_157,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_157 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_159 = torch.nn.functional.dropout(x_158, 0.0, False, False)
        x_158 = None
        x_160 = x_151 + x_159
        x_151 = x_159 = None
        x_161 = x_160.permute(0, 2, 3, 1)
        x_162 = torch.nn.functional.layer_norm(
            x_161,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_161 = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_weight_ = (None)
        x_163 = x_162.permute(0, 3, 1, 2)
        x_162 = None
        x_164 = torch.conv2d(
            x_163,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_163 = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_18 = torch.nn.functional.relu(x_164, inplace=False)
        x_164 = None
        pow_19 = relu_18**2
        relu_18 = None
        mul_18 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_token_mixer_modules_act1_parameters_scale_
            * pow_19
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_19
        ) = None
        x_165 = (
            mul_18
            + l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_18 = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_166 = torch.conv2d(
            x_165,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        x_165 = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_167 = torch.conv2d(
            x_166,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_166 = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_168 = x_160 + x_167
        x_160 = x_167 = None
        x_169 = x_168.permute(0, 2, 3, 1)
        x_170 = torch.nn.functional.layer_norm(
            x_169,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_169 = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_weight_ = (None)
        x_171 = x_170.permute(0, 3, 1, 2)
        x_170 = None
        x_172 = torch.conv2d(
            x_171,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_171 = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_19 = torch.nn.functional.relu(x_172, inplace=False)
        x_172 = None
        pow_20 = relu_19**2
        relu_19 = None
        mul_19 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_act_parameters_scale_
            * pow_20
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_act_parameters_scale_ = (
            pow_20
        ) = None
        x_173 = (
            mul_19
            + l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_act_parameters_bias_
        )
        mul_19 = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_act_parameters_bias_ = (None)
        x_174 = torch.nn.functional.dropout(x_173, 0.0, False, False)
        x_173 = None
        x_175 = torch.conv2d(
            x_174,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_174 = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_176 = torch.nn.functional.dropout(x_175, 0.0, False, False)
        x_175 = None
        x_177 = x_168 + x_176
        x_168 = x_176 = None
        x_178 = x_177.permute(0, 2, 3, 1)
        x_179 = torch.nn.functional.layer_norm(
            x_178,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_178 = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_weight_ = (None)
        x_180 = x_179.permute(0, 3, 1, 2)
        x_179 = None
        x_181 = torch.conv2d(
            x_180,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_180 = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_20 = torch.nn.functional.relu(x_181, inplace=False)
        x_181 = None
        pow_21 = relu_20**2
        relu_20 = None
        mul_20 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_token_mixer_modules_act1_parameters_scale_
            * pow_21
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_21
        ) = None
        x_182 = (
            mul_20
            + l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_20 = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_183 = torch.conv2d(
            x_182,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        x_182 = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_184 = torch.conv2d(
            x_183,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_183 = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_185 = x_177 + x_184
        x_177 = x_184 = None
        x_186 = x_185.permute(0, 2, 3, 1)
        x_187 = torch.nn.functional.layer_norm(
            x_186,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_186 = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_weight_ = (None)
        x_188 = x_187.permute(0, 3, 1, 2)
        x_187 = None
        x_189 = torch.conv2d(
            x_188,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_188 = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_21 = torch.nn.functional.relu(x_189, inplace=False)
        x_189 = None
        pow_22 = relu_21**2
        relu_21 = None
        mul_21 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_act_parameters_scale_
            * pow_22
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_act_parameters_scale_ = (
            pow_22
        ) = None
        x_190 = (
            mul_21
            + l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_act_parameters_bias_
        )
        mul_21 = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_act_parameters_bias_ = (None)
        x_191 = torch.nn.functional.dropout(x_190, 0.0, False, False)
        x_190 = None
        x_192 = torch.conv2d(
            x_191,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_191 = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_193 = torch.nn.functional.dropout(x_192, 0.0, False, False)
        x_192 = None
        x_194 = x_185 + x_193
        x_185 = x_193 = None
        x_195 = x_194.permute(0, 2, 3, 1)
        x_196 = torch.nn.functional.layer_norm(
            x_195,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_195 = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_parameters_weight_ = (None)
        x_197 = x_196.permute(0, 3, 1, 2)
        x_196 = None
        x_198 = torch.conv2d(
            x_197,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_197 = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_22 = torch.nn.functional.relu(x_198, inplace=False)
        x_198 = None
        pow_23 = relu_22**2
        relu_22 = None
        mul_22 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_token_mixer_modules_act1_parameters_scale_
            * pow_23
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_23
        ) = None
        x_199 = (
            mul_22
            + l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_22 = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_200 = torch.conv2d(
            x_199,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        x_199 = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_201 = torch.conv2d(
            x_200,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_200 = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_202 = x_194 + x_201
        x_194 = x_201 = None
        x_203 = x_202.permute(0, 2, 3, 1)
        x_204 = torch.nn.functional.layer_norm(
            x_203,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_203 = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_parameters_weight_ = (None)
        x_205 = x_204.permute(0, 3, 1, 2)
        x_204 = None
        x_206 = torch.conv2d(
            x_205,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_205 = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_23 = torch.nn.functional.relu(x_206, inplace=False)
        x_206 = None
        pow_24 = relu_23**2
        relu_23 = None
        mul_23 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_mlp_modules_act_parameters_scale_
            * pow_24
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_mlp_modules_act_parameters_scale_ = (
            pow_24
        ) = None
        x_207 = (
            mul_23
            + l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_mlp_modules_act_parameters_bias_
        )
        mul_23 = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_mlp_modules_act_parameters_bias_ = (None)
        x_208 = torch.nn.functional.dropout(x_207, 0.0, False, False)
        x_207 = None
        x_209 = torch.conv2d(
            x_208,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_208 = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_210 = torch.nn.functional.dropout(x_209, 0.0, False, False)
        x_209 = None
        x_211 = x_202 + x_210
        x_202 = x_210 = None
        x_212 = x_211.permute(0, 2, 3, 1)
        x_213 = torch.nn.functional.layer_norm(
            x_212,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_212 = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_parameters_weight_ = (None)
        x_214 = x_213.permute(0, 3, 1, 2)
        x_213 = None
        x_215 = torch.conv2d(
            x_214,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_214 = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_24 = torch.nn.functional.relu(x_215, inplace=False)
        x_215 = None
        pow_25 = relu_24**2
        relu_24 = None
        mul_24 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_token_mixer_modules_act1_parameters_scale_
            * pow_25
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_25
        ) = None
        x_216 = (
            mul_24
            + l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_24 = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_217 = torch.conv2d(
            x_216,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        x_216 = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_218 = torch.conv2d(
            x_217,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_217 = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_219 = x_211 + x_218
        x_211 = x_218 = None
        x_220 = x_219.permute(0, 2, 3, 1)
        x_221 = torch.nn.functional.layer_norm(
            x_220,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_220 = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_parameters_weight_ = (None)
        x_222 = x_221.permute(0, 3, 1, 2)
        x_221 = None
        x_223 = torch.conv2d(
            x_222,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_222 = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_25 = torch.nn.functional.relu(x_223, inplace=False)
        x_223 = None
        pow_26 = relu_25**2
        relu_25 = None
        mul_25 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_mlp_modules_act_parameters_scale_
            * pow_26
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_mlp_modules_act_parameters_scale_ = (
            pow_26
        ) = None
        x_224 = (
            mul_25
            + l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_mlp_modules_act_parameters_bias_
        )
        mul_25 = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_mlp_modules_act_parameters_bias_ = (None)
        x_225 = torch.nn.functional.dropout(x_224, 0.0, False, False)
        x_224 = None
        x_226 = torch.conv2d(
            x_225,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_225 = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_227 = torch.nn.functional.dropout(x_226, 0.0, False, False)
        x_226 = None
        x_228 = x_219 + x_227
        x_219 = x_227 = None
        x_229 = x_228.permute(0, 2, 3, 1)
        x_230 = torch.nn.functional.layer_norm(
            x_229,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_229 = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_parameters_weight_ = (None)
        x_231 = x_230.permute(0, 3, 1, 2)
        x_230 = None
        x_232 = torch.conv2d(
            x_231,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_231 = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_26 = torch.nn.functional.relu(x_232, inplace=False)
        x_232 = None
        pow_27 = relu_26**2
        relu_26 = None
        mul_26 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_token_mixer_modules_act1_parameters_scale_
            * pow_27
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_27
        ) = None
        x_233 = (
            mul_26
            + l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_26 = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_234 = torch.conv2d(
            x_233,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        x_233 = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_235 = torch.conv2d(
            x_234,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_234 = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_236 = x_228 + x_235
        x_228 = x_235 = None
        x_237 = x_236.permute(0, 2, 3, 1)
        x_238 = torch.nn.functional.layer_norm(
            x_237,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_237 = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_parameters_weight_ = (None)
        x_239 = x_238.permute(0, 3, 1, 2)
        x_238 = None
        x_240 = torch.conv2d(
            x_239,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_239 = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_27 = torch.nn.functional.relu(x_240, inplace=False)
        x_240 = None
        pow_28 = relu_27**2
        relu_27 = None
        mul_27 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_mlp_modules_act_parameters_scale_
            * pow_28
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_mlp_modules_act_parameters_scale_ = (
            pow_28
        ) = None
        x_241 = (
            mul_27
            + l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_mlp_modules_act_parameters_bias_
        )
        mul_27 = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_mlp_modules_act_parameters_bias_ = (None)
        x_242 = torch.nn.functional.dropout(x_241, 0.0, False, False)
        x_241 = None
        x_243 = torch.conv2d(
            x_242,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_242 = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_244 = torch.nn.functional.dropout(x_243, 0.0, False, False)
        x_243 = None
        x_245 = x_236 + x_244
        x_236 = x_244 = None
        x_246 = x_245.permute(0, 2, 3, 1)
        x_247 = torch.nn.functional.layer_norm(
            x_246,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_246 = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_parameters_weight_ = (None)
        x_248 = x_247.permute(0, 3, 1, 2)
        x_247 = None
        x_249 = torch.conv2d(
            x_248,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_248 = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_28 = torch.nn.functional.relu(x_249, inplace=False)
        x_249 = None
        pow_29 = relu_28**2
        relu_28 = None
        mul_28 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_token_mixer_modules_act1_parameters_scale_
            * pow_29
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_29
        ) = None
        x_250 = (
            mul_28
            + l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_28 = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_251 = torch.conv2d(
            x_250,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        x_250 = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_252 = torch.conv2d(
            x_251,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_251 = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_253 = x_245 + x_252
        x_245 = x_252 = None
        x_254 = x_253.permute(0, 2, 3, 1)
        x_255 = torch.nn.functional.layer_norm(
            x_254,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_254 = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_parameters_weight_ = (None)
        x_256 = x_255.permute(0, 3, 1, 2)
        x_255 = None
        x_257 = torch.conv2d(
            x_256,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_256 = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_29 = torch.nn.functional.relu(x_257, inplace=False)
        x_257 = None
        pow_30 = relu_29**2
        relu_29 = None
        mul_29 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_mlp_modules_act_parameters_scale_
            * pow_30
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_mlp_modules_act_parameters_scale_ = (
            pow_30
        ) = None
        x_258 = (
            mul_29
            + l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_mlp_modules_act_parameters_bias_
        )
        mul_29 = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_mlp_modules_act_parameters_bias_ = (None)
        x_259 = torch.nn.functional.dropout(x_258, 0.0, False, False)
        x_258 = None
        x_260 = torch.conv2d(
            x_259,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_259 = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_261 = torch.nn.functional.dropout(x_260, 0.0, False, False)
        x_260 = None
        x_262 = x_253 + x_261
        x_253 = x_261 = None
        x_263 = x_262.permute(0, 2, 3, 1)
        x_262 = None
        x_264 = torch.nn.functional.layer_norm(
            x_263,
            (192,),
            l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_,
            None,
            1e-06,
        )
        x_263 = l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_ = (None)
        x_265 = x_264.permute(0, 3, 1, 2)
        x_264 = None
        x_266 = torch.conv2d(
            x_265,
            l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_265 = l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_ = (None)
        view = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_30 = x_266 * view
        view = None
        x_267 = x_266.permute(0, 2, 3, 1)
        x_266 = None
        x_268 = torch.nn.functional.layer_norm(
            x_267,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_267 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (None)
        x_269 = x_268.permute(0, 3, 1, 2)
        x_268 = None
        x_270 = torch.conv2d(
            x_269,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_269 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_30 = torch.nn.functional.relu(x_270, inplace=False)
        x_270 = None
        pow_31 = relu_30**2
        relu_30 = None
        mul_31 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_scale_
            * pow_31
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_31
        ) = None
        x_271 = (
            mul_31
            + l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_31 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_272 = torch.conv2d(
            x_271,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        x_271 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_273 = torch.conv2d(
            x_272,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_272 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_274 = mul_30 + x_273
        mul_30 = x_273 = None
        view_1 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_32 = x_274 * view_1
        view_1 = None
        x_275 = x_274.permute(0, 2, 3, 1)
        x_274 = None
        x_276 = torch.nn.functional.layer_norm(
            x_275,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_275 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (None)
        x_277 = x_276.permute(0, 3, 1, 2)
        x_276 = None
        x_278 = torch.conv2d(
            x_277,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_277 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_31 = torch.nn.functional.relu(x_278, inplace=False)
        x_278 = None
        pow_32 = relu_31**2
        relu_31 = None
        mul_33 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
            * pow_32
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = (
            pow_32
        ) = None
        x_279 = (
            mul_33
            + l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        )
        mul_33 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = (None)
        x_280 = torch.nn.functional.dropout(x_279, 0.0, False, False)
        x_279 = None
        x_281 = torch.conv2d(
            x_280,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_280 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_282 = torch.nn.functional.dropout(x_281, 0.0, False, False)
        x_281 = None
        x_283 = mul_32 + x_282
        mul_32 = x_282 = None
        view_2 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_34 = x_283 * view_2
        view_2 = None
        x_284 = x_283.permute(0, 2, 3, 1)
        x_283 = None
        x_285 = torch.nn.functional.layer_norm(
            x_284,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_284 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (None)
        x_286 = x_285.permute(0, 3, 1, 2)
        x_285 = None
        x_287 = torch.conv2d(
            x_286,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_286 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_32 = torch.nn.functional.relu(x_287, inplace=False)
        x_287 = None
        pow_33 = relu_32**2
        relu_32 = None
        mul_35 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_scale_
            * pow_33
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_33
        ) = None
        x_288 = (
            mul_35
            + l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_35 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_289 = torch.conv2d(
            x_288,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        x_288 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_290 = torch.conv2d(
            x_289,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_289 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_291 = mul_34 + x_290
        mul_34 = x_290 = None
        view_3 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_36 = x_291 * view_3
        view_3 = None
        x_292 = x_291.permute(0, 2, 3, 1)
        x_291 = None
        x_293 = torch.nn.functional.layer_norm(
            x_292,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_292 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (None)
        x_294 = x_293.permute(0, 3, 1, 2)
        x_293 = None
        x_295 = torch.conv2d(
            x_294,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_294 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_33 = torch.nn.functional.relu(x_295, inplace=False)
        x_295 = None
        pow_34 = relu_33**2
        relu_33 = None
        mul_37 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
            * pow_34
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = (
            pow_34
        ) = None
        x_296 = (
            mul_37
            + l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        )
        mul_37 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = (None)
        x_297 = torch.nn.functional.dropout(x_296, 0.0, False, False)
        x_296 = None
        x_298 = torch.conv2d(
            x_297,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_297 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_299 = torch.nn.functional.dropout(x_298, 0.0, False, False)
        x_298 = None
        x_300 = mul_36 + x_299
        mul_36 = x_299 = None
        view_4 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_38 = x_300 * view_4
        view_4 = None
        x_301 = x_300.permute(0, 2, 3, 1)
        x_300 = None
        x_302 = torch.nn.functional.layer_norm(
            x_301,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_301 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (None)
        x_303 = x_302.permute(0, 3, 1, 2)
        x_302 = None
        x_304 = torch.conv2d(
            x_303,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_303 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_34 = torch.nn.functional.relu(x_304, inplace=False)
        x_304 = None
        pow_35 = relu_34**2
        relu_34 = None
        mul_39 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_scale_
            * pow_35
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_35
        ) = None
        x_305 = (
            mul_39
            + l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_39 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_306 = torch.conv2d(
            x_305,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        x_305 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_307 = torch.conv2d(
            x_306,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_306 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_308 = mul_38 + x_307
        mul_38 = x_307 = None
        view_5 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_40 = x_308 * view_5
        view_5 = None
        x_309 = x_308.permute(0, 2, 3, 1)
        x_308 = None
        x_310 = torch.nn.functional.layer_norm(
            x_309,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_309 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (None)
        x_311 = x_310.permute(0, 3, 1, 2)
        x_310 = None
        x_312 = torch.conv2d(
            x_311,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_311 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_35 = torch.nn.functional.relu(x_312, inplace=False)
        x_312 = None
        pow_36 = relu_35**2
        relu_35 = None
        mul_41 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_
            * pow_36
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_ = (
            pow_36
        ) = None
        x_313 = (
            mul_41
            + l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_
        )
        mul_41 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_ = (None)
        x_314 = torch.nn.functional.dropout(x_313, 0.0, False, False)
        x_313 = None
        x_315 = torch.conv2d(
            x_314,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_314 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_316 = torch.nn.functional.dropout(x_315, 0.0, False, False)
        x_315 = None
        x_317 = mul_40 + x_316
        mul_40 = x_316 = None
        view_6 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_42 = x_317 * view_6
        view_6 = None
        x_318 = x_317.permute(0, 2, 3, 1)
        x_317 = None
        x_319 = torch.nn.functional.layer_norm(
            x_318,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_318 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_ = (None)
        x_320 = x_319.permute(0, 3, 1, 2)
        x_319 = None
        x_321 = torch.conv2d(
            x_320,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_320 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_36 = torch.nn.functional.relu(x_321, inplace=False)
        x_321 = None
        pow_37 = relu_36**2
        relu_36 = None
        mul_43 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_act1_parameters_scale_
            * pow_37
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_37
        ) = None
        x_322 = (
            mul_43
            + l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_43 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_323 = torch.conv2d(
            x_322,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        x_322 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_324 = torch.conv2d(
            x_323,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_323 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_325 = mul_42 + x_324
        mul_42 = x_324 = None
        view_7 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_44 = x_325 * view_7
        view_7 = None
        x_326 = x_325.permute(0, 2, 3, 1)
        x_325 = None
        x_327 = torch.nn.functional.layer_norm(
            x_326,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_326 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_ = (None)
        x_328 = x_327.permute(0, 3, 1, 2)
        x_327 = None
        x_329 = torch.conv2d(
            x_328,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_328 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_37 = torch.nn.functional.relu(x_329, inplace=False)
        x_329 = None
        pow_38 = relu_37**2
        relu_37 = None
        mul_45 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_
            * pow_38
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_ = (
            pow_38
        ) = None
        x_330 = (
            mul_45
            + l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_
        )
        mul_45 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_ = (None)
        x_331 = torch.nn.functional.dropout(x_330, 0.0, False, False)
        x_330 = None
        x_332 = torch.conv2d(
            x_331,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_331 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_333 = torch.nn.functional.dropout(x_332, 0.0, False, False)
        x_332 = None
        x_334 = mul_44 + x_333
        mul_44 = x_333 = None
        view_8 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_46 = x_334 * view_8
        view_8 = None
        x_335 = x_334.permute(0, 2, 3, 1)
        x_334 = None
        x_336 = torch.nn.functional.layer_norm(
            x_335,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_335 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_ = (None)
        x_337 = x_336.permute(0, 3, 1, 2)
        x_336 = None
        x_338 = torch.conv2d(
            x_337,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_337 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_38 = torch.nn.functional.relu(x_338, inplace=False)
        x_338 = None
        pow_39 = relu_38**2
        relu_38 = None
        mul_47 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_act1_parameters_scale_
            * pow_39
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_39
        ) = None
        x_339 = (
            mul_47
            + l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_47 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_340 = torch.conv2d(
            x_339,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        x_339 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_341 = torch.conv2d(
            x_340,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_340 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_342 = mul_46 + x_341
        mul_46 = x_341 = None
        view_9 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_48 = x_342 * view_9
        view_9 = None
        x_343 = x_342.permute(0, 2, 3, 1)
        x_342 = None
        x_344 = torch.nn.functional.layer_norm(
            x_343,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_343 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_ = (None)
        x_345 = x_344.permute(0, 3, 1, 2)
        x_344 = None
        x_346 = torch.conv2d(
            x_345,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_345 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_39 = torch.nn.functional.relu(x_346, inplace=False)
        x_346 = None
        pow_40 = relu_39**2
        relu_39 = None
        mul_49 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_
            * pow_40
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_ = (
            pow_40
        ) = None
        x_347 = (
            mul_49
            + l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_
        )
        mul_49 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_ = (None)
        x_348 = torch.nn.functional.dropout(x_347, 0.0, False, False)
        x_347 = None
        x_349 = torch.conv2d(
            x_348,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_348 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_350 = torch.nn.functional.dropout(x_349, 0.0, False, False)
        x_349 = None
        x_351 = mul_48 + x_350
        mul_48 = x_350 = None
        view_10 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_50 = x_351 * view_10
        view_10 = None
        x_352 = x_351.permute(0, 2, 3, 1)
        x_351 = None
        x_353 = torch.nn.functional.layer_norm(
            x_352,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_352 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_ = (None)
        x_354 = x_353.permute(0, 3, 1, 2)
        x_353 = None
        x_355 = torch.conv2d(
            x_354,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_354 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_40 = torch.nn.functional.relu(x_355, inplace=False)
        x_355 = None
        pow_41 = relu_40**2
        relu_40 = None
        mul_51 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_act1_parameters_scale_
            * pow_41
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_41
        ) = None
        x_356 = (
            mul_51
            + l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_51 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_357 = torch.conv2d(
            x_356,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        x_356 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_358 = torch.conv2d(
            x_357,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_357 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_359 = mul_50 + x_358
        mul_50 = x_358 = None
        view_11 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_52 = x_359 * view_11
        view_11 = None
        x_360 = x_359.permute(0, 2, 3, 1)
        x_359 = None
        x_361 = torch.nn.functional.layer_norm(
            x_360,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_360 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_ = (None)
        x_362 = x_361.permute(0, 3, 1, 2)
        x_361 = None
        x_363 = torch.conv2d(
            x_362,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_362 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_41 = torch.nn.functional.relu(x_363, inplace=False)
        x_363 = None
        pow_42 = relu_41**2
        relu_41 = None
        mul_53 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_
            * pow_42
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_ = (
            pow_42
        ) = None
        x_364 = (
            mul_53
            + l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_
        )
        mul_53 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_ = (None)
        x_365 = torch.nn.functional.dropout(x_364, 0.0, False, False)
        x_364 = None
        x_366 = torch.conv2d(
            x_365,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_365 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_367 = torch.nn.functional.dropout(x_366, 0.0, False, False)
        x_366 = None
        x_368 = mul_52 + x_367
        mul_52 = x_367 = None
        view_12 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_54 = x_368 * view_12
        view_12 = None
        x_369 = x_368.permute(0, 2, 3, 1)
        x_368 = None
        x_370 = torch.nn.functional.layer_norm(
            x_369,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_369 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_ = (None)
        x_371 = x_370.permute(0, 3, 1, 2)
        x_370 = None
        x_372 = torch.conv2d(
            x_371,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_371 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_42 = torch.nn.functional.relu(x_372, inplace=False)
        x_372 = None
        pow_43 = relu_42**2
        relu_42 = None
        mul_55 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_act1_parameters_scale_
            * pow_43
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_43
        ) = None
        x_373 = (
            mul_55
            + l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_55 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_374 = torch.conv2d(
            x_373,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        x_373 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_375 = torch.conv2d(
            x_374,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_374 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_376 = mul_54 + x_375
        mul_54 = x_375 = None
        view_13 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_56 = x_376 * view_13
        view_13 = None
        x_377 = x_376.permute(0, 2, 3, 1)
        x_376 = None
        x_378 = torch.nn.functional.layer_norm(
            x_377,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_377 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_ = (None)
        x_379 = x_378.permute(0, 3, 1, 2)
        x_378 = None
        x_380 = torch.conv2d(
            x_379,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_379 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_43 = torch.nn.functional.relu(x_380, inplace=False)
        x_380 = None
        pow_44 = relu_43**2
        relu_43 = None
        mul_57 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_scale_
            * pow_44
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_scale_ = (
            pow_44
        ) = None
        x_381 = (
            mul_57
            + l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_bias_
        )
        mul_57 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_bias_ = (None)
        x_382 = torch.nn.functional.dropout(x_381, 0.0, False, False)
        x_381 = None
        x_383 = torch.conv2d(
            x_382,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_382 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_384 = torch.nn.functional.dropout(x_383, 0.0, False, False)
        x_383 = None
        x_385 = mul_56 + x_384
        mul_56 = x_384 = None
        view_14 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_58 = x_385 * view_14
        view_14 = None
        x_386 = x_385.permute(0, 2, 3, 1)
        x_385 = None
        x_387 = torch.nn.functional.layer_norm(
            x_386,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_386 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_ = (None)
        x_388 = x_387.permute(0, 3, 1, 2)
        x_387 = None
        x_389 = torch.conv2d(
            x_388,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_388 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_44 = torch.nn.functional.relu(x_389, inplace=False)
        x_389 = None
        pow_45 = relu_44**2
        relu_44 = None
        mul_59 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_act1_parameters_scale_
            * pow_45
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_45
        ) = None
        x_390 = (
            mul_59
            + l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_59 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_391 = torch.conv2d(
            x_390,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        x_390 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_392 = torch.conv2d(
            x_391,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_391 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_393 = mul_58 + x_392
        mul_58 = x_392 = None
        view_15 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_60 = x_393 * view_15
        view_15 = None
        x_394 = x_393.permute(0, 2, 3, 1)
        x_393 = None
        x_395 = torch.nn.functional.layer_norm(
            x_394,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_394 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_ = (None)
        x_396 = x_395.permute(0, 3, 1, 2)
        x_395 = None
        x_397 = torch.conv2d(
            x_396,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_396 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_45 = torch.nn.functional.relu(x_397, inplace=False)
        x_397 = None
        pow_46 = relu_45**2
        relu_45 = None
        mul_61 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_scale_
            * pow_46
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_scale_ = (
            pow_46
        ) = None
        x_398 = (
            mul_61
            + l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_bias_
        )
        mul_61 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_bias_ = (None)
        x_399 = torch.nn.functional.dropout(x_398, 0.0, False, False)
        x_398 = None
        x_400 = torch.conv2d(
            x_399,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_399 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_401 = torch.nn.functional.dropout(x_400, 0.0, False, False)
        x_400 = None
        x_402 = mul_60 + x_401
        mul_60 = x_401 = None
        view_16 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_62 = x_402 * view_16
        view_16 = None
        x_403 = x_402.permute(0, 2, 3, 1)
        x_402 = None
        x_404 = torch.nn.functional.layer_norm(
            x_403,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_403 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_ = (None)
        x_405 = x_404.permute(0, 3, 1, 2)
        x_404 = None
        x_406 = torch.conv2d(
            x_405,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_405 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_46 = torch.nn.functional.relu(x_406, inplace=False)
        x_406 = None
        pow_47 = relu_46**2
        relu_46 = None
        mul_63 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_act1_parameters_scale_
            * pow_47
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_47
        ) = None
        x_407 = (
            mul_63
            + l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_63 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_408 = torch.conv2d(
            x_407,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        x_407 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_409 = torch.conv2d(
            x_408,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_408 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_410 = mul_62 + x_409
        mul_62 = x_409 = None
        view_17 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_64 = x_410 * view_17
        view_17 = None
        x_411 = x_410.permute(0, 2, 3, 1)
        x_410 = None
        x_412 = torch.nn.functional.layer_norm(
            x_411,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_411 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_ = (None)
        x_413 = x_412.permute(0, 3, 1, 2)
        x_412 = None
        x_414 = torch.conv2d(
            x_413,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_413 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_47 = torch.nn.functional.relu(x_414, inplace=False)
        x_414 = None
        pow_48 = relu_47**2
        relu_47 = None
        mul_65 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_scale_
            * pow_48
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_scale_ = (
            pow_48
        ) = None
        x_415 = (
            mul_65
            + l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_bias_
        )
        mul_65 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_bias_ = (None)
        x_416 = torch.nn.functional.dropout(x_415, 0.0, False, False)
        x_415 = None
        x_417 = torch.conv2d(
            x_416,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_416 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_418 = torch.nn.functional.dropout(x_417, 0.0, False, False)
        x_417 = None
        x_419 = mul_64 + x_418
        mul_64 = x_418 = None
        view_18 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_66 = x_419 * view_18
        view_18 = None
        x_420 = x_419.permute(0, 2, 3, 1)
        x_419 = None
        x_421 = torch.nn.functional.layer_norm(
            x_420,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_420 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_ = (None)
        x_422 = x_421.permute(0, 3, 1, 2)
        x_421 = None
        x_423 = torch.conv2d(
            x_422,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_422 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_48 = torch.nn.functional.relu(x_423, inplace=False)
        x_423 = None
        pow_49 = relu_48**2
        relu_48 = None
        mul_67 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_act1_parameters_scale_
            * pow_49
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_49
        ) = None
        x_424 = (
            mul_67
            + l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_67 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_425 = torch.conv2d(
            x_424,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        x_424 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_426 = torch.conv2d(
            x_425,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_425 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_427 = mul_66 + x_426
        mul_66 = x_426 = None
        view_19 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_68 = x_427 * view_19
        view_19 = None
        x_428 = x_427.permute(0, 2, 3, 1)
        x_427 = None
        x_429 = torch.nn.functional.layer_norm(
            x_428,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_428 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_ = (None)
        x_430 = x_429.permute(0, 3, 1, 2)
        x_429 = None
        x_431 = torch.conv2d(
            x_430,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_430 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_49 = torch.nn.functional.relu(x_431, inplace=False)
        x_431 = None
        pow_50 = relu_49**2
        relu_49 = None
        mul_69 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_scale_
            * pow_50
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_scale_ = (
            pow_50
        ) = None
        x_432 = (
            mul_69
            + l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_bias_
        )
        mul_69 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_bias_ = (None)
        x_433 = torch.nn.functional.dropout(x_432, 0.0, False, False)
        x_432 = None
        x_434 = torch.conv2d(
            x_433,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_433 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_435 = torch.nn.functional.dropout(x_434, 0.0, False, False)
        x_434 = None
        x_436 = mul_68 + x_435
        mul_68 = x_435 = None
        view_20 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_70 = x_436 * view_20
        view_20 = None
        x_437 = x_436.permute(0, 2, 3, 1)
        x_436 = None
        x_438 = torch.nn.functional.layer_norm(
            x_437,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_437 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_ = (None)
        x_439 = x_438.permute(0, 3, 1, 2)
        x_438 = None
        x_440 = torch.conv2d(
            x_439,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_439 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_50 = torch.nn.functional.relu(x_440, inplace=False)
        x_440 = None
        pow_51 = relu_50**2
        relu_50 = None
        mul_71 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_act1_parameters_scale_
            * pow_51
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_51
        ) = None
        x_441 = (
            mul_71
            + l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_71 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_442 = torch.conv2d(
            x_441,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        x_441 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_443 = torch.conv2d(
            x_442,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_442 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_444 = mul_70 + x_443
        mul_70 = x_443 = None
        view_21 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_72 = x_444 * view_21
        view_21 = None
        x_445 = x_444.permute(0, 2, 3, 1)
        x_444 = None
        x_446 = torch.nn.functional.layer_norm(
            x_445,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_445 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_ = (None)
        x_447 = x_446.permute(0, 3, 1, 2)
        x_446 = None
        x_448 = torch.conv2d(
            x_447,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_447 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_51 = torch.nn.functional.relu(x_448, inplace=False)
        x_448 = None
        pow_52 = relu_51**2
        relu_51 = None
        mul_73 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_scale_
            * pow_52
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_scale_ = (
            pow_52
        ) = None
        x_449 = (
            mul_73
            + l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_bias_
        )
        mul_73 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_bias_ = (None)
        x_450 = torch.nn.functional.dropout(x_449, 0.0, False, False)
        x_449 = None
        x_451 = torch.conv2d(
            x_450,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_450 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_452 = torch.nn.functional.dropout(x_451, 0.0, False, False)
        x_451 = None
        x_453 = mul_72 + x_452
        mul_72 = x_452 = None
        view_22 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_74 = x_453 * view_22
        view_22 = None
        x_454 = x_453.permute(0, 2, 3, 1)
        x_453 = None
        x_455 = torch.nn.functional.layer_norm(
            x_454,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_454 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_ = (None)
        x_456 = x_455.permute(0, 3, 1, 2)
        x_455 = None
        x_457 = torch.conv2d(
            x_456,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_456 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_52 = torch.nn.functional.relu(x_457, inplace=False)
        x_457 = None
        pow_53 = relu_52**2
        relu_52 = None
        mul_75 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_act1_parameters_scale_
            * pow_53
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_53
        ) = None
        x_458 = (
            mul_75
            + l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_75 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_459 = torch.conv2d(
            x_458,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        x_458 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_460 = torch.conv2d(
            x_459,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_459 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_461 = mul_74 + x_460
        mul_74 = x_460 = None
        view_23 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_76 = x_461 * view_23
        view_23 = None
        x_462 = x_461.permute(0, 2, 3, 1)
        x_461 = None
        x_463 = torch.nn.functional.layer_norm(
            x_462,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_462 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_ = (None)
        x_464 = x_463.permute(0, 3, 1, 2)
        x_463 = None
        x_465 = torch.conv2d(
            x_464,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_464 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_53 = torch.nn.functional.relu(x_465, inplace=False)
        x_465 = None
        pow_54 = relu_53**2
        relu_53 = None
        mul_77 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_scale_
            * pow_54
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_scale_ = (
            pow_54
        ) = None
        x_466 = (
            mul_77
            + l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_bias_
        )
        mul_77 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_bias_ = (None)
        x_467 = torch.nn.functional.dropout(x_466, 0.0, False, False)
        x_466 = None
        x_468 = torch.conv2d(
            x_467,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_467 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_469 = torch.nn.functional.dropout(x_468, 0.0, False, False)
        x_468 = None
        x_470 = mul_76 + x_469
        mul_76 = x_469 = None
        view_24 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_78 = x_470 * view_24
        view_24 = None
        x_471 = x_470.permute(0, 2, 3, 1)
        x_470 = None
        x_472 = torch.nn.functional.layer_norm(
            x_471,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_471 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_ = (None)
        x_473 = x_472.permute(0, 3, 1, 2)
        x_472 = None
        x_474 = torch.conv2d(
            x_473,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_473 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_54 = torch.nn.functional.relu(x_474, inplace=False)
        x_474 = None
        pow_55 = relu_54**2
        relu_54 = None
        mul_79 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_act1_parameters_scale_
            * pow_55
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_55
        ) = None
        x_475 = (
            mul_79
            + l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_79 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_476 = torch.conv2d(
            x_475,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        x_475 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_477 = torch.conv2d(
            x_476,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_476 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_478 = mul_78 + x_477
        mul_78 = x_477 = None
        view_25 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_80 = x_478 * view_25
        view_25 = None
        x_479 = x_478.permute(0, 2, 3, 1)
        x_478 = None
        x_480 = torch.nn.functional.layer_norm(
            x_479,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_479 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_ = (None)
        x_481 = x_480.permute(0, 3, 1, 2)
        x_480 = None
        x_482 = torch.conv2d(
            x_481,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_481 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_55 = torch.nn.functional.relu(x_482, inplace=False)
        x_482 = None
        pow_56 = relu_55**2
        relu_55 = None
        mul_81 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_scale_
            * pow_56
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_scale_ = (
            pow_56
        ) = None
        x_483 = (
            mul_81
            + l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_bias_
        )
        mul_81 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_bias_ = (None)
        x_484 = torch.nn.functional.dropout(x_483, 0.0, False, False)
        x_483 = None
        x_485 = torch.conv2d(
            x_484,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_484 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_486 = torch.nn.functional.dropout(x_485, 0.0, False, False)
        x_485 = None
        x_487 = mul_80 + x_486
        mul_80 = x_486 = None
        view_26 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_82 = x_487 * view_26
        view_26 = None
        x_488 = x_487.permute(0, 2, 3, 1)
        x_487 = None
        x_489 = torch.nn.functional.layer_norm(
            x_488,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_488 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_ = (None)
        x_490 = x_489.permute(0, 3, 1, 2)
        x_489 = None
        x_491 = torch.conv2d(
            x_490,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_490 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_56 = torch.nn.functional.relu(x_491, inplace=False)
        x_491 = None
        pow_57 = relu_56**2
        relu_56 = None
        mul_83 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_act1_parameters_scale_
            * pow_57
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_57
        ) = None
        x_492 = (
            mul_83
            + l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_83 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_493 = torch.conv2d(
            x_492,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        x_492 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_494 = torch.conv2d(
            x_493,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_493 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_495 = mul_82 + x_494
        mul_82 = x_494 = None
        view_27 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_84 = x_495 * view_27
        view_27 = None
        x_496 = x_495.permute(0, 2, 3, 1)
        x_495 = None
        x_497 = torch.nn.functional.layer_norm(
            x_496,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_496 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_ = (None)
        x_498 = x_497.permute(0, 3, 1, 2)
        x_497 = None
        x_499 = torch.conv2d(
            x_498,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_498 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_57 = torch.nn.functional.relu(x_499, inplace=False)
        x_499 = None
        pow_58 = relu_57**2
        relu_57 = None
        mul_85 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_scale_
            * pow_58
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_scale_ = (
            pow_58
        ) = None
        x_500 = (
            mul_85
            + l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_bias_
        )
        mul_85 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_bias_ = (None)
        x_501 = torch.nn.functional.dropout(x_500, 0.0, False, False)
        x_500 = None
        x_502 = torch.conv2d(
            x_501,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_501 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_503 = torch.nn.functional.dropout(x_502, 0.0, False, False)
        x_502 = None
        x_504 = mul_84 + x_503
        mul_84 = x_503 = None
        view_28 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_86 = x_504 * view_28
        view_28 = None
        x_505 = x_504.permute(0, 2, 3, 1)
        x_504 = None
        x_506 = torch.nn.functional.layer_norm(
            x_505,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_505 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_ = (None)
        x_507 = x_506.permute(0, 3, 1, 2)
        x_506 = None
        x_508 = torch.conv2d(
            x_507,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_507 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_58 = torch.nn.functional.relu(x_508, inplace=False)
        x_508 = None
        pow_59 = relu_58**2
        relu_58 = None
        mul_87 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_act1_parameters_scale_
            * pow_59
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_59
        ) = None
        x_509 = (
            mul_87
            + l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_87 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_510 = torch.conv2d(
            x_509,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        x_509 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_511 = torch.conv2d(
            x_510,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_510 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_512 = mul_86 + x_511
        mul_86 = x_511 = None
        view_29 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_88 = x_512 * view_29
        view_29 = None
        x_513 = x_512.permute(0, 2, 3, 1)
        x_512 = None
        x_514 = torch.nn.functional.layer_norm(
            x_513,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_513 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_ = (None)
        x_515 = x_514.permute(0, 3, 1, 2)
        x_514 = None
        x_516 = torch.conv2d(
            x_515,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_515 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_59 = torch.nn.functional.relu(x_516, inplace=False)
        x_516 = None
        pow_60 = relu_59**2
        relu_59 = None
        mul_89 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_scale_
            * pow_60
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_scale_ = (
            pow_60
        ) = None
        x_517 = (
            mul_89
            + l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_bias_
        )
        mul_89 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_bias_ = (None)
        x_518 = torch.nn.functional.dropout(x_517, 0.0, False, False)
        x_517 = None
        x_519 = torch.conv2d(
            x_518,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_518 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_520 = torch.nn.functional.dropout(x_519, 0.0, False, False)
        x_519 = None
        x_521 = mul_88 + x_520
        mul_88 = x_520 = None
        view_30 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_90 = x_521 * view_30
        view_30 = None
        x_522 = x_521.permute(0, 2, 3, 1)
        x_521 = None
        x_523 = torch.nn.functional.layer_norm(
            x_522,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_522 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_ = (None)
        x_524 = x_523.permute(0, 3, 1, 2)
        x_523 = None
        x_525 = torch.conv2d(
            x_524,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_524 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_60 = torch.nn.functional.relu(x_525, inplace=False)
        x_525 = None
        pow_61 = relu_60**2
        relu_60 = None
        mul_91 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_act1_parameters_scale_
            * pow_61
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_61
        ) = None
        x_526 = (
            mul_91
            + l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_91 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_527 = torch.conv2d(
            x_526,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        x_526 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_528 = torch.conv2d(
            x_527,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_527 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_529 = mul_90 + x_528
        mul_90 = x_528 = None
        view_31 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_92 = x_529 * view_31
        view_31 = None
        x_530 = x_529.permute(0, 2, 3, 1)
        x_529 = None
        x_531 = torch.nn.functional.layer_norm(
            x_530,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_530 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_ = (None)
        x_532 = x_531.permute(0, 3, 1, 2)
        x_531 = None
        x_533 = torch.conv2d(
            x_532,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_532 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_61 = torch.nn.functional.relu(x_533, inplace=False)
        x_533 = None
        pow_62 = relu_61**2
        relu_61 = None
        mul_93 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_scale_
            * pow_62
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_scale_ = (
            pow_62
        ) = None
        x_534 = (
            mul_93
            + l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_bias_
        )
        mul_93 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_bias_ = (None)
        x_535 = torch.nn.functional.dropout(x_534, 0.0, False, False)
        x_534 = None
        x_536 = torch.conv2d(
            x_535,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_535 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_537 = torch.nn.functional.dropout(x_536, 0.0, False, False)
        x_536 = None
        x_538 = mul_92 + x_537
        mul_92 = x_537 = None
        view_32 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_94 = x_538 * view_32
        view_32 = None
        x_539 = x_538.permute(0, 2, 3, 1)
        x_538 = None
        x_540 = torch.nn.functional.layer_norm(
            x_539,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_539 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_ = (None)
        x_541 = x_540.permute(0, 3, 1, 2)
        x_540 = None
        x_542 = torch.conv2d(
            x_541,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_541 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_62 = torch.nn.functional.relu(x_542, inplace=False)
        x_542 = None
        pow_63 = relu_62**2
        relu_62 = None
        mul_95 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_act1_parameters_scale_
            * pow_63
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_63
        ) = None
        x_543 = (
            mul_95
            + l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_95 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_544 = torch.conv2d(
            x_543,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        x_543 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_545 = torch.conv2d(
            x_544,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_544 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_546 = mul_94 + x_545
        mul_94 = x_545 = None
        view_33 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_96 = x_546 * view_33
        view_33 = None
        x_547 = x_546.permute(0, 2, 3, 1)
        x_546 = None
        x_548 = torch.nn.functional.layer_norm(
            x_547,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_547 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_ = (None)
        x_549 = x_548.permute(0, 3, 1, 2)
        x_548 = None
        x_550 = torch.conv2d(
            x_549,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_549 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_63 = torch.nn.functional.relu(x_550, inplace=False)
        x_550 = None
        pow_64 = relu_63**2
        relu_63 = None
        mul_97 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_scale_
            * pow_64
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_scale_ = (
            pow_64
        ) = None
        x_551 = (
            mul_97
            + l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_bias_
        )
        mul_97 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_bias_ = (None)
        x_552 = torch.nn.functional.dropout(x_551, 0.0, False, False)
        x_551 = None
        x_553 = torch.conv2d(
            x_552,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_552 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_554 = torch.nn.functional.dropout(x_553, 0.0, False, False)
        x_553 = None
        x_555 = mul_96 + x_554
        mul_96 = x_554 = None
        view_34 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_res_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_98 = x_555 * view_34
        view_34 = None
        x_556 = x_555.permute(0, 2, 3, 1)
        x_555 = None
        x_557 = torch.nn.functional.layer_norm(
            x_556,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_556 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_ = (None)
        x_558 = x_557.permute(0, 3, 1, 2)
        x_557 = None
        x_559 = torch.conv2d(
            x_558,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_558 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_64 = torch.nn.functional.relu(x_559, inplace=False)
        x_559 = None
        pow_65 = relu_64**2
        relu_64 = None
        mul_99 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_act1_parameters_scale_
            * pow_65
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_65
        ) = None
        x_560 = (
            mul_99
            + l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_99 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_561 = torch.conv2d(
            x_560,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        x_560 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_562 = torch.conv2d(
            x_561,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_561 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_563 = mul_98 + x_562
        mul_98 = x_562 = None
        view_35 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_res_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_100 = x_563 * view_35
        view_35 = None
        x_564 = x_563.permute(0, 2, 3, 1)
        x_563 = None
        x_565 = torch.nn.functional.layer_norm(
            x_564,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_564 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_ = (None)
        x_566 = x_565.permute(0, 3, 1, 2)
        x_565 = None
        x_567 = torch.conv2d(
            x_566,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_566 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_65 = torch.nn.functional.relu(x_567, inplace=False)
        x_567 = None
        pow_66 = relu_65**2
        relu_65 = None
        mul_101 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_act_parameters_scale_
            * pow_66
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_act_parameters_scale_ = (
            pow_66
        ) = None
        x_568 = (
            mul_101
            + l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_act_parameters_bias_
        )
        mul_101 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_act_parameters_bias_ = (None)
        x_569 = torch.nn.functional.dropout(x_568, 0.0, False, False)
        x_568 = None
        x_570 = torch.conv2d(
            x_569,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_569 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_571 = torch.nn.functional.dropout(x_570, 0.0, False, False)
        x_570 = None
        x_572 = mul_100 + x_571
        mul_100 = x_571 = None
        x_573 = x_572.permute(0, 2, 3, 1)
        x_572 = None
        x_574 = torch.nn.functional.layer_norm(
            x_573,
            (384,),
            l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_,
            None,
            1e-06,
        )
        x_573 = l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_ = (None)
        x_575 = x_574.permute(0, 3, 1, 2)
        x_574 = None
        x_576 = torch.conv2d(
            x_575,
            l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_575 = l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_ = (None)
        view_36 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale1_parameters_scale_.view(
            (576, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_102 = x_576 * view_36
        view_36 = None
        x_577 = x_576.permute(0, 2, 3, 1)
        x_576 = None
        x_578 = torch.nn.functional.layer_norm(
            x_577,
            (576,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_577 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (None)
        x_579 = x_578.permute(0, 3, 1, 2)
        x_578 = None
        x_580 = torch.conv2d(
            x_579,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_579 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_66 = torch.nn.functional.relu(x_580, inplace=False)
        x_580 = None
        pow_67 = relu_66**2
        relu_66 = None
        mul_103 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_scale_
            * pow_67
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_67
        ) = None
        x_581 = (
            mul_103
            + l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_103 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_582 = torch.conv2d(
            x_581,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            1152,
        )
        x_581 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_583 = torch.conv2d(
            x_582,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_582 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_584 = mul_102 + x_583
        mul_102 = x_583 = None
        view_37 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale2_parameters_scale_.view(
            (576, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_104 = x_584 * view_37
        view_37 = None
        x_585 = x_584.permute(0, 2, 3, 1)
        x_584 = None
        x_586 = torch.nn.functional.layer_norm(
            x_585,
            (576,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_585 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (None)
        x_587 = x_586.permute(0, 3, 1, 2)
        x_586 = None
        x_588 = torch.conv2d(
            x_587,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_587 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_67 = torch.nn.functional.relu(x_588, inplace=False)
        x_588 = None
        pow_68 = relu_67**2
        relu_67 = None
        mul_105 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
            * pow_68
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = (
            pow_68
        ) = None
        x_589 = (
            mul_105
            + l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        )
        mul_105 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = (None)
        x_590 = torch.nn.functional.dropout(x_589, 0.0, False, False)
        x_589 = None
        x_591 = torch.conv2d(
            x_590,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_590 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_592 = torch.nn.functional.dropout(x_591, 0.0, False, False)
        x_591 = None
        x_593 = mul_104 + x_592
        mul_104 = x_592 = None
        view_38 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale1_parameters_scale_.view(
            (576, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_106 = x_593 * view_38
        view_38 = None
        x_594 = x_593.permute(0, 2, 3, 1)
        x_593 = None
        x_595 = torch.nn.functional.layer_norm(
            x_594,
            (576,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_594 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (None)
        x_596 = x_595.permute(0, 3, 1, 2)
        x_595 = None
        x_597 = torch.conv2d(
            x_596,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_596 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_68 = torch.nn.functional.relu(x_597, inplace=False)
        x_597 = None
        pow_69 = relu_68**2
        relu_68 = None
        mul_107 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_scale_
            * pow_69
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_69
        ) = None
        x_598 = (
            mul_107
            + l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_107 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_599 = torch.conv2d(
            x_598,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            1152,
        )
        x_598 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_600 = torch.conv2d(
            x_599,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_599 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_601 = mul_106 + x_600
        mul_106 = x_600 = None
        view_39 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale2_parameters_scale_.view(
            (576, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_108 = x_601 * view_39
        view_39 = None
        x_602 = x_601.permute(0, 2, 3, 1)
        x_601 = None
        x_603 = torch.nn.functional.layer_norm(
            x_602,
            (576,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_602 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (None)
        x_604 = x_603.permute(0, 3, 1, 2)
        x_603 = None
        x_605 = torch.conv2d(
            x_604,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_604 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_69 = torch.nn.functional.relu(x_605, inplace=False)
        x_605 = None
        pow_70 = relu_69**2
        relu_69 = None
        mul_109 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
            * pow_70
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = (
            pow_70
        ) = None
        x_606 = (
            mul_109
            + l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        )
        mul_109 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = (None)
        x_607 = torch.nn.functional.dropout(x_606, 0.0, False, False)
        x_606 = None
        x_608 = torch.conv2d(
            x_607,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_607 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_609 = torch.nn.functional.dropout(x_608, 0.0, False, False)
        x_608 = None
        x_610 = mul_108 + x_609
        mul_108 = x_609 = None
        view_40 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale1_parameters_scale_.view(
            (576, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_110 = x_610 * view_40
        view_40 = None
        x_611 = x_610.permute(0, 2, 3, 1)
        x_610 = None
        x_612 = torch.nn.functional.layer_norm(
            x_611,
            (576,),
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_611 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (None)
        x_613 = x_612.permute(0, 3, 1, 2)
        x_612 = None
        x_614 = torch.conv2d(
            x_613,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_613 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_70 = torch.nn.functional.relu(x_614, inplace=False)
        x_614 = None
        pow_71 = relu_70**2
        relu_70 = None
        mul_111 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_scale_
            * pow_71
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_71
        ) = None
        x_615 = (
            mul_111
            + l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_111 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_616 = torch.conv2d(
            x_615,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            1152,
        )
        x_615 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_617 = torch.conv2d(
            x_616,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_616 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_618 = mul_110 + x_617
        mul_110 = x_617 = None
        view_41 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale2_parameters_scale_.view(
            (576, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_112 = x_618 * view_41
        view_41 = None
        x_619 = x_618.permute(0, 2, 3, 1)
        x_618 = None
        x_620 = torch.nn.functional.layer_norm(
            x_619,
            (576,),
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_619 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (None)
        x_621 = x_620.permute(0, 3, 1, 2)
        x_620 = None
        x_622 = torch.conv2d(
            x_621,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_621 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_71 = torch.nn.functional.relu(x_622, inplace=False)
        x_622 = None
        pow_72 = relu_71**2
        relu_71 = None
        mul_113 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_
            * pow_72
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_ = (
            pow_72
        ) = None
        x_623 = (
            mul_113
            + l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_
        )
        mul_113 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_ = (None)
        x_624 = torch.nn.functional.dropout(x_623, 0.0, False, False)
        x_623 = None
        x_625 = torch.conv2d(
            x_624,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_624 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_626 = torch.nn.functional.dropout(x_625, 0.0, False, False)
        x_625 = None
        x_627 = mul_112 + x_626
        mul_112 = x_626 = None
        x_628 = torch.nn.functional.adaptive_avg_pool2d(x_627, 1)
        x_627 = None
        x_629 = x_628.permute(0, 2, 3, 1)
        x_628 = None
        x_630 = torch.nn.functional.layer_norm(
            x_629,
            (576,),
            l_self_modules_head_modules_norm_parameters_weight_,
            l_self_modules_head_modules_norm_parameters_bias_,
            1e-06,
        )
        x_629 = (
            l_self_modules_head_modules_norm_parameters_weight_
        ) = l_self_modules_head_modules_norm_parameters_bias_ = None
        x_631 = x_630.permute(0, 3, 1, 2)
        x_630 = None
        x_632 = x_631.flatten(1, -1)
        x_631 = None
        x_633 = torch.nn.functional.dropout(x_632, 0.0, False, False)
        x_632 = None
        x_634 = torch._C._nn.linear(
            x_633,
            l_self_modules_head_modules_fc_modules_fc1_parameters_weight_,
            l_self_modules_head_modules_fc_modules_fc1_parameters_bias_,
        )
        x_633 = (
            l_self_modules_head_modules_fc_modules_fc1_parameters_weight_
        ) = l_self_modules_head_modules_fc_modules_fc1_parameters_bias_ = None
        relu_72 = torch.nn.functional.relu(x_634, inplace=False)
        x_634 = None
        x_635 = torch.square(relu_72)
        relu_72 = None
        x_636 = torch.nn.functional.layer_norm(
            x_635,
            (2304,),
            l_self_modules_head_modules_fc_modules_norm_parameters_weight_,
            l_self_modules_head_modules_fc_modules_norm_parameters_bias_,
            1e-06,
        )
        x_635 = (
            l_self_modules_head_modules_fc_modules_norm_parameters_weight_
        ) = l_self_modules_head_modules_fc_modules_norm_parameters_bias_ = None
        x_637 = torch.nn.functional.dropout(x_636, 0.0, False, False)
        x_636 = None
        x_638 = torch._C._nn.linear(
            x_637,
            l_self_modules_head_modules_fc_modules_fc2_parameters_weight_,
            l_self_modules_head_modules_fc_modules_fc2_parameters_bias_,
        )
        x_637 = (
            l_self_modules_head_modules_fc_modules_fc2_parameters_weight_
        ) = l_self_modules_head_modules_fc_modules_fc2_parameters_bias_ = None
        return (x_638,)
