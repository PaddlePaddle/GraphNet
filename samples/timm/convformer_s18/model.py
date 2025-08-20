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
            x_1, (64,), l_self_modules_stem_modules_norm_parameters_weight_, None, 1e-06
        )
        x_1 = l_self_modules_stem_modules_norm_parameters_weight_ = None
        x_3 = x_2.permute(0, 3, 1, 2)
        x_2 = None
        x_4 = x_3.permute(0, 2, 3, 1)
        x_5 = torch.nn.functional.layer_norm(
            x_4,
            (64,),
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
            128,
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
            (64,),
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
            (64,),
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
            128,
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
            (64,),
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
            (64,),
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
            128,
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
            (64,),
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
            (64,),
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
            (128,),
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
            256,
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
            (128,),
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
            (128,),
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
            256,
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
            (128,),
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
            (128,),
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
            256,
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
            (128,),
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
        x_109 = None
        x_111 = torch.nn.functional.layer_norm(
            x_110,
            (128,),
            l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_,
            None,
            1e-06,
        )
        x_110 = l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_ = (None)
        x_112 = x_111.permute(0, 3, 1, 2)
        x_111 = None
        x_113 = torch.conv2d(
            x_112,
            l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_112 = l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_ = (None)
        view = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale1_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_12 = x_113 * view
        view = None
        x_114 = x_113.permute(0, 2, 3, 1)
        x_113 = None
        x_115 = torch.nn.functional.layer_norm(
            x_114,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_114 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (None)
        x_116 = x_115.permute(0, 3, 1, 2)
        x_115 = None
        x_117 = torch.conv2d(
            x_116,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_116 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_12 = torch.nn.functional.relu(x_117, inplace=False)
        x_117 = None
        pow_13 = relu_12**2
        relu_12 = None
        mul_13 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_scale_
            * pow_13
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_13
        ) = None
        x_118 = (
            mul_13
            + l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_13 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_119 = torch.conv2d(
            x_118,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            640,
        )
        x_118 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_120 = torch.conv2d(
            x_119,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_119 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_121 = mul_12 + x_120
        mul_12 = x_120 = None
        view_1 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale2_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_14 = x_121 * view_1
        view_1 = None
        x_122 = x_121.permute(0, 2, 3, 1)
        x_121 = None
        x_123 = torch.nn.functional.layer_norm(
            x_122,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_122 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (None)
        x_124 = x_123.permute(0, 3, 1, 2)
        x_123 = None
        x_125 = torch.conv2d(
            x_124,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_124 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_13 = torch.nn.functional.relu(x_125, inplace=False)
        x_125 = None
        pow_14 = relu_13**2
        relu_13 = None
        mul_15 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
            * pow_14
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = (
            pow_14
        ) = None
        x_126 = (
            mul_15
            + l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        )
        mul_15 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = (None)
        x_127 = torch.nn.functional.dropout(x_126, 0.0, False, False)
        x_126 = None
        x_128 = torch.conv2d(
            x_127,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_127 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_129 = torch.nn.functional.dropout(x_128, 0.0, False, False)
        x_128 = None
        x_130 = mul_14 + x_129
        mul_14 = x_129 = None
        view_2 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale1_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_16 = x_130 * view_2
        view_2 = None
        x_131 = x_130.permute(0, 2, 3, 1)
        x_130 = None
        x_132 = torch.nn.functional.layer_norm(
            x_131,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_131 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (None)
        x_133 = x_132.permute(0, 3, 1, 2)
        x_132 = None
        x_134 = torch.conv2d(
            x_133,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_133 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_14 = torch.nn.functional.relu(x_134, inplace=False)
        x_134 = None
        pow_15 = relu_14**2
        relu_14 = None
        mul_17 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_scale_
            * pow_15
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_15
        ) = None
        x_135 = (
            mul_17
            + l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_17 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_136 = torch.conv2d(
            x_135,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            640,
        )
        x_135 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_137 = torch.conv2d(
            x_136,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_136 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_138 = mul_16 + x_137
        mul_16 = x_137 = None
        view_3 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale2_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_18 = x_138 * view_3
        view_3 = None
        x_139 = x_138.permute(0, 2, 3, 1)
        x_138 = None
        x_140 = torch.nn.functional.layer_norm(
            x_139,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_139 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (None)
        x_141 = x_140.permute(0, 3, 1, 2)
        x_140 = None
        x_142 = torch.conv2d(
            x_141,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_141 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_15 = torch.nn.functional.relu(x_142, inplace=False)
        x_142 = None
        pow_16 = relu_15**2
        relu_15 = None
        mul_19 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
            * pow_16
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = (
            pow_16
        ) = None
        x_143 = (
            mul_19
            + l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        )
        mul_19 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = (None)
        x_144 = torch.nn.functional.dropout(x_143, 0.0, False, False)
        x_143 = None
        x_145 = torch.conv2d(
            x_144,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_144 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_146 = torch.nn.functional.dropout(x_145, 0.0, False, False)
        x_145 = None
        x_147 = mul_18 + x_146
        mul_18 = x_146 = None
        view_4 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale1_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_20 = x_147 * view_4
        view_4 = None
        x_148 = x_147.permute(0, 2, 3, 1)
        x_147 = None
        x_149 = torch.nn.functional.layer_norm(
            x_148,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_148 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (None)
        x_150 = x_149.permute(0, 3, 1, 2)
        x_149 = None
        x_151 = torch.conv2d(
            x_150,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_150 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_16 = torch.nn.functional.relu(x_151, inplace=False)
        x_151 = None
        pow_17 = relu_16**2
        relu_16 = None
        mul_21 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_scale_
            * pow_17
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_17
        ) = None
        x_152 = (
            mul_21
            + l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_21 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_153 = torch.conv2d(
            x_152,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            640,
        )
        x_152 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_154 = torch.conv2d(
            x_153,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_153 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_155 = mul_20 + x_154
        mul_20 = x_154 = None
        view_5 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale2_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_22 = x_155 * view_5
        view_5 = None
        x_156 = x_155.permute(0, 2, 3, 1)
        x_155 = None
        x_157 = torch.nn.functional.layer_norm(
            x_156,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_156 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (None)
        x_158 = x_157.permute(0, 3, 1, 2)
        x_157 = None
        x_159 = torch.conv2d(
            x_158,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_158 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_17 = torch.nn.functional.relu(x_159, inplace=False)
        x_159 = None
        pow_18 = relu_17**2
        relu_17 = None
        mul_23 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_
            * pow_18
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_ = (
            pow_18
        ) = None
        x_160 = (
            mul_23
            + l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_
        )
        mul_23 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_ = (None)
        x_161 = torch.nn.functional.dropout(x_160, 0.0, False, False)
        x_160 = None
        x_162 = torch.conv2d(
            x_161,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_161 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_163 = torch.nn.functional.dropout(x_162, 0.0, False, False)
        x_162 = None
        x_164 = mul_22 + x_163
        mul_22 = x_163 = None
        view_6 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale1_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_24 = x_164 * view_6
        view_6 = None
        x_165 = x_164.permute(0, 2, 3, 1)
        x_164 = None
        x_166 = torch.nn.functional.layer_norm(
            x_165,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_165 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_ = (None)
        x_167 = x_166.permute(0, 3, 1, 2)
        x_166 = None
        x_168 = torch.conv2d(
            x_167,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_167 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_18 = torch.nn.functional.relu(x_168, inplace=False)
        x_168 = None
        pow_19 = relu_18**2
        relu_18 = None
        mul_25 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_act1_parameters_scale_
            * pow_19
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_19
        ) = None
        x_169 = (
            mul_25
            + l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_25 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_170 = torch.conv2d(
            x_169,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            640,
        )
        x_169 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_171 = torch.conv2d(
            x_170,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_170 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_172 = mul_24 + x_171
        mul_24 = x_171 = None
        view_7 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale2_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_26 = x_172 * view_7
        view_7 = None
        x_173 = x_172.permute(0, 2, 3, 1)
        x_172 = None
        x_174 = torch.nn.functional.layer_norm(
            x_173,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_173 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_ = (None)
        x_175 = x_174.permute(0, 3, 1, 2)
        x_174 = None
        x_176 = torch.conv2d(
            x_175,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_175 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_19 = torch.nn.functional.relu(x_176, inplace=False)
        x_176 = None
        pow_20 = relu_19**2
        relu_19 = None
        mul_27 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_
            * pow_20
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_ = (
            pow_20
        ) = None
        x_177 = (
            mul_27
            + l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_
        )
        mul_27 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_ = (None)
        x_178 = torch.nn.functional.dropout(x_177, 0.0, False, False)
        x_177 = None
        x_179 = torch.conv2d(
            x_178,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_178 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_180 = torch.nn.functional.dropout(x_179, 0.0, False, False)
        x_179 = None
        x_181 = mul_26 + x_180
        mul_26 = x_180 = None
        view_8 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale1_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_28 = x_181 * view_8
        view_8 = None
        x_182 = x_181.permute(0, 2, 3, 1)
        x_181 = None
        x_183 = torch.nn.functional.layer_norm(
            x_182,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_182 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_ = (None)
        x_184 = x_183.permute(0, 3, 1, 2)
        x_183 = None
        x_185 = torch.conv2d(
            x_184,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_184 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_20 = torch.nn.functional.relu(x_185, inplace=False)
        x_185 = None
        pow_21 = relu_20**2
        relu_20 = None
        mul_29 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_act1_parameters_scale_
            * pow_21
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_21
        ) = None
        x_186 = (
            mul_29
            + l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_29 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_187 = torch.conv2d(
            x_186,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            640,
        )
        x_186 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_188 = torch.conv2d(
            x_187,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_187 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_189 = mul_28 + x_188
        mul_28 = x_188 = None
        view_9 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale2_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_30 = x_189 * view_9
        view_9 = None
        x_190 = x_189.permute(0, 2, 3, 1)
        x_189 = None
        x_191 = torch.nn.functional.layer_norm(
            x_190,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_190 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_ = (None)
        x_192 = x_191.permute(0, 3, 1, 2)
        x_191 = None
        x_193 = torch.conv2d(
            x_192,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_192 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_21 = torch.nn.functional.relu(x_193, inplace=False)
        x_193 = None
        pow_22 = relu_21**2
        relu_21 = None
        mul_31 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_
            * pow_22
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_ = (
            pow_22
        ) = None
        x_194 = (
            mul_31
            + l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_
        )
        mul_31 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_ = (None)
        x_195 = torch.nn.functional.dropout(x_194, 0.0, False, False)
        x_194 = None
        x_196 = torch.conv2d(
            x_195,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_195 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_197 = torch.nn.functional.dropout(x_196, 0.0, False, False)
        x_196 = None
        x_198 = mul_30 + x_197
        mul_30 = x_197 = None
        view_10 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale1_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_32 = x_198 * view_10
        view_10 = None
        x_199 = x_198.permute(0, 2, 3, 1)
        x_198 = None
        x_200 = torch.nn.functional.layer_norm(
            x_199,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_199 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_ = (None)
        x_201 = x_200.permute(0, 3, 1, 2)
        x_200 = None
        x_202 = torch.conv2d(
            x_201,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_201 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_22 = torch.nn.functional.relu(x_202, inplace=False)
        x_202 = None
        pow_23 = relu_22**2
        relu_22 = None
        mul_33 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_act1_parameters_scale_
            * pow_23
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_23
        ) = None
        x_203 = (
            mul_33
            + l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_33 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_204 = torch.conv2d(
            x_203,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            640,
        )
        x_203 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_205 = torch.conv2d(
            x_204,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_204 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_206 = mul_32 + x_205
        mul_32 = x_205 = None
        view_11 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale2_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_34 = x_206 * view_11
        view_11 = None
        x_207 = x_206.permute(0, 2, 3, 1)
        x_206 = None
        x_208 = torch.nn.functional.layer_norm(
            x_207,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_207 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_ = (None)
        x_209 = x_208.permute(0, 3, 1, 2)
        x_208 = None
        x_210 = torch.conv2d(
            x_209,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_209 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_23 = torch.nn.functional.relu(x_210, inplace=False)
        x_210 = None
        pow_24 = relu_23**2
        relu_23 = None
        mul_35 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_
            * pow_24
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_ = (
            pow_24
        ) = None
        x_211 = (
            mul_35
            + l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_
        )
        mul_35 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_ = (None)
        x_212 = torch.nn.functional.dropout(x_211, 0.0, False, False)
        x_211 = None
        x_213 = torch.conv2d(
            x_212,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_212 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_214 = torch.nn.functional.dropout(x_213, 0.0, False, False)
        x_213 = None
        x_215 = mul_34 + x_214
        mul_34 = x_214 = None
        view_12 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale1_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_36 = x_215 * view_12
        view_12 = None
        x_216 = x_215.permute(0, 2, 3, 1)
        x_215 = None
        x_217 = torch.nn.functional.layer_norm(
            x_216,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_216 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_ = (None)
        x_218 = x_217.permute(0, 3, 1, 2)
        x_217 = None
        x_219 = torch.conv2d(
            x_218,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_218 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_24 = torch.nn.functional.relu(x_219, inplace=False)
        x_219 = None
        pow_25 = relu_24**2
        relu_24 = None
        mul_37 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_act1_parameters_scale_
            * pow_25
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_25
        ) = None
        x_220 = (
            mul_37
            + l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_37 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_221 = torch.conv2d(
            x_220,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            640,
        )
        x_220 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_222 = torch.conv2d(
            x_221,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_221 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_223 = mul_36 + x_222
        mul_36 = x_222 = None
        view_13 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale2_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_38 = x_223 * view_13
        view_13 = None
        x_224 = x_223.permute(0, 2, 3, 1)
        x_223 = None
        x_225 = torch.nn.functional.layer_norm(
            x_224,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_224 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_ = (None)
        x_226 = x_225.permute(0, 3, 1, 2)
        x_225 = None
        x_227 = torch.conv2d(
            x_226,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_226 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_25 = torch.nn.functional.relu(x_227, inplace=False)
        x_227 = None
        pow_26 = relu_25**2
        relu_25 = None
        mul_39 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_scale_
            * pow_26
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_scale_ = (
            pow_26
        ) = None
        x_228 = (
            mul_39
            + l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_bias_
        )
        mul_39 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_bias_ = (None)
        x_229 = torch.nn.functional.dropout(x_228, 0.0, False, False)
        x_228 = None
        x_230 = torch.conv2d(
            x_229,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_229 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_231 = torch.nn.functional.dropout(x_230, 0.0, False, False)
        x_230 = None
        x_232 = mul_38 + x_231
        mul_38 = x_231 = None
        view_14 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale1_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_40 = x_232 * view_14
        view_14 = None
        x_233 = x_232.permute(0, 2, 3, 1)
        x_232 = None
        x_234 = torch.nn.functional.layer_norm(
            x_233,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_233 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_ = (None)
        x_235 = x_234.permute(0, 3, 1, 2)
        x_234 = None
        x_236 = torch.conv2d(
            x_235,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_235 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_26 = torch.nn.functional.relu(x_236, inplace=False)
        x_236 = None
        pow_27 = relu_26**2
        relu_26 = None
        mul_41 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_act1_parameters_scale_
            * pow_27
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_27
        ) = None
        x_237 = (
            mul_41
            + l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_41 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_238 = torch.conv2d(
            x_237,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            640,
        )
        x_237 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_239 = torch.conv2d(
            x_238,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_238 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_240 = mul_40 + x_239
        mul_40 = x_239 = None
        view_15 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale2_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_42 = x_240 * view_15
        view_15 = None
        x_241 = x_240.permute(0, 2, 3, 1)
        x_240 = None
        x_242 = torch.nn.functional.layer_norm(
            x_241,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_241 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_ = (None)
        x_243 = x_242.permute(0, 3, 1, 2)
        x_242 = None
        x_244 = torch.conv2d(
            x_243,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_243 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_27 = torch.nn.functional.relu(x_244, inplace=False)
        x_244 = None
        pow_28 = relu_27**2
        relu_27 = None
        mul_43 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_scale_
            * pow_28
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_scale_ = (
            pow_28
        ) = None
        x_245 = (
            mul_43
            + l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_bias_
        )
        mul_43 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_bias_ = (None)
        x_246 = torch.nn.functional.dropout(x_245, 0.0, False, False)
        x_245 = None
        x_247 = torch.conv2d(
            x_246,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_246 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_248 = torch.nn.functional.dropout(x_247, 0.0, False, False)
        x_247 = None
        x_249 = mul_42 + x_248
        mul_42 = x_248 = None
        view_16 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale1_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_44 = x_249 * view_16
        view_16 = None
        x_250 = x_249.permute(0, 2, 3, 1)
        x_249 = None
        x_251 = torch.nn.functional.layer_norm(
            x_250,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_250 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_ = (None)
        x_252 = x_251.permute(0, 3, 1, 2)
        x_251 = None
        x_253 = torch.conv2d(
            x_252,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_252 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_28 = torch.nn.functional.relu(x_253, inplace=False)
        x_253 = None
        pow_29 = relu_28**2
        relu_28 = None
        mul_45 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_act1_parameters_scale_
            * pow_29
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_29
        ) = None
        x_254 = (
            mul_45
            + l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_45 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_255 = torch.conv2d(
            x_254,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            640,
        )
        x_254 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_256 = torch.conv2d(
            x_255,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_255 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_257 = mul_44 + x_256
        mul_44 = x_256 = None
        view_17 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale2_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_46 = x_257 * view_17
        view_17 = None
        x_258 = x_257.permute(0, 2, 3, 1)
        x_257 = None
        x_259 = torch.nn.functional.layer_norm(
            x_258,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_258 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_ = (None)
        x_260 = x_259.permute(0, 3, 1, 2)
        x_259 = None
        x_261 = torch.conv2d(
            x_260,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_260 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_29 = torch.nn.functional.relu(x_261, inplace=False)
        x_261 = None
        pow_30 = relu_29**2
        relu_29 = None
        mul_47 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_scale_
            * pow_30
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_scale_ = (
            pow_30
        ) = None
        x_262 = (
            mul_47
            + l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_bias_
        )
        mul_47 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_bias_ = (None)
        x_263 = torch.nn.functional.dropout(x_262, 0.0, False, False)
        x_262 = None
        x_264 = torch.conv2d(
            x_263,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_263 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_265 = torch.nn.functional.dropout(x_264, 0.0, False, False)
        x_264 = None
        x_266 = mul_46 + x_265
        mul_46 = x_265 = None
        x_267 = x_266.permute(0, 2, 3, 1)
        x_266 = None
        x_268 = torch.nn.functional.layer_norm(
            x_267,
            (320,),
            l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_,
            None,
            1e-06,
        )
        x_267 = l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_ = (None)
        x_269 = x_268.permute(0, 3, 1, 2)
        x_268 = None
        x_270 = torch.conv2d(
            x_269,
            l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_269 = l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_ = (None)
        view_18 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale1_parameters_scale_.view(
            (512, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_48 = x_270 * view_18
        view_18 = None
        x_271 = x_270.permute(0, 2, 3, 1)
        x_270 = None
        x_272 = torch.nn.functional.layer_norm(
            x_271,
            (512,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_271 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (None)
        x_273 = x_272.permute(0, 3, 1, 2)
        x_272 = None
        x_274 = torch.conv2d(
            x_273,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_273 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_30 = torch.nn.functional.relu(x_274, inplace=False)
        x_274 = None
        pow_31 = relu_30**2
        relu_30 = None
        mul_49 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_scale_
            * pow_31
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_31
        ) = None
        x_275 = (
            mul_49
            + l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_49 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_276 = torch.conv2d(
            x_275,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        x_275 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_277 = torch.conv2d(
            x_276,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_276 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_278 = mul_48 + x_277
        mul_48 = x_277 = None
        view_19 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale2_parameters_scale_.view(
            (512, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_50 = x_278 * view_19
        view_19 = None
        x_279 = x_278.permute(0, 2, 3, 1)
        x_278 = None
        x_280 = torch.nn.functional.layer_norm(
            x_279,
            (512,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_279 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (None)
        x_281 = x_280.permute(0, 3, 1, 2)
        x_280 = None
        x_282 = torch.conv2d(
            x_281,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_281 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_31 = torch.nn.functional.relu(x_282, inplace=False)
        x_282 = None
        pow_32 = relu_31**2
        relu_31 = None
        mul_51 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
            * pow_32
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = (
            pow_32
        ) = None
        x_283 = (
            mul_51
            + l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        )
        mul_51 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = (None)
        x_284 = torch.nn.functional.dropout(x_283, 0.0, False, False)
        x_283 = None
        x_285 = torch.conv2d(
            x_284,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_284 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_286 = torch.nn.functional.dropout(x_285, 0.0, False, False)
        x_285 = None
        x_287 = mul_50 + x_286
        mul_50 = x_286 = None
        view_20 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale1_parameters_scale_.view(
            (512, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_52 = x_287 * view_20
        view_20 = None
        x_288 = x_287.permute(0, 2, 3, 1)
        x_287 = None
        x_289 = torch.nn.functional.layer_norm(
            x_288,
            (512,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_288 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (None)
        x_290 = x_289.permute(0, 3, 1, 2)
        x_289 = None
        x_291 = torch.conv2d(
            x_290,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_290 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_32 = torch.nn.functional.relu(x_291, inplace=False)
        x_291 = None
        pow_33 = relu_32**2
        relu_32 = None
        mul_53 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_scale_
            * pow_33
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_33
        ) = None
        x_292 = (
            mul_53
            + l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_53 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_293 = torch.conv2d(
            x_292,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        x_292 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_294 = torch.conv2d(
            x_293,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_293 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_295 = mul_52 + x_294
        mul_52 = x_294 = None
        view_21 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale2_parameters_scale_.view(
            (512, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_54 = x_295 * view_21
        view_21 = None
        x_296 = x_295.permute(0, 2, 3, 1)
        x_295 = None
        x_297 = torch.nn.functional.layer_norm(
            x_296,
            (512,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_296 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (None)
        x_298 = x_297.permute(0, 3, 1, 2)
        x_297 = None
        x_299 = torch.conv2d(
            x_298,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_298 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_33 = torch.nn.functional.relu(x_299, inplace=False)
        x_299 = None
        pow_34 = relu_33**2
        relu_33 = None
        mul_55 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
            * pow_34
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = (
            pow_34
        ) = None
        x_300 = (
            mul_55
            + l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        )
        mul_55 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = (None)
        x_301 = torch.nn.functional.dropout(x_300, 0.0, False, False)
        x_300 = None
        x_302 = torch.conv2d(
            x_301,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_301 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_303 = torch.nn.functional.dropout(x_302, 0.0, False, False)
        x_302 = None
        x_304 = mul_54 + x_303
        mul_54 = x_303 = None
        view_22 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale1_parameters_scale_.view(
            (512, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_56 = x_304 * view_22
        view_22 = None
        x_305 = x_304.permute(0, 2, 3, 1)
        x_304 = None
        x_306 = torch.nn.functional.layer_norm(
            x_305,
            (512,),
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_305 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (None)
        x_307 = x_306.permute(0, 3, 1, 2)
        x_306 = None
        x_308 = torch.conv2d(
            x_307,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_pwconv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_307 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_pwconv1_parameters_weight_ = (None)
        relu_34 = torch.nn.functional.relu(x_308, inplace=False)
        x_308 = None
        pow_35 = relu_34**2
        relu_34 = None
        mul_57 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_scale_
            * pow_35
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_scale_ = (
            pow_35
        ) = None
        x_309 = (
            mul_57
            + l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_bias_
        )
        mul_57 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_act1_parameters_bias_ = (None)
        x_310 = torch.conv2d(
            x_309,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        x_309 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_parameters_weight_ = (None)
        x_311 = torch.conv2d(
            x_310,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_pwconv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_310 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_pwconv2_parameters_weight_ = (None)
        x_312 = mul_56 + x_311
        mul_56 = x_311 = None
        view_23 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale2_parameters_scale_.view(
            (512, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_58 = x_312 * view_23
        view_23 = None
        x_313 = x_312.permute(0, 2, 3, 1)
        x_312 = None
        x_314 = torch.nn.functional.layer_norm(
            x_313,
            (512,),
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_313 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (None)
        x_315 = x_314.permute(0, 3, 1, 2)
        x_314 = None
        x_316 = torch.conv2d(
            x_315,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_315 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_35 = torch.nn.functional.relu(x_316, inplace=False)
        x_316 = None
        pow_36 = relu_35**2
        relu_35 = None
        mul_59 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_
            * pow_36
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_ = (
            pow_36
        ) = None
        x_317 = (
            mul_59
            + l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_
        )
        mul_59 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_ = (None)
        x_318 = torch.nn.functional.dropout(x_317, 0.0, False, False)
        x_317 = None
        x_319 = torch.conv2d(
            x_318,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_318 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_320 = torch.nn.functional.dropout(x_319, 0.0, False, False)
        x_319 = None
        x_321 = mul_58 + x_320
        mul_58 = x_320 = None
        x_322 = torch.nn.functional.adaptive_avg_pool2d(x_321, 1)
        x_321 = None
        x_323 = x_322.permute(0, 2, 3, 1)
        x_322 = None
        x_324 = torch.nn.functional.layer_norm(
            x_323,
            (512,),
            l_self_modules_head_modules_norm_parameters_weight_,
            l_self_modules_head_modules_norm_parameters_bias_,
            1e-06,
        )
        x_323 = (
            l_self_modules_head_modules_norm_parameters_weight_
        ) = l_self_modules_head_modules_norm_parameters_bias_ = None
        x_325 = x_324.permute(0, 3, 1, 2)
        x_324 = None
        x_326 = x_325.flatten(1, -1)
        x_325 = None
        x_327 = torch.nn.functional.dropout(x_326, 0.0, False, False)
        x_326 = None
        x_328 = torch._C._nn.linear(
            x_327,
            l_self_modules_head_modules_fc_modules_fc1_parameters_weight_,
            l_self_modules_head_modules_fc_modules_fc1_parameters_bias_,
        )
        x_327 = (
            l_self_modules_head_modules_fc_modules_fc1_parameters_weight_
        ) = l_self_modules_head_modules_fc_modules_fc1_parameters_bias_ = None
        relu_36 = torch.nn.functional.relu(x_328, inplace=False)
        x_328 = None
        x_329 = torch.square(relu_36)
        relu_36 = None
        x_330 = torch.nn.functional.layer_norm(
            x_329,
            (2048,),
            l_self_modules_head_modules_fc_modules_norm_parameters_weight_,
            l_self_modules_head_modules_fc_modules_norm_parameters_bias_,
            1e-06,
        )
        x_329 = (
            l_self_modules_head_modules_fc_modules_norm_parameters_weight_
        ) = l_self_modules_head_modules_fc_modules_norm_parameters_bias_ = None
        x_331 = torch.nn.functional.dropout(x_330, 0.0, False, False)
        x_330 = None
        x_332 = torch._C._nn.linear(
            x_331,
            l_self_modules_head_modules_fc_modules_fc2_parameters_weight_,
            l_self_modules_head_modules_fc_modules_fc2_parameters_bias_,
        )
        x_331 = (
            l_self_modules_head_modules_fc_modules_fc2_parameters_weight_
        ) = l_self_modules_head_modules_fc_modules_fc2_parameters_bias_ = None
        return (x_332,)
