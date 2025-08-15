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
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
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
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_proj_parameters_weight_
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
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_proj_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_proj_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_proj_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_proj_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_proj_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_proj_parameters_weight_
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
        x_111 = torch.nn.functional.layer_norm(
            x_110,
            (128,),
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
            256,
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
            (128,),
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
            (128,),
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
            256,
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
            (128,),
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
            (128,),
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
            256,
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
            (128,),
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
            (128,),
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
            256,
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
            (128,),
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
            (128,),
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
            256,
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
            (128,),
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
            (128,),
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
            256,
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
            (128,),
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
            (128,),
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
            256,
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
            (128,),
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
            (128,),
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
            256,
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
            (128,),
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
            (128,),
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
            256,
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
            (128,),
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
            (128,),
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
        reshape = x_266.reshape(1, 320, -1)
        x_266 = None
        x_267 = reshape.transpose(1, 2)
        reshape = None
        view = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale1_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_30 = x_267 * view
        view = None
        layer_norm_33 = torch.nn.functional.layer_norm(
            x_267,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_267 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (None)
        linear = torch._C._nn.linear(
            layer_norm_33,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_qkv_parameters_weight_,
            None,
        )
        layer_norm_33 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_qkv_parameters_weight_ = (None)
        reshape_1 = linear.reshape(1, 196, 3, 10, 32)
        linear = None
        qkv = reshape_1.permute(2, 0, 3, 1, 4)
        reshape_1 = None
        unbind = qkv.unbind(0)
        qkv = None
        q = unbind[0]
        k = unbind[1]
        v = unbind[2]
        unbind = None
        x_268 = torch._C._nn.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        q = k = v = None
        transpose_1 = x_268.transpose(1, 2)
        x_268 = None
        x_269 = transpose_1.reshape(1, 196, 320)
        transpose_1 = None
        x_270 = torch._C._nn.linear(
            x_269,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_proj_parameters_weight_,
            None,
        )
        x_269 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_proj_parameters_weight_ = (None)
        x_271 = torch.nn.functional.dropout(x_270, 0.0, False, False)
        x_270 = None
        x_272 = mul_30 + x_271
        mul_30 = x_271 = None
        view_1 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale2_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_31 = x_272 * view_1
        view_1 = None
        layer_norm_34 = torch.nn.functional.layer_norm(
            x_272,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_272 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (None)
        x_273 = torch._C._nn.linear(
            layer_norm_34,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        layer_norm_34 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_30 = torch.nn.functional.relu(x_273, inplace=False)
        x_273 = None
        pow_31 = relu_30**2
        relu_30 = None
        mul_32 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
            * pow_31
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = (
            pow_31
        ) = None
        x_274 = (
            mul_32
            + l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        )
        mul_32 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = (None)
        x_275 = torch.nn.functional.dropout(x_274, 0.0, False, False)
        x_274 = None
        x_276 = torch._C._nn.linear(
            x_275,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_275 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_277 = torch.nn.functional.dropout(x_276, 0.0, False, False)
        x_276 = None
        x_278 = mul_31 + x_277
        mul_31 = x_277 = None
        view_2 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale1_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_33 = x_278 * view_2
        view_2 = None
        layer_norm_35 = torch.nn.functional.layer_norm(
            x_278,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_278 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (None)
        linear_4 = torch._C._nn.linear(
            layer_norm_35,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_qkv_parameters_weight_,
            None,
        )
        layer_norm_35 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_qkv_parameters_weight_ = (None)
        reshape_3 = linear_4.reshape(1, 196, 3, 10, 32)
        linear_4 = None
        qkv_1 = reshape_3.permute(2, 0, 3, 1, 4)
        reshape_3 = None
        unbind_1 = qkv_1.unbind(0)
        qkv_1 = None
        q_1 = unbind_1[0]
        k_1 = unbind_1[1]
        v_1 = unbind_1[2]
        unbind_1 = None
        x_279 = torch._C._nn.scaled_dot_product_attention(q_1, k_1, v_1, dropout_p=0.0)
        q_1 = k_1 = v_1 = None
        transpose_2 = x_279.transpose(1, 2)
        x_279 = None
        x_280 = transpose_2.reshape(1, 196, 320)
        transpose_2 = None
        x_281 = torch._C._nn.linear(
            x_280,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_proj_parameters_weight_,
            None,
        )
        x_280 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_proj_parameters_weight_ = (None)
        x_282 = torch.nn.functional.dropout(x_281, 0.0, False, False)
        x_281 = None
        x_283 = mul_33 + x_282
        mul_33 = x_282 = None
        view_3 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale2_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_34 = x_283 * view_3
        view_3 = None
        layer_norm_36 = torch.nn.functional.layer_norm(
            x_283,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_283 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (None)
        x_284 = torch._C._nn.linear(
            layer_norm_36,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        layer_norm_36 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_31 = torch.nn.functional.relu(x_284, inplace=False)
        x_284 = None
        pow_32 = relu_31**2
        relu_31 = None
        mul_35 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
            * pow_32
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = (
            pow_32
        ) = None
        x_285 = (
            mul_35
            + l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        )
        mul_35 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = (None)
        x_286 = torch.nn.functional.dropout(x_285, 0.0, False, False)
        x_285 = None
        x_287 = torch._C._nn.linear(
            x_286,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_286 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_288 = torch.nn.functional.dropout(x_287, 0.0, False, False)
        x_287 = None
        x_289 = mul_34 + x_288
        mul_34 = x_288 = None
        view_4 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale1_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_36 = x_289 * view_4
        view_4 = None
        layer_norm_37 = torch.nn.functional.layer_norm(
            x_289,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_289 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (None)
        linear_8 = torch._C._nn.linear(
            layer_norm_37,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_qkv_parameters_weight_,
            None,
        )
        layer_norm_37 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_qkv_parameters_weight_ = (None)
        reshape_5 = linear_8.reshape(1, 196, 3, 10, 32)
        linear_8 = None
        qkv_2 = reshape_5.permute(2, 0, 3, 1, 4)
        reshape_5 = None
        unbind_2 = qkv_2.unbind(0)
        qkv_2 = None
        q_2 = unbind_2[0]
        k_2 = unbind_2[1]
        v_2 = unbind_2[2]
        unbind_2 = None
        x_290 = torch._C._nn.scaled_dot_product_attention(q_2, k_2, v_2, dropout_p=0.0)
        q_2 = k_2 = v_2 = None
        transpose_3 = x_290.transpose(1, 2)
        x_290 = None
        x_291 = transpose_3.reshape(1, 196, 320)
        transpose_3 = None
        x_292 = torch._C._nn.linear(
            x_291,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_proj_parameters_weight_,
            None,
        )
        x_291 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_proj_parameters_weight_ = (None)
        x_293 = torch.nn.functional.dropout(x_292, 0.0, False, False)
        x_292 = None
        x_294 = mul_36 + x_293
        mul_36 = x_293 = None
        view_5 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale2_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_37 = x_294 * view_5
        view_5 = None
        layer_norm_38 = torch.nn.functional.layer_norm(
            x_294,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_294 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (None)
        x_295 = torch._C._nn.linear(
            layer_norm_38,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        layer_norm_38 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_32 = torch.nn.functional.relu(x_295, inplace=False)
        x_295 = None
        pow_33 = relu_32**2
        relu_32 = None
        mul_38 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_
            * pow_33
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_ = (
            pow_33
        ) = None
        x_296 = (
            mul_38
            + l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_
        )
        mul_38 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_ = (None)
        x_297 = torch.nn.functional.dropout(x_296, 0.0, False, False)
        x_296 = None
        x_298 = torch._C._nn.linear(
            x_297,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_297 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_299 = torch.nn.functional.dropout(x_298, 0.0, False, False)
        x_298 = None
        x_300 = mul_37 + x_299
        mul_37 = x_299 = None
        view_6 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale1_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_39 = x_300 * view_6
        view_6 = None
        layer_norm_39 = torch.nn.functional.layer_norm(
            x_300,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_300 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_ = (None)
        linear_12 = torch._C._nn.linear(
            layer_norm_39,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_qkv_parameters_weight_,
            None,
        )
        layer_norm_39 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_qkv_parameters_weight_ = (None)
        reshape_7 = linear_12.reshape(1, 196, 3, 10, 32)
        linear_12 = None
        qkv_3 = reshape_7.permute(2, 0, 3, 1, 4)
        reshape_7 = None
        unbind_3 = qkv_3.unbind(0)
        qkv_3 = None
        q_3 = unbind_3[0]
        k_3 = unbind_3[1]
        v_3 = unbind_3[2]
        unbind_3 = None
        x_301 = torch._C._nn.scaled_dot_product_attention(q_3, k_3, v_3, dropout_p=0.0)
        q_3 = k_3 = v_3 = None
        transpose_4 = x_301.transpose(1, 2)
        x_301 = None
        x_302 = transpose_4.reshape(1, 196, 320)
        transpose_4 = None
        x_303 = torch._C._nn.linear(
            x_302,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_proj_parameters_weight_,
            None,
        )
        x_302 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_proj_parameters_weight_ = (None)
        x_304 = torch.nn.functional.dropout(x_303, 0.0, False, False)
        x_303 = None
        x_305 = mul_39 + x_304
        mul_39 = x_304 = None
        view_7 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale2_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_40 = x_305 * view_7
        view_7 = None
        layer_norm_40 = torch.nn.functional.layer_norm(
            x_305,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_305 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_ = (None)
        x_306 = torch._C._nn.linear(
            layer_norm_40,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        layer_norm_40 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_33 = torch.nn.functional.relu(x_306, inplace=False)
        x_306 = None
        pow_34 = relu_33**2
        relu_33 = None
        mul_41 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_
            * pow_34
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_ = (
            pow_34
        ) = None
        x_307 = (
            mul_41
            + l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_
        )
        mul_41 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_ = (None)
        x_308 = torch.nn.functional.dropout(x_307, 0.0, False, False)
        x_307 = None
        x_309 = torch._C._nn.linear(
            x_308,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_308 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_310 = torch.nn.functional.dropout(x_309, 0.0, False, False)
        x_309 = None
        x_311 = mul_40 + x_310
        mul_40 = x_310 = None
        view_8 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale1_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_42 = x_311 * view_8
        view_8 = None
        layer_norm_41 = torch.nn.functional.layer_norm(
            x_311,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_311 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_ = (None)
        linear_16 = torch._C._nn.linear(
            layer_norm_41,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_qkv_parameters_weight_,
            None,
        )
        layer_norm_41 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_qkv_parameters_weight_ = (None)
        reshape_9 = linear_16.reshape(1, 196, 3, 10, 32)
        linear_16 = None
        qkv_4 = reshape_9.permute(2, 0, 3, 1, 4)
        reshape_9 = None
        unbind_4 = qkv_4.unbind(0)
        qkv_4 = None
        q_4 = unbind_4[0]
        k_4 = unbind_4[1]
        v_4 = unbind_4[2]
        unbind_4 = None
        x_312 = torch._C._nn.scaled_dot_product_attention(q_4, k_4, v_4, dropout_p=0.0)
        q_4 = k_4 = v_4 = None
        transpose_5 = x_312.transpose(1, 2)
        x_312 = None
        x_313 = transpose_5.reshape(1, 196, 320)
        transpose_5 = None
        x_314 = torch._C._nn.linear(
            x_313,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_proj_parameters_weight_,
            None,
        )
        x_313 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_proj_parameters_weight_ = (None)
        x_315 = torch.nn.functional.dropout(x_314, 0.0, False, False)
        x_314 = None
        x_316 = mul_42 + x_315
        mul_42 = x_315 = None
        view_9 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale2_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_43 = x_316 * view_9
        view_9 = None
        layer_norm_42 = torch.nn.functional.layer_norm(
            x_316,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_316 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_ = (None)
        x_317 = torch._C._nn.linear(
            layer_norm_42,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        layer_norm_42 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_34 = torch.nn.functional.relu(x_317, inplace=False)
        x_317 = None
        pow_35 = relu_34**2
        relu_34 = None
        mul_44 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_
            * pow_35
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_ = (
            pow_35
        ) = None
        x_318 = (
            mul_44
            + l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_
        )
        mul_44 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_ = (None)
        x_319 = torch.nn.functional.dropout(x_318, 0.0, False, False)
        x_318 = None
        x_320 = torch._C._nn.linear(
            x_319,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_319 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_321 = torch.nn.functional.dropout(x_320, 0.0, False, False)
        x_320 = None
        x_322 = mul_43 + x_321
        mul_43 = x_321 = None
        view_10 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale1_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_45 = x_322 * view_10
        view_10 = None
        layer_norm_43 = torch.nn.functional.layer_norm(
            x_322,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_322 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_ = (None)
        linear_20 = torch._C._nn.linear(
            layer_norm_43,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_qkv_parameters_weight_,
            None,
        )
        layer_norm_43 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_qkv_parameters_weight_ = (None)
        reshape_11 = linear_20.reshape(1, 196, 3, 10, 32)
        linear_20 = None
        qkv_5 = reshape_11.permute(2, 0, 3, 1, 4)
        reshape_11 = None
        unbind_5 = qkv_5.unbind(0)
        qkv_5 = None
        q_5 = unbind_5[0]
        k_5 = unbind_5[1]
        v_5 = unbind_5[2]
        unbind_5 = None
        x_323 = torch._C._nn.scaled_dot_product_attention(q_5, k_5, v_5, dropout_p=0.0)
        q_5 = k_5 = v_5 = None
        transpose_6 = x_323.transpose(1, 2)
        x_323 = None
        x_324 = transpose_6.reshape(1, 196, 320)
        transpose_6 = None
        x_325 = torch._C._nn.linear(
            x_324,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_proj_parameters_weight_,
            None,
        )
        x_324 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_proj_parameters_weight_ = (None)
        x_326 = torch.nn.functional.dropout(x_325, 0.0, False, False)
        x_325 = None
        x_327 = mul_45 + x_326
        mul_45 = x_326 = None
        view_11 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale2_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_46 = x_327 * view_11
        view_11 = None
        layer_norm_44 = torch.nn.functional.layer_norm(
            x_327,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_327 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_ = (None)
        x_328 = torch._C._nn.linear(
            layer_norm_44,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        layer_norm_44 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_35 = torch.nn.functional.relu(x_328, inplace=False)
        x_328 = None
        pow_36 = relu_35**2
        relu_35 = None
        mul_47 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_
            * pow_36
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_ = (
            pow_36
        ) = None
        x_329 = (
            mul_47
            + l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_
        )
        mul_47 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_ = (None)
        x_330 = torch.nn.functional.dropout(x_329, 0.0, False, False)
        x_329 = None
        x_331 = torch._C._nn.linear(
            x_330,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_330 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_332 = torch.nn.functional.dropout(x_331, 0.0, False, False)
        x_331 = None
        x_333 = mul_46 + x_332
        mul_46 = x_332 = None
        view_12 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale1_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_48 = x_333 * view_12
        view_12 = None
        layer_norm_45 = torch.nn.functional.layer_norm(
            x_333,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_333 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_ = (None)
        linear_24 = torch._C._nn.linear(
            layer_norm_45,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_qkv_parameters_weight_,
            None,
        )
        layer_norm_45 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_qkv_parameters_weight_ = (None)
        reshape_13 = linear_24.reshape(1, 196, 3, 10, 32)
        linear_24 = None
        qkv_6 = reshape_13.permute(2, 0, 3, 1, 4)
        reshape_13 = None
        unbind_6 = qkv_6.unbind(0)
        qkv_6 = None
        q_6 = unbind_6[0]
        k_6 = unbind_6[1]
        v_6 = unbind_6[2]
        unbind_6 = None
        x_334 = torch._C._nn.scaled_dot_product_attention(q_6, k_6, v_6, dropout_p=0.0)
        q_6 = k_6 = v_6 = None
        transpose_7 = x_334.transpose(1, 2)
        x_334 = None
        x_335 = transpose_7.reshape(1, 196, 320)
        transpose_7 = None
        x_336 = torch._C._nn.linear(
            x_335,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_proj_parameters_weight_,
            None,
        )
        x_335 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_proj_parameters_weight_ = (None)
        x_337 = torch.nn.functional.dropout(x_336, 0.0, False, False)
        x_336 = None
        x_338 = mul_48 + x_337
        mul_48 = x_337 = None
        view_13 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale2_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_49 = x_338 * view_13
        view_13 = None
        layer_norm_46 = torch.nn.functional.layer_norm(
            x_338,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_338 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_ = (None)
        x_339 = torch._C._nn.linear(
            layer_norm_46,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        layer_norm_46 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_36 = torch.nn.functional.relu(x_339, inplace=False)
        x_339 = None
        pow_37 = relu_36**2
        relu_36 = None
        mul_50 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_scale_
            * pow_37
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_scale_ = (
            pow_37
        ) = None
        x_340 = (
            mul_50
            + l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_bias_
        )
        mul_50 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_act_parameters_bias_ = (None)
        x_341 = torch.nn.functional.dropout(x_340, 0.0, False, False)
        x_340 = None
        x_342 = torch._C._nn.linear(
            x_341,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_341 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_343 = torch.nn.functional.dropout(x_342, 0.0, False, False)
        x_342 = None
        x_344 = mul_49 + x_343
        mul_49 = x_343 = None
        view_14 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale1_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_51 = x_344 * view_14
        view_14 = None
        layer_norm_47 = torch.nn.functional.layer_norm(
            x_344,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_344 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_ = (None)
        linear_28 = torch._C._nn.linear(
            layer_norm_47,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_qkv_parameters_weight_,
            None,
        )
        layer_norm_47 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_qkv_parameters_weight_ = (None)
        reshape_15 = linear_28.reshape(1, 196, 3, 10, 32)
        linear_28 = None
        qkv_7 = reshape_15.permute(2, 0, 3, 1, 4)
        reshape_15 = None
        unbind_7 = qkv_7.unbind(0)
        qkv_7 = None
        q_7 = unbind_7[0]
        k_7 = unbind_7[1]
        v_7 = unbind_7[2]
        unbind_7 = None
        x_345 = torch._C._nn.scaled_dot_product_attention(q_7, k_7, v_7, dropout_p=0.0)
        q_7 = k_7 = v_7 = None
        transpose_8 = x_345.transpose(1, 2)
        x_345 = None
        x_346 = transpose_8.reshape(1, 196, 320)
        transpose_8 = None
        x_347 = torch._C._nn.linear(
            x_346,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_proj_parameters_weight_,
            None,
        )
        x_346 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_proj_parameters_weight_ = (None)
        x_348 = torch.nn.functional.dropout(x_347, 0.0, False, False)
        x_347 = None
        x_349 = mul_51 + x_348
        mul_51 = x_348 = None
        view_15 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale2_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_52 = x_349 * view_15
        view_15 = None
        layer_norm_48 = torch.nn.functional.layer_norm(
            x_349,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_349 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_ = (None)
        x_350 = torch._C._nn.linear(
            layer_norm_48,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        layer_norm_48 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_37 = torch.nn.functional.relu(x_350, inplace=False)
        x_350 = None
        pow_38 = relu_37**2
        relu_37 = None
        mul_53 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_scale_
            * pow_38
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_scale_ = (
            pow_38
        ) = None
        x_351 = (
            mul_53
            + l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_bias_
        )
        mul_53 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_act_parameters_bias_ = (None)
        x_352 = torch.nn.functional.dropout(x_351, 0.0, False, False)
        x_351 = None
        x_353 = torch._C._nn.linear(
            x_352,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_352 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_354 = torch.nn.functional.dropout(x_353, 0.0, False, False)
        x_353 = None
        x_355 = mul_52 + x_354
        mul_52 = x_354 = None
        view_16 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale1_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_54 = x_355 * view_16
        view_16 = None
        layer_norm_49 = torch.nn.functional.layer_norm(
            x_355,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_355 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_ = (None)
        linear_32 = torch._C._nn.linear(
            layer_norm_49,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_qkv_parameters_weight_,
            None,
        )
        layer_norm_49 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_qkv_parameters_weight_ = (None)
        reshape_17 = linear_32.reshape(1, 196, 3, 10, 32)
        linear_32 = None
        qkv_8 = reshape_17.permute(2, 0, 3, 1, 4)
        reshape_17 = None
        unbind_8 = qkv_8.unbind(0)
        qkv_8 = None
        q_8 = unbind_8[0]
        k_8 = unbind_8[1]
        v_8 = unbind_8[2]
        unbind_8 = None
        x_356 = torch._C._nn.scaled_dot_product_attention(q_8, k_8, v_8, dropout_p=0.0)
        q_8 = k_8 = v_8 = None
        transpose_9 = x_356.transpose(1, 2)
        x_356 = None
        x_357 = transpose_9.reshape(1, 196, 320)
        transpose_9 = None
        x_358 = torch._C._nn.linear(
            x_357,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_proj_parameters_weight_,
            None,
        )
        x_357 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_proj_parameters_weight_ = (None)
        x_359 = torch.nn.functional.dropout(x_358, 0.0, False, False)
        x_358 = None
        x_360 = mul_54 + x_359
        mul_54 = x_359 = None
        view_17 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale2_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_55 = x_360 * view_17
        view_17 = None
        layer_norm_50 = torch.nn.functional.layer_norm(
            x_360,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_360 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_ = (None)
        x_361 = torch._C._nn.linear(
            layer_norm_50,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        layer_norm_50 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_38 = torch.nn.functional.relu(x_361, inplace=False)
        x_361 = None
        pow_39 = relu_38**2
        relu_38 = None
        mul_56 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_scale_
            * pow_39
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_scale_ = (
            pow_39
        ) = None
        x_362 = (
            mul_56
            + l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_bias_
        )
        mul_56 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_act_parameters_bias_ = (None)
        x_363 = torch.nn.functional.dropout(x_362, 0.0, False, False)
        x_362 = None
        x_364 = torch._C._nn.linear(
            x_363,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_363 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_365 = torch.nn.functional.dropout(x_364, 0.0, False, False)
        x_364 = None
        x_366 = mul_55 + x_365
        mul_55 = x_365 = None
        view_18 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale1_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_57 = x_366 * view_18
        view_18 = None
        layer_norm_51 = torch.nn.functional.layer_norm(
            x_366,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_366 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_ = (None)
        linear_36 = torch._C._nn.linear(
            layer_norm_51,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_qkv_parameters_weight_,
            None,
        )
        layer_norm_51 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_qkv_parameters_weight_ = (None)
        reshape_19 = linear_36.reshape(1, 196, 3, 10, 32)
        linear_36 = None
        qkv_9 = reshape_19.permute(2, 0, 3, 1, 4)
        reshape_19 = None
        unbind_9 = qkv_9.unbind(0)
        qkv_9 = None
        q_9 = unbind_9[0]
        k_9 = unbind_9[1]
        v_9 = unbind_9[2]
        unbind_9 = None
        x_367 = torch._C._nn.scaled_dot_product_attention(q_9, k_9, v_9, dropout_p=0.0)
        q_9 = k_9 = v_9 = None
        transpose_10 = x_367.transpose(1, 2)
        x_367 = None
        x_368 = transpose_10.reshape(1, 196, 320)
        transpose_10 = None
        x_369 = torch._C._nn.linear(
            x_368,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_proj_parameters_weight_,
            None,
        )
        x_368 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_proj_parameters_weight_ = (None)
        x_370 = torch.nn.functional.dropout(x_369, 0.0, False, False)
        x_369 = None
        x_371 = mul_57 + x_370
        mul_57 = x_370 = None
        view_19 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale2_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_58 = x_371 * view_19
        view_19 = None
        layer_norm_52 = torch.nn.functional.layer_norm(
            x_371,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_371 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_ = (None)
        x_372 = torch._C._nn.linear(
            layer_norm_52,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        layer_norm_52 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_39 = torch.nn.functional.relu(x_372, inplace=False)
        x_372 = None
        pow_40 = relu_39**2
        relu_39 = None
        mul_59 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_scale_
            * pow_40
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_scale_ = (
            pow_40
        ) = None
        x_373 = (
            mul_59
            + l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_bias_
        )
        mul_59 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_act_parameters_bias_ = (None)
        x_374 = torch.nn.functional.dropout(x_373, 0.0, False, False)
        x_373 = None
        x_375 = torch._C._nn.linear(
            x_374,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_374 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_376 = torch.nn.functional.dropout(x_375, 0.0, False, False)
        x_375 = None
        x_377 = mul_58 + x_376
        mul_58 = x_376 = None
        view_20 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale1_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_60 = x_377 * view_20
        view_20 = None
        layer_norm_53 = torch.nn.functional.layer_norm(
            x_377,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_377 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_ = (None)
        linear_40 = torch._C._nn.linear(
            layer_norm_53,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_qkv_parameters_weight_,
            None,
        )
        layer_norm_53 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_qkv_parameters_weight_ = (None)
        reshape_21 = linear_40.reshape(1, 196, 3, 10, 32)
        linear_40 = None
        qkv_10 = reshape_21.permute(2, 0, 3, 1, 4)
        reshape_21 = None
        unbind_10 = qkv_10.unbind(0)
        qkv_10 = None
        q_10 = unbind_10[0]
        k_10 = unbind_10[1]
        v_10 = unbind_10[2]
        unbind_10 = None
        x_378 = torch._C._nn.scaled_dot_product_attention(
            q_10, k_10, v_10, dropout_p=0.0
        )
        q_10 = k_10 = v_10 = None
        transpose_11 = x_378.transpose(1, 2)
        x_378 = None
        x_379 = transpose_11.reshape(1, 196, 320)
        transpose_11 = None
        x_380 = torch._C._nn.linear(
            x_379,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_proj_parameters_weight_,
            None,
        )
        x_379 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_proj_parameters_weight_ = (None)
        x_381 = torch.nn.functional.dropout(x_380, 0.0, False, False)
        x_380 = None
        x_382 = mul_60 + x_381
        mul_60 = x_381 = None
        view_21 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale2_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_61 = x_382 * view_21
        view_21 = None
        layer_norm_54 = torch.nn.functional.layer_norm(
            x_382,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_382 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_ = (None)
        x_383 = torch._C._nn.linear(
            layer_norm_54,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        layer_norm_54 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_40 = torch.nn.functional.relu(x_383, inplace=False)
        x_383 = None
        pow_41 = relu_40**2
        relu_40 = None
        mul_62 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_scale_
            * pow_41
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_scale_ = (
            pow_41
        ) = None
        x_384 = (
            mul_62
            + l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_bias_
        )
        mul_62 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_act_parameters_bias_ = (None)
        x_385 = torch.nn.functional.dropout(x_384, 0.0, False, False)
        x_384 = None
        x_386 = torch._C._nn.linear(
            x_385,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_385 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_387 = torch.nn.functional.dropout(x_386, 0.0, False, False)
        x_386 = None
        x_388 = mul_61 + x_387
        mul_61 = x_387 = None
        view_22 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale1_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_63 = x_388 * view_22
        view_22 = None
        layer_norm_55 = torch.nn.functional.layer_norm(
            x_388,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_388 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_ = (None)
        linear_44 = torch._C._nn.linear(
            layer_norm_55,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_qkv_parameters_weight_,
            None,
        )
        layer_norm_55 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_qkv_parameters_weight_ = (None)
        reshape_23 = linear_44.reshape(1, 196, 3, 10, 32)
        linear_44 = None
        qkv_11 = reshape_23.permute(2, 0, 3, 1, 4)
        reshape_23 = None
        unbind_11 = qkv_11.unbind(0)
        qkv_11 = None
        q_11 = unbind_11[0]
        k_11 = unbind_11[1]
        v_11 = unbind_11[2]
        unbind_11 = None
        x_389 = torch._C._nn.scaled_dot_product_attention(
            q_11, k_11, v_11, dropout_p=0.0
        )
        q_11 = k_11 = v_11 = None
        transpose_12 = x_389.transpose(1, 2)
        x_389 = None
        x_390 = transpose_12.reshape(1, 196, 320)
        transpose_12 = None
        x_391 = torch._C._nn.linear(
            x_390,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_proj_parameters_weight_,
            None,
        )
        x_390 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_proj_parameters_weight_ = (None)
        x_392 = torch.nn.functional.dropout(x_391, 0.0, False, False)
        x_391 = None
        x_393 = mul_63 + x_392
        mul_63 = x_392 = None
        view_23 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale2_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_64 = x_393 * view_23
        view_23 = None
        layer_norm_56 = torch.nn.functional.layer_norm(
            x_393,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_393 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_ = (None)
        x_394 = torch._C._nn.linear(
            layer_norm_56,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        layer_norm_56 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_41 = torch.nn.functional.relu(x_394, inplace=False)
        x_394 = None
        pow_42 = relu_41**2
        relu_41 = None
        mul_65 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_scale_
            * pow_42
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_scale_ = (
            pow_42
        ) = None
        x_395 = (
            mul_65
            + l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_bias_
        )
        mul_65 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_act_parameters_bias_ = (None)
        x_396 = torch.nn.functional.dropout(x_395, 0.0, False, False)
        x_395 = None
        x_397 = torch._C._nn.linear(
            x_396,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_396 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_398 = torch.nn.functional.dropout(x_397, 0.0, False, False)
        x_397 = None
        x_399 = mul_64 + x_398
        mul_64 = x_398 = None
        view_24 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale1_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_66 = x_399 * view_24
        view_24 = None
        layer_norm_57 = torch.nn.functional.layer_norm(
            x_399,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_399 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_ = (None)
        linear_48 = torch._C._nn.linear(
            layer_norm_57,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_qkv_parameters_weight_,
            None,
        )
        layer_norm_57 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_qkv_parameters_weight_ = (None)
        reshape_25 = linear_48.reshape(1, 196, 3, 10, 32)
        linear_48 = None
        qkv_12 = reshape_25.permute(2, 0, 3, 1, 4)
        reshape_25 = None
        unbind_12 = qkv_12.unbind(0)
        qkv_12 = None
        q_12 = unbind_12[0]
        k_12 = unbind_12[1]
        v_12 = unbind_12[2]
        unbind_12 = None
        x_400 = torch._C._nn.scaled_dot_product_attention(
            q_12, k_12, v_12, dropout_p=0.0
        )
        q_12 = k_12 = v_12 = None
        transpose_13 = x_400.transpose(1, 2)
        x_400 = None
        x_401 = transpose_13.reshape(1, 196, 320)
        transpose_13 = None
        x_402 = torch._C._nn.linear(
            x_401,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_proj_parameters_weight_,
            None,
        )
        x_401 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_proj_parameters_weight_ = (None)
        x_403 = torch.nn.functional.dropout(x_402, 0.0, False, False)
        x_402 = None
        x_404 = mul_66 + x_403
        mul_66 = x_403 = None
        view_25 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale2_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_67 = x_404 * view_25
        view_25 = None
        layer_norm_58 = torch.nn.functional.layer_norm(
            x_404,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_404 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_ = (None)
        x_405 = torch._C._nn.linear(
            layer_norm_58,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        layer_norm_58 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_42 = torch.nn.functional.relu(x_405, inplace=False)
        x_405 = None
        pow_43 = relu_42**2
        relu_42 = None
        mul_68 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_scale_
            * pow_43
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_scale_ = (
            pow_43
        ) = None
        x_406 = (
            mul_68
            + l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_bias_
        )
        mul_68 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_act_parameters_bias_ = (None)
        x_407 = torch.nn.functional.dropout(x_406, 0.0, False, False)
        x_406 = None
        x_408 = torch._C._nn.linear(
            x_407,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_407 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_409 = torch.nn.functional.dropout(x_408, 0.0, False, False)
        x_408 = None
        x_410 = mul_67 + x_409
        mul_67 = x_409 = None
        view_26 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale1_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_69 = x_410 * view_26
        view_26 = None
        layer_norm_59 = torch.nn.functional.layer_norm(
            x_410,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_410 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_ = (None)
        linear_52 = torch._C._nn.linear(
            layer_norm_59,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_qkv_parameters_weight_,
            None,
        )
        layer_norm_59 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_qkv_parameters_weight_ = (None)
        reshape_27 = linear_52.reshape(1, 196, 3, 10, 32)
        linear_52 = None
        qkv_13 = reshape_27.permute(2, 0, 3, 1, 4)
        reshape_27 = None
        unbind_13 = qkv_13.unbind(0)
        qkv_13 = None
        q_13 = unbind_13[0]
        k_13 = unbind_13[1]
        v_13 = unbind_13[2]
        unbind_13 = None
        x_411 = torch._C._nn.scaled_dot_product_attention(
            q_13, k_13, v_13, dropout_p=0.0
        )
        q_13 = k_13 = v_13 = None
        transpose_14 = x_411.transpose(1, 2)
        x_411 = None
        x_412 = transpose_14.reshape(1, 196, 320)
        transpose_14 = None
        x_413 = torch._C._nn.linear(
            x_412,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_proj_parameters_weight_,
            None,
        )
        x_412 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_proj_parameters_weight_ = (None)
        x_414 = torch.nn.functional.dropout(x_413, 0.0, False, False)
        x_413 = None
        x_415 = mul_69 + x_414
        mul_69 = x_414 = None
        view_27 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale2_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_70 = x_415 * view_27
        view_27 = None
        layer_norm_60 = torch.nn.functional.layer_norm(
            x_415,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_415 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_ = (None)
        x_416 = torch._C._nn.linear(
            layer_norm_60,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        layer_norm_60 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_43 = torch.nn.functional.relu(x_416, inplace=False)
        x_416 = None
        pow_44 = relu_43**2
        relu_43 = None
        mul_71 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_scale_
            * pow_44
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_scale_ = (
            pow_44
        ) = None
        x_417 = (
            mul_71
            + l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_bias_
        )
        mul_71 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_act_parameters_bias_ = (None)
        x_418 = torch.nn.functional.dropout(x_417, 0.0, False, False)
        x_417 = None
        x_419 = torch._C._nn.linear(
            x_418,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_418 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_420 = torch.nn.functional.dropout(x_419, 0.0, False, False)
        x_419 = None
        x_421 = mul_70 + x_420
        mul_70 = x_420 = None
        view_28 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale1_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_72 = x_421 * view_28
        view_28 = None
        layer_norm_61 = torch.nn.functional.layer_norm(
            x_421,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_421 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_ = (None)
        linear_56 = torch._C._nn.linear(
            layer_norm_61,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_qkv_parameters_weight_,
            None,
        )
        layer_norm_61 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_qkv_parameters_weight_ = (None)
        reshape_29 = linear_56.reshape(1, 196, 3, 10, 32)
        linear_56 = None
        qkv_14 = reshape_29.permute(2, 0, 3, 1, 4)
        reshape_29 = None
        unbind_14 = qkv_14.unbind(0)
        qkv_14 = None
        q_14 = unbind_14[0]
        k_14 = unbind_14[1]
        v_14 = unbind_14[2]
        unbind_14 = None
        x_422 = torch._C._nn.scaled_dot_product_attention(
            q_14, k_14, v_14, dropout_p=0.0
        )
        q_14 = k_14 = v_14 = None
        transpose_15 = x_422.transpose(1, 2)
        x_422 = None
        x_423 = transpose_15.reshape(1, 196, 320)
        transpose_15 = None
        x_424 = torch._C._nn.linear(
            x_423,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_proj_parameters_weight_,
            None,
        )
        x_423 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_proj_parameters_weight_ = (None)
        x_425 = torch.nn.functional.dropout(x_424, 0.0, False, False)
        x_424 = None
        x_426 = mul_72 + x_425
        mul_72 = x_425 = None
        view_29 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale2_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_73 = x_426 * view_29
        view_29 = None
        layer_norm_62 = torch.nn.functional.layer_norm(
            x_426,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_426 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_ = (None)
        x_427 = torch._C._nn.linear(
            layer_norm_62,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        layer_norm_62 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_44 = torch.nn.functional.relu(x_427, inplace=False)
        x_427 = None
        pow_45 = relu_44**2
        relu_44 = None
        mul_74 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_scale_
            * pow_45
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_scale_ = (
            pow_45
        ) = None
        x_428 = (
            mul_74
            + l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_bias_
        )
        mul_74 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_act_parameters_bias_ = (None)
        x_429 = torch.nn.functional.dropout(x_428, 0.0, False, False)
        x_428 = None
        x_430 = torch._C._nn.linear(
            x_429,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_429 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_431 = torch.nn.functional.dropout(x_430, 0.0, False, False)
        x_430 = None
        x_432 = mul_73 + x_431
        mul_73 = x_431 = None
        view_30 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale1_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_75 = x_432 * view_30
        view_30 = None
        layer_norm_63 = torch.nn.functional.layer_norm(
            x_432,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_432 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_ = (None)
        linear_60 = torch._C._nn.linear(
            layer_norm_63,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_qkv_parameters_weight_,
            None,
        )
        layer_norm_63 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_qkv_parameters_weight_ = (None)
        reshape_31 = linear_60.reshape(1, 196, 3, 10, 32)
        linear_60 = None
        qkv_15 = reshape_31.permute(2, 0, 3, 1, 4)
        reshape_31 = None
        unbind_15 = qkv_15.unbind(0)
        qkv_15 = None
        q_15 = unbind_15[0]
        k_15 = unbind_15[1]
        v_15 = unbind_15[2]
        unbind_15 = None
        x_433 = torch._C._nn.scaled_dot_product_attention(
            q_15, k_15, v_15, dropout_p=0.0
        )
        q_15 = k_15 = v_15 = None
        transpose_16 = x_433.transpose(1, 2)
        x_433 = None
        x_434 = transpose_16.reshape(1, 196, 320)
        transpose_16 = None
        x_435 = torch._C._nn.linear(
            x_434,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_proj_parameters_weight_,
            None,
        )
        x_434 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_proj_parameters_weight_ = (None)
        x_436 = torch.nn.functional.dropout(x_435, 0.0, False, False)
        x_435 = None
        x_437 = mul_75 + x_436
        mul_75 = x_436 = None
        view_31 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale2_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_76 = x_437 * view_31
        view_31 = None
        layer_norm_64 = torch.nn.functional.layer_norm(
            x_437,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_437 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_ = (None)
        x_438 = torch._C._nn.linear(
            layer_norm_64,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        layer_norm_64 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_45 = torch.nn.functional.relu(x_438, inplace=False)
        x_438 = None
        pow_46 = relu_45**2
        relu_45 = None
        mul_77 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_scale_
            * pow_46
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_scale_ = (
            pow_46
        ) = None
        x_439 = (
            mul_77
            + l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_bias_
        )
        mul_77 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_act_parameters_bias_ = (None)
        x_440 = torch.nn.functional.dropout(x_439, 0.0, False, False)
        x_439 = None
        x_441 = torch._C._nn.linear(
            x_440,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_440 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_442 = torch.nn.functional.dropout(x_441, 0.0, False, False)
        x_441 = None
        x_443 = mul_76 + x_442
        mul_76 = x_442 = None
        view_32 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale1_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_78 = x_443 * view_32
        view_32 = None
        layer_norm_65 = torch.nn.functional.layer_norm(
            x_443,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_443 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_ = (None)
        linear_64 = torch._C._nn.linear(
            layer_norm_65,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_qkv_parameters_weight_,
            None,
        )
        layer_norm_65 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_qkv_parameters_weight_ = (None)
        reshape_33 = linear_64.reshape(1, 196, 3, 10, 32)
        linear_64 = None
        qkv_16 = reshape_33.permute(2, 0, 3, 1, 4)
        reshape_33 = None
        unbind_16 = qkv_16.unbind(0)
        qkv_16 = None
        q_16 = unbind_16[0]
        k_16 = unbind_16[1]
        v_16 = unbind_16[2]
        unbind_16 = None
        x_444 = torch._C._nn.scaled_dot_product_attention(
            q_16, k_16, v_16, dropout_p=0.0
        )
        q_16 = k_16 = v_16 = None
        transpose_17 = x_444.transpose(1, 2)
        x_444 = None
        x_445 = transpose_17.reshape(1, 196, 320)
        transpose_17 = None
        x_446 = torch._C._nn.linear(
            x_445,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_proj_parameters_weight_,
            None,
        )
        x_445 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_proj_parameters_weight_ = (None)
        x_447 = torch.nn.functional.dropout(x_446, 0.0, False, False)
        x_446 = None
        x_448 = mul_78 + x_447
        mul_78 = x_447 = None
        view_33 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale2_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_79 = x_448 * view_33
        view_33 = None
        layer_norm_66 = torch.nn.functional.layer_norm(
            x_448,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_448 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_ = (None)
        x_449 = torch._C._nn.linear(
            layer_norm_66,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        layer_norm_66 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_46 = torch.nn.functional.relu(x_449, inplace=False)
        x_449 = None
        pow_47 = relu_46**2
        relu_46 = None
        mul_80 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_scale_
            * pow_47
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_scale_ = (
            pow_47
        ) = None
        x_450 = (
            mul_80
            + l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_bias_
        )
        mul_80 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_act_parameters_bias_ = (None)
        x_451 = torch.nn.functional.dropout(x_450, 0.0, False, False)
        x_450 = None
        x_452 = torch._C._nn.linear(
            x_451,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_451 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_453 = torch.nn.functional.dropout(x_452, 0.0, False, False)
        x_452 = None
        x_454 = mul_79 + x_453
        mul_79 = x_453 = None
        view_34 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_res_scale1_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_81 = x_454 * view_34
        view_34 = None
        layer_norm_67 = torch.nn.functional.layer_norm(
            x_454,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_454 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_ = (None)
        linear_68 = torch._C._nn.linear(
            layer_norm_67,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_qkv_parameters_weight_,
            None,
        )
        layer_norm_67 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_qkv_parameters_weight_ = (None)
        reshape_35 = linear_68.reshape(1, 196, 3, 10, 32)
        linear_68 = None
        qkv_17 = reshape_35.permute(2, 0, 3, 1, 4)
        reshape_35 = None
        unbind_17 = qkv_17.unbind(0)
        qkv_17 = None
        q_17 = unbind_17[0]
        k_17 = unbind_17[1]
        v_17 = unbind_17[2]
        unbind_17 = None
        x_455 = torch._C._nn.scaled_dot_product_attention(
            q_17, k_17, v_17, dropout_p=0.0
        )
        q_17 = k_17 = v_17 = None
        transpose_18 = x_455.transpose(1, 2)
        x_455 = None
        x_456 = transpose_18.reshape(1, 196, 320)
        transpose_18 = None
        x_457 = torch._C._nn.linear(
            x_456,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_proj_parameters_weight_,
            None,
        )
        x_456 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_proj_parameters_weight_ = (None)
        x_458 = torch.nn.functional.dropout(x_457, 0.0, False, False)
        x_457 = None
        x_459 = mul_81 + x_458
        mul_81 = x_458 = None
        view_35 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_res_scale2_parameters_scale_.view(
            (320,)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_82 = x_459 * view_35
        view_35 = None
        layer_norm_68 = torch.nn.functional.layer_norm(
            x_459,
            (320,),
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_459 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_ = (None)
        x_460 = torch._C._nn.linear(
            layer_norm_68,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        layer_norm_68 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_47 = torch.nn.functional.relu(x_460, inplace=False)
        x_460 = None
        pow_48 = relu_47**2
        relu_47 = None
        mul_83 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_act_parameters_scale_
            * pow_48
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_act_parameters_scale_ = (
            pow_48
        ) = None
        x_461 = (
            mul_83
            + l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_act_parameters_bias_
        )
        mul_83 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_act_parameters_bias_ = (None)
        x_462 = torch.nn.functional.dropout(x_461, 0.0, False, False)
        x_461 = None
        x_463 = torch._C._nn.linear(
            x_462,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_462 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_464 = torch.nn.functional.dropout(x_463, 0.0, False, False)
        x_463 = None
        x_465 = mul_82 + x_464
        mul_82 = x_464 = None
        transpose_19 = x_465.transpose(1, 2)
        x_465 = None
        x_466 = transpose_19.reshape(1, 320, 14, 14)
        transpose_19 = None
        x_467 = x_466.permute(0, 2, 3, 1)
        x_466 = None
        x_468 = torch.nn.functional.layer_norm(
            x_467,
            (320,),
            l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_,
            None,
            1e-06,
        )
        x_467 = l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_ = (None)
        x_469 = x_468.permute(0, 3, 1, 2)
        x_468 = None
        x_470 = torch.conv2d(
            x_469,
            l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_469 = l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_ = (None)
        reshape_38 = x_470.reshape(1, 512, -1)
        x_470 = None
        x_471 = reshape_38.transpose(1, 2)
        reshape_38 = None
        view_36 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale1_parameters_scale_.view(
            (512,)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_84 = x_471 * view_36
        view_36 = None
        layer_norm_70 = torch.nn.functional.layer_norm(
            x_471,
            (512,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_471 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (None)
        linear_72 = torch._C._nn.linear(
            layer_norm_70,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_qkv_parameters_weight_,
            None,
        )
        layer_norm_70 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_qkv_parameters_weight_ = (None)
        reshape_39 = linear_72.reshape(1, 49, 3, 16, 32)
        linear_72 = None
        qkv_18 = reshape_39.permute(2, 0, 3, 1, 4)
        reshape_39 = None
        unbind_18 = qkv_18.unbind(0)
        qkv_18 = None
        q_18 = unbind_18[0]
        k_18 = unbind_18[1]
        v_18 = unbind_18[2]
        unbind_18 = None
        x_472 = torch._C._nn.scaled_dot_product_attention(
            q_18, k_18, v_18, dropout_p=0.0
        )
        q_18 = k_18 = v_18 = None
        transpose_21 = x_472.transpose(1, 2)
        x_472 = None
        x_473 = transpose_21.reshape(1, 49, 512)
        transpose_21 = None
        x_474 = torch._C._nn.linear(
            x_473,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_proj_parameters_weight_,
            None,
        )
        x_473 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_proj_parameters_weight_ = (None)
        x_475 = torch.nn.functional.dropout(x_474, 0.0, False, False)
        x_474 = None
        x_476 = mul_84 + x_475
        mul_84 = x_475 = None
        view_37 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale2_parameters_scale_.view(
            (512,)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_85 = x_476 * view_37
        view_37 = None
        layer_norm_71 = torch.nn.functional.layer_norm(
            x_476,
            (512,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_476 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (None)
        x_477 = torch._C._nn.linear(
            layer_norm_71,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        layer_norm_71 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_48 = torch.nn.functional.relu(x_477, inplace=False)
        x_477 = None
        pow_49 = relu_48**2
        relu_48 = None
        mul_86 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
            * pow_49
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = (
            pow_49
        ) = None
        x_478 = (
            mul_86
            + l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        )
        mul_86 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = (None)
        x_479 = torch.nn.functional.dropout(x_478, 0.0, False, False)
        x_478 = None
        x_480 = torch._C._nn.linear(
            x_479,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_479 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_481 = torch.nn.functional.dropout(x_480, 0.0, False, False)
        x_480 = None
        x_482 = mul_85 + x_481
        mul_85 = x_481 = None
        view_38 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale1_parameters_scale_.view(
            (512,)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_87 = x_482 * view_38
        view_38 = None
        layer_norm_72 = torch.nn.functional.layer_norm(
            x_482,
            (512,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_482 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (None)
        linear_76 = torch._C._nn.linear(
            layer_norm_72,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_qkv_parameters_weight_,
            None,
        )
        layer_norm_72 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_qkv_parameters_weight_ = (None)
        reshape_41 = linear_76.reshape(1, 49, 3, 16, 32)
        linear_76 = None
        qkv_19 = reshape_41.permute(2, 0, 3, 1, 4)
        reshape_41 = None
        unbind_19 = qkv_19.unbind(0)
        qkv_19 = None
        q_19 = unbind_19[0]
        k_19 = unbind_19[1]
        v_19 = unbind_19[2]
        unbind_19 = None
        x_483 = torch._C._nn.scaled_dot_product_attention(
            q_19, k_19, v_19, dropout_p=0.0
        )
        q_19 = k_19 = v_19 = None
        transpose_22 = x_483.transpose(1, 2)
        x_483 = None
        x_484 = transpose_22.reshape(1, 49, 512)
        transpose_22 = None
        x_485 = torch._C._nn.linear(
            x_484,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_proj_parameters_weight_,
            None,
        )
        x_484 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_proj_parameters_weight_ = (None)
        x_486 = torch.nn.functional.dropout(x_485, 0.0, False, False)
        x_485 = None
        x_487 = mul_87 + x_486
        mul_87 = x_486 = None
        view_39 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale2_parameters_scale_.view(
            (512,)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_88 = x_487 * view_39
        view_39 = None
        layer_norm_73 = torch.nn.functional.layer_norm(
            x_487,
            (512,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_487 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (None)
        x_488 = torch._C._nn.linear(
            layer_norm_73,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        layer_norm_73 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_49 = torch.nn.functional.relu(x_488, inplace=False)
        x_488 = None
        pow_50 = relu_49**2
        relu_49 = None
        mul_89 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
            * pow_50
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = (
            pow_50
        ) = None
        x_489 = (
            mul_89
            + l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        )
        mul_89 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = (None)
        x_490 = torch.nn.functional.dropout(x_489, 0.0, False, False)
        x_489 = None
        x_491 = torch._C._nn.linear(
            x_490,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_490 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_492 = torch.nn.functional.dropout(x_491, 0.0, False, False)
        x_491 = None
        x_493 = mul_88 + x_492
        mul_88 = x_492 = None
        view_40 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale1_parameters_scale_.view(
            (512,)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_90 = x_493 * view_40
        view_40 = None
        layer_norm_74 = torch.nn.functional.layer_norm(
            x_493,
            (512,),
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_493 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (None)
        linear_80 = torch._C._nn.linear(
            layer_norm_74,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_qkv_parameters_weight_,
            None,
        )
        layer_norm_74 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_qkv_parameters_weight_ = (None)
        reshape_43 = linear_80.reshape(1, 49, 3, 16, 32)
        linear_80 = None
        qkv_20 = reshape_43.permute(2, 0, 3, 1, 4)
        reshape_43 = None
        unbind_20 = qkv_20.unbind(0)
        qkv_20 = None
        q_20 = unbind_20[0]
        k_20 = unbind_20[1]
        v_20 = unbind_20[2]
        unbind_20 = None
        x_494 = torch._C._nn.scaled_dot_product_attention(
            q_20, k_20, v_20, dropout_p=0.0
        )
        q_20 = k_20 = v_20 = None
        transpose_23 = x_494.transpose(1, 2)
        x_494 = None
        x_495 = transpose_23.reshape(1, 49, 512)
        transpose_23 = None
        x_496 = torch._C._nn.linear(
            x_495,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_proj_parameters_weight_,
            None,
        )
        x_495 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_proj_parameters_weight_ = (None)
        x_497 = torch.nn.functional.dropout(x_496, 0.0, False, False)
        x_496 = None
        x_498 = mul_90 + x_497
        mul_90 = x_497 = None
        view_41 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale2_parameters_scale_.view(
            (512,)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_91 = x_498 * view_41
        view_41 = None
        layer_norm_75 = torch.nn.functional.layer_norm(
            x_498,
            (512,),
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_498 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (None)
        x_499 = torch._C._nn.linear(
            layer_norm_75,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        layer_norm_75 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_50 = torch.nn.functional.relu(x_499, inplace=False)
        x_499 = None
        pow_51 = relu_50**2
        relu_50 = None
        mul_92 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_
            * pow_51
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_ = (
            pow_51
        ) = None
        x_500 = (
            mul_92
            + l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_
        )
        mul_92 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_ = (None)
        x_501 = torch.nn.functional.dropout(x_500, 0.0, False, False)
        x_500 = None
        x_502 = torch._C._nn.linear(
            x_501,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_501 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_503 = torch.nn.functional.dropout(x_502, 0.0, False, False)
        x_502 = None
        x_504 = mul_91 + x_503
        mul_91 = x_503 = None
        transpose_24 = x_504.transpose(1, 2)
        x_504 = None
        x_505 = transpose_24.reshape(1, 512, 7, 7)
        transpose_24 = None
        x_506 = torch.nn.functional.adaptive_avg_pool2d(x_505, 1)
        x_505 = None
        x_507 = x_506.permute(0, 2, 3, 1)
        x_506 = None
        x_508 = torch.nn.functional.layer_norm(
            x_507,
            (512,),
            l_self_modules_head_modules_norm_parameters_weight_,
            l_self_modules_head_modules_norm_parameters_bias_,
            1e-06,
        )
        x_507 = (
            l_self_modules_head_modules_norm_parameters_weight_
        ) = l_self_modules_head_modules_norm_parameters_bias_ = None
        x_509 = x_508.permute(0, 3, 1, 2)
        x_508 = None
        x_510 = x_509.flatten(1, -1)
        x_509 = None
        x_511 = torch.nn.functional.dropout(x_510, 0.0, False, False)
        x_510 = None
        x_512 = torch._C._nn.linear(
            x_511,
            l_self_modules_head_modules_fc_modules_fc1_parameters_weight_,
            l_self_modules_head_modules_fc_modules_fc1_parameters_bias_,
        )
        x_511 = (
            l_self_modules_head_modules_fc_modules_fc1_parameters_weight_
        ) = l_self_modules_head_modules_fc_modules_fc1_parameters_bias_ = None
        relu_51 = torch.nn.functional.relu(x_512, inplace=False)
        x_512 = None
        x_513 = torch.square(relu_51)
        relu_51 = None
        x_514 = torch.nn.functional.layer_norm(
            x_513,
            (2048,),
            l_self_modules_head_modules_fc_modules_norm_parameters_weight_,
            l_self_modules_head_modules_fc_modules_norm_parameters_bias_,
            1e-06,
        )
        x_513 = (
            l_self_modules_head_modules_fc_modules_norm_parameters_weight_
        ) = l_self_modules_head_modules_fc_modules_norm_parameters_bias_ = None
        x_515 = torch.nn.functional.dropout(x_514, 0.0, False, False)
        x_514 = None
        x_516 = torch._C._nn.linear(
            x_515,
            l_self_modules_head_modules_fc_modules_fc2_parameters_weight_,
            l_self_modules_head_modules_fc_modules_fc2_parameters_bias_,
        )
        x_515 = (
            l_self_modules_head_modules_fc_modules_fc2_parameters_weight_
        ) = l_self_modules_head_modules_fc_modules_fc2_parameters_bias_ = None
        return (x_516,)
