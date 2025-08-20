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
        L_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_6_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_3_modules_3_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_3_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_3_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_3_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_3_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_3_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_3_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_3_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_3_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_3_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_3_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_3_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_3_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_4_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_4_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_4_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_4_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_4_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_4_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_4_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_4_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_4_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_4_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_4_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_4_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_5_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_5_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_5_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_5_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_5_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_5_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_5_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_5_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_5_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_5_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_5_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_5_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_5_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_6_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_6_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_6_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_6_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_6_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_6_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_6_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_6_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_6_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_6_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_6_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_6_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_6_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_7_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_7_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_7_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_7_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_7_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_7_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_7_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_7_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_7_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_7_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_7_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_7_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_7_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_8_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_8_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_8_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_8_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_8_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_8_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_8_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_8_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_8_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_8_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_8_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_8_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_8_modules_attn_last_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        l_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_0_modules_1_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_0_modules_1_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_1_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_1_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_attn_last_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_0_modules_2_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_0_modules_2_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_2_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_2_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_attn_last_modules_conv_parameters_weight_
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
        l_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_2_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_2_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_2_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_2_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_attn_last_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_3_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_3_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_3_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_3_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_attn_last_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_4_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_4_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_4_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_4_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_4_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_4_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_4_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_4_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_4_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_4_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_4_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_4_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_4_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_attn_last_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_5_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_5_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_5_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_5_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_5_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_5_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_5_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_5_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_5_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_5_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_5_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_5_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_5_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_5_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_5_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_5_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_5_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_5_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_5_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_5_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_5_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_5_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_5_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_5_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_attn_last_modules_conv_parameters_weight_
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
        l_self_modules_stages_modules_2_modules_6_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_6_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_6_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_6_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_6_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_6_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_6_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_6_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_6_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_6_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_6_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_6_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_6_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_6_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_attn_last_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_7_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_7_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_7_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_7_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_7_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_7_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_7_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_7_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_7_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_7_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_7_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_7_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_7_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_7_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_7_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_7_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_7_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_7_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_7_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_7_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_7_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_7_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_7_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_7_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_attn_last_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_8_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_8_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_8_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_8_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_8_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_8_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_8_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_8_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_8_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_8_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_8_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_8_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_8_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_attn_last_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_9_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_9_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_9_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_9_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_9_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_9_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_9_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_9_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_9_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_9_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_9_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_9_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_9_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_9_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_9_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_9_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_9_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_9_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_9_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_9_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_9_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_9_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_9_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_9_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_attn_last_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_10_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_10_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_10_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_10_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_10_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_10_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_10_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_10_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_10_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_10_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_10_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_10_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_10_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_attn_last_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_11_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_11_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_11_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_11_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_11_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_11_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_11_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_11_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_11_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_11_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_11_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_11_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_11_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_11_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_11_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_11_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_11_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_11_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_11_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_11_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_11_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_11_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_11_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_11_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_attn_last_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_12_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_12_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_12_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_12_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_12_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_12_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_12_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_12_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_12_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_12_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_12_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_12_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_12_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_attn_last_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_13_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_13_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_13_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_13_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_13_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_13_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_13_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_13_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_13_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_13_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_13_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_13_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_13_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_13_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_13_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_13_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_13_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_13_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_13_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_13_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_13_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_13_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_13_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_13_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_attn_last_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_14_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_14_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_14_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_14_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_14_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_14_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_14_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_14_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_14_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_14_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_14_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_14_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_14_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_attn_last_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_15_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_15_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_15_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_15_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_15_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_15_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_15_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_15_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_15_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_15_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_15_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_15_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_15_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_15_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_15_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_15_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_15_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_15_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_15_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_15_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_15_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_15_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_15_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_15_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_attn_last_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_16_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_16_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_16_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_16_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_16_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_16_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_16_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_16_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_16_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_16_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_16_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_16_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_16_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_16_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_16_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_16_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_16_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_16_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_16_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_16_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_16_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_16_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_16_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_16_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_16_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_16_modules_attn_last_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_17_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_17_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_17_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_17_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_17_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_17_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_17_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_17_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_17_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_17_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_17_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_17_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_17_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_17_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_17_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_17_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_17_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_17_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_17_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_17_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_17_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_17_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_17_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_17_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_17_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_17_modules_attn_last_modules_conv_parameters_weight_
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
        l_self_modules_stages_modules_3_modules_3_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_3_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_3_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_3_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_3_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_3_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_3_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_3_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_3_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_3_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_3_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_3_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_3_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_3_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_3_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_3_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_3_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_3_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_3_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_3_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_3_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_3_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_3_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_3_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_3_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_3_modules_attn_last_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_4_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_4_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_4_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_4_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_4_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_4_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_4_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_4_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_4_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_4_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_4_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_4_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_4_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_4_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_4_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_4_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_4_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_4_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_4_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_4_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_4_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_4_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_4_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_4_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_4_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_4_modules_attn_last_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_5_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_5_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_5_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_5_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_5_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_5_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_5_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_5_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_5_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_5_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_5_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_5_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_5_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_5_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_5_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_5_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_5_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_5_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_5_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_5_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_5_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_5_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_5_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_5_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_5_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_5_modules_attn_last_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_6_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_6_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_6_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_6_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_6_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_6_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_6_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_6_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_6_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_6_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_6_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_6_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_6_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_6_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_6_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_6_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_6_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_6_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_6_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_6_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_6_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_6_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_6_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_6_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_6_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_6_modules_attn_last_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_7_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_7_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_7_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_7_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_7_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_7_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_7_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_7_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_7_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_7_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_7_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_7_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_7_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_7_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_7_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_7_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_7_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_7_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_7_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_7_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_7_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_7_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_7_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_7_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_7_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_7_modules_attn_last_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_8_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_8_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_8_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_8_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_8_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_8_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_8_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_8_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_8_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_8_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_8_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_8_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_8_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_8_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_8_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_8_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_8_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_8_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_8_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_8_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_8_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_8_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_8_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_8_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_8_modules_attn_last_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_8_modules_attn_last_modules_conv_parameters_weight_
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
        out_7 = silu_7 * 0.9805806756909201
        silu_7 = None
        reshape_9 = l_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_weight_.reshape(
            1, 64, -1
        )
        mul_14 = (
            l_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_gain_
            * 0.11175808310508728
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_gain_ = None
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
            l_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_weight_
        )
        batch_norm_9 = (
            l_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_weight_
        ) = None
        out_8 = torch.conv2d(
            out_7,
            weight_9,
            l_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_7 = (
            weight_9
        ) = (
            l_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_bias_
        ) = None
        silu_8 = torch.nn.functional.silu(out_8, inplace=True)
        out_8 = None
        reshape_10 = l_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_weight_.reshape(
            1, 64, -1
        )
        mul_15 = (
            l_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_gain_ = None
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
            l_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_weight_
        )
        batch_norm_10 = (
            l_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_weight_
        ) = None
        out_9 = torch.conv2d(
            silu_8,
            weight_10,
            l_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_8 = (
            weight_10
        ) = (
            l_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_bias_
        ) = None
        silu_9 = torch.nn.functional.silu(out_9, inplace=True)
        out_9 = None
        reshape_11 = l_self_modules_stages_modules_0_modules_1_modules_conv2b_parameters_weight_.reshape(
            1, 64, -1
        )
        mul_16 = (
            l_self_modules_stages_modules_0_modules_1_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv2b_parameters_gain_ = None
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
            l_self_modules_stages_modules_0_modules_1_modules_conv2b_parameters_weight_
        )
        batch_norm_11 = (
            l_self_modules_stages_modules_0_modules_1_modules_conv2b_parameters_weight_
        ) = None
        out_10 = torch.conv2d(
            silu_9,
            weight_11,
            l_self_modules_stages_modules_0_modules_1_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_9 = (
            weight_11
        ) = (
            l_self_modules_stages_modules_0_modules_1_modules_conv2b_parameters_bias_
        ) = None
        silu_10 = torch.nn.functional.silu(out_10, inplace=False)
        out_10 = None
        reshape_12 = l_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_17 = (
            l_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_gain_
            * 0.22351616621017456
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_gain_ = None
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
            l_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_weight_
        )
        batch_norm_12 = (
            l_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_weight_
        ) = None
        out_11 = torch.conv2d(
            silu_10,
            weight_12,
            l_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_10 = (
            weight_12
        ) = (
            l_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_bias_
        ) = None
        mean_1 = out_11.mean((2, 3))
        y_3 = mean_1.view(1, 1, -1)
        mean_1 = None
        y_4 = torch.conv1d(
            y_3,
            l_self_modules_stages_modules_0_modules_1_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_3 = l_self_modules_stages_modules_0_modules_1_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_1 = y_4.sigmoid()
        y_4 = None
        y_5 = sigmoid_1.view(1, -1, 1, 1)
        sigmoid_1 = None
        expand_as_1 = y_5.expand_as(out_11)
        y_5 = None
        mul_18 = out_11 * expand_as_1
        out_11 = expand_as_1 = None
        out_12 = 2.0 * mul_18
        mul_18 = None
        mul_20 = out_12 * 0.2
        out_12 = None
        out_13 = mul_20 + out_6
        mul_20 = out_6 = None
        silu_11 = torch.nn.functional.silu(out_13, inplace=False)
        out_14 = silu_11 * 0.9622504486493761
        silu_11 = None
        reshape_13 = l_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_weight_.reshape(
            1, 64, -1
        )
        mul_22 = (
            l_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_gain_
            * 0.11175808310508728
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_gain_ = None
        view_17 = mul_22.view(-1)
        mul_22 = None
        batch_norm_13 = torch.nn.functional.batch_norm(
            reshape_13,
            None,
            None,
            weight=view_17,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_13 = view_17 = None
        weight_13 = batch_norm_13.reshape_as(
            l_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_weight_
        )
        batch_norm_13 = (
            l_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_weight_
        ) = None
        out_15 = torch.conv2d(
            out_14,
            weight_13,
            l_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_14 = (
            weight_13
        ) = (
            l_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_bias_
        ) = None
        silu_12 = torch.nn.functional.silu(out_15, inplace=True)
        out_15 = None
        reshape_14 = l_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_weight_.reshape(
            1, 64, -1
        )
        mul_23 = (
            l_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_gain_ = None
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
            l_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_weight_
        )
        batch_norm_14 = (
            l_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_weight_
        ) = None
        out_16 = torch.conv2d(
            silu_12,
            weight_14,
            l_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_12 = (
            weight_14
        ) = (
            l_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_bias_
        ) = None
        silu_13 = torch.nn.functional.silu(out_16, inplace=True)
        out_16 = None
        reshape_15 = l_self_modules_stages_modules_0_modules_2_modules_conv2b_parameters_weight_.reshape(
            1, 64, -1
        )
        mul_24 = (
            l_self_modules_stages_modules_0_modules_2_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv2b_parameters_gain_ = None
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
            l_self_modules_stages_modules_0_modules_2_modules_conv2b_parameters_weight_
        )
        batch_norm_15 = (
            l_self_modules_stages_modules_0_modules_2_modules_conv2b_parameters_weight_
        ) = None
        out_17 = torch.conv2d(
            silu_13,
            weight_15,
            l_self_modules_stages_modules_0_modules_2_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_13 = (
            weight_15
        ) = (
            l_self_modules_stages_modules_0_modules_2_modules_conv2b_parameters_bias_
        ) = None
        silu_14 = torch.nn.functional.silu(out_17, inplace=False)
        out_17 = None
        reshape_16 = l_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_25 = (
            l_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_gain_
            * 0.22351616621017456
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_gain_ = None
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
            l_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_weight_
        )
        batch_norm_16 = (
            l_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_weight_
        ) = None
        out_18 = torch.conv2d(
            silu_14,
            weight_16,
            l_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_14 = (
            weight_16
        ) = (
            l_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_bias_
        ) = None
        mean_2 = out_18.mean((2, 3))
        y_6 = mean_2.view(1, 1, -1)
        mean_2 = None
        y_7 = torch.conv1d(
            y_6,
            l_self_modules_stages_modules_0_modules_2_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_6 = l_self_modules_stages_modules_0_modules_2_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_2 = y_7.sigmoid()
        y_7 = None
        y_8 = sigmoid_2.view(1, -1, 1, 1)
        sigmoid_2 = None
        expand_as_2 = y_8.expand_as(out_18)
        y_8 = None
        mul_26 = out_18 * expand_as_2
        out_18 = expand_as_2 = None
        out_19 = 2.0 * mul_26
        mul_26 = None
        mul_28 = out_19 * 0.2
        out_19 = None
        out_20 = mul_28 + out_13
        mul_28 = out_13 = None
        silu_15 = torch.nn.functional.silu(out_20, inplace=False)
        out_20 = None
        out_21 = silu_15 * 0.9449111825230679
        silu_15 = None
        avg_pool2d = torch._C._nn.avg_pool2d(out_21, 2, 2, 0, True, False, None)
        reshape_17 = l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_30 = (
            l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_gain_
            * 0.11175808310508728
        )
        l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_gain_ = (
            None
        )
        view_23 = mul_30.view(-1)
        mul_30 = None
        batch_norm_17 = torch.nn.functional.batch_norm(
            reshape_17,
            None,
            None,
            weight=view_23,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_17 = view_23 = None
        weight_17 = batch_norm_17.reshape_as(
            l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_
        )
        batch_norm_17 = l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        shortcut_1 = torch.conv2d(
            avg_pool2d,
            weight_17,
            l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        avg_pool2d = (
            weight_17
        ) = l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_bias_ = (None)
        reshape_18 = l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_31 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_gain_
            * 0.11175808310508728
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_gain_ = None
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
            l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_weight_
        )
        batch_norm_18 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_weight_
        ) = None
        out_22 = torch.conv2d(
            out_21,
            weight_18,
            l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_21 = (
            weight_18
        ) = (
            l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_bias_
        ) = None
        silu_16 = torch.nn.functional.silu(out_22, inplace=True)
        out_22 = None
        reshape_19 = l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_32 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_gain_ = None
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
            l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_weight_
        )
        batch_norm_19 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_weight_
        ) = None
        out_23 = torch.conv2d(
            silu_16,
            weight_19,
            l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            2,
        )
        silu_16 = (
            weight_19
        ) = (
            l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_bias_
        ) = None
        silu_17 = torch.nn.functional.silu(out_23, inplace=True)
        out_23 = None
        reshape_20 = l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_33 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_gain_ = None
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
            l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_weight_
        )
        batch_norm_20 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_weight_
        ) = None
        out_24 = torch.conv2d(
            silu_17,
            weight_20,
            l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        silu_17 = (
            weight_20
        ) = (
            l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_bias_
        ) = None
        silu_18 = torch.nn.functional.silu(out_24, inplace=False)
        out_24 = None
        reshape_21 = l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_34 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_gain_
            * 0.1580497968320339
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_gain_ = None
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
            l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_weight_
        )
        batch_norm_21 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_weight_
        ) = None
        out_25 = torch.conv2d(
            silu_18,
            weight_21,
            l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_18 = (
            weight_21
        ) = (
            l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_bias_
        ) = None
        mean_3 = out_25.mean((2, 3))
        y_9 = mean_3.view(1, 1, -1)
        mean_3 = None
        y_10 = torch.conv1d(
            y_9,
            l_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_9 = l_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_3 = y_10.sigmoid()
        y_10 = None
        y_11 = sigmoid_3.view(1, -1, 1, 1)
        sigmoid_3 = None
        expand_as_3 = y_11.expand_as(out_25)
        y_11 = None
        mul_35 = out_25 * expand_as_3
        out_25 = expand_as_3 = None
        out_26 = 2.0 * mul_35
        mul_35 = None
        mul_37 = out_26 * 0.2
        out_26 = None
        out_27 = mul_37 + shortcut_1
        mul_37 = shortcut_1 = None
        silu_19 = torch.nn.functional.silu(out_27, inplace=False)
        out_28 = silu_19 * 0.9805806756909201
        silu_19 = None
        reshape_22 = l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_39 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_gain_
            * 0.07902489841601695
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_gain_ = None
        view_30 = mul_39.view(-1)
        mul_39 = None
        batch_norm_22 = torch.nn.functional.batch_norm(
            reshape_22,
            None,
            None,
            weight=view_30,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_22 = view_30 = None
        weight_22 = batch_norm_22.reshape_as(
            l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_weight_
        )
        batch_norm_22 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_weight_
        ) = None
        out_29 = torch.conv2d(
            out_28,
            weight_22,
            l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_28 = (
            weight_22
        ) = (
            l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_bias_
        ) = None
        silu_20 = torch.nn.functional.silu(out_29, inplace=True)
        out_29 = None
        reshape_23 = l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_40 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_gain_ = None
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
            l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_weight_
        )
        batch_norm_23 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_weight_
        ) = None
        out_30 = torch.conv2d(
            silu_20,
            weight_23,
            l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        silu_20 = (
            weight_23
        ) = (
            l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_bias_
        ) = None
        silu_21 = torch.nn.functional.silu(out_30, inplace=True)
        out_30 = None
        reshape_24 = l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_41 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_gain_ = None
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
            l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_weight_
        )
        batch_norm_24 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_weight_
        ) = None
        out_31 = torch.conv2d(
            silu_21,
            weight_24,
            l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        silu_21 = (
            weight_24
        ) = (
            l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_bias_
        ) = None
        silu_22 = torch.nn.functional.silu(out_31, inplace=False)
        out_31 = None
        reshape_25 = l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_42 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_gain_
            * 0.1580497968320339
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_gain_ = None
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
            l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_weight_
        )
        batch_norm_25 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_weight_
        ) = None
        out_32 = torch.conv2d(
            silu_22,
            weight_25,
            l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_22 = (
            weight_25
        ) = (
            l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_bias_
        ) = None
        mean_4 = out_32.mean((2, 3))
        y_12 = mean_4.view(1, 1, -1)
        mean_4 = None
        y_13 = torch.conv1d(
            y_12,
            l_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_12 = l_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_4 = y_13.sigmoid()
        y_13 = None
        y_14 = sigmoid_4.view(1, -1, 1, 1)
        sigmoid_4 = None
        expand_as_4 = y_14.expand_as(out_32)
        y_14 = None
        mul_43 = out_32 * expand_as_4
        out_32 = expand_as_4 = None
        out_33 = 2.0 * mul_43
        mul_43 = None
        mul_45 = out_33 * 0.2
        out_33 = None
        out_34 = mul_45 + out_27
        mul_45 = out_27 = None
        silu_23 = torch.nn.functional.silu(out_34, inplace=False)
        out_35 = silu_23 * 0.9622504486493761
        silu_23 = None
        reshape_26 = l_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_47 = (
            l_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_gain_
            * 0.07902489841601695
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_gain_ = None
        view_36 = mul_47.view(-1)
        mul_47 = None
        batch_norm_26 = torch.nn.functional.batch_norm(
            reshape_26,
            None,
            None,
            weight=view_36,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_26 = view_36 = None
        weight_26 = batch_norm_26.reshape_as(
            l_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_weight_
        )
        batch_norm_26 = (
            l_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_weight_
        ) = None
        out_36 = torch.conv2d(
            out_35,
            weight_26,
            l_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_35 = (
            weight_26
        ) = (
            l_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_bias_
        ) = None
        silu_24 = torch.nn.functional.silu(out_36, inplace=True)
        out_36 = None
        reshape_27 = l_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_48 = (
            l_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_gain_ = None
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
            l_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_weight_
        )
        batch_norm_27 = (
            l_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_weight_
        ) = None
        out_37 = torch.conv2d(
            silu_24,
            weight_27,
            l_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        silu_24 = (
            weight_27
        ) = (
            l_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_bias_
        ) = None
        silu_25 = torch.nn.functional.silu(out_37, inplace=True)
        out_37 = None
        reshape_28 = l_self_modules_stages_modules_1_modules_2_modules_conv2b_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_49 = (
            l_self_modules_stages_modules_1_modules_2_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv2b_parameters_gain_ = None
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
            l_self_modules_stages_modules_1_modules_2_modules_conv2b_parameters_weight_
        )
        batch_norm_28 = (
            l_self_modules_stages_modules_1_modules_2_modules_conv2b_parameters_weight_
        ) = None
        out_38 = torch.conv2d(
            silu_25,
            weight_28,
            l_self_modules_stages_modules_1_modules_2_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        silu_25 = (
            weight_28
        ) = (
            l_self_modules_stages_modules_1_modules_2_modules_conv2b_parameters_bias_
        ) = None
        silu_26 = torch.nn.functional.silu(out_38, inplace=False)
        out_38 = None
        reshape_29 = l_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_50 = (
            l_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_gain_
            * 0.1580497968320339
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_gain_ = None
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
            l_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_weight_
        )
        batch_norm_29 = (
            l_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_weight_
        ) = None
        out_39 = torch.conv2d(
            silu_26,
            weight_29,
            l_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_26 = (
            weight_29
        ) = (
            l_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_bias_
        ) = None
        mean_5 = out_39.mean((2, 3))
        y_15 = mean_5.view(1, 1, -1)
        mean_5 = None
        y_16 = torch.conv1d(
            y_15,
            l_self_modules_stages_modules_1_modules_2_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_15 = l_self_modules_stages_modules_1_modules_2_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_5 = y_16.sigmoid()
        y_16 = None
        y_17 = sigmoid_5.view(1, -1, 1, 1)
        sigmoid_5 = None
        expand_as_5 = y_17.expand_as(out_39)
        y_17 = None
        mul_51 = out_39 * expand_as_5
        out_39 = expand_as_5 = None
        out_40 = 2.0 * mul_51
        mul_51 = None
        mul_53 = out_40 * 0.2
        out_40 = None
        out_41 = mul_53 + out_34
        mul_53 = out_34 = None
        silu_27 = torch.nn.functional.silu(out_41, inplace=False)
        out_42 = silu_27 * 0.9449111825230679
        silu_27 = None
        reshape_30 = l_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_55 = (
            l_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_gain_
            * 0.07902489841601695
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_gain_ = None
        view_42 = mul_55.view(-1)
        mul_55 = None
        batch_norm_30 = torch.nn.functional.batch_norm(
            reshape_30,
            None,
            None,
            weight=view_42,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_30 = view_42 = None
        weight_30 = batch_norm_30.reshape_as(
            l_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_weight_
        )
        batch_norm_30 = (
            l_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_weight_
        ) = None
        out_43 = torch.conv2d(
            out_42,
            weight_30,
            l_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_42 = (
            weight_30
        ) = (
            l_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_bias_
        ) = None
        silu_28 = torch.nn.functional.silu(out_43, inplace=True)
        out_43 = None
        reshape_31 = l_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_56 = (
            l_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_gain_ = None
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
            l_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_weight_
        )
        batch_norm_31 = (
            l_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_weight_
        ) = None
        out_44 = torch.conv2d(
            silu_28,
            weight_31,
            l_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        silu_28 = (
            weight_31
        ) = (
            l_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_bias_
        ) = None
        silu_29 = torch.nn.functional.silu(out_44, inplace=True)
        out_44 = None
        reshape_32 = l_self_modules_stages_modules_1_modules_3_modules_conv2b_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_57 = (
            l_self_modules_stages_modules_1_modules_3_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv2b_parameters_gain_ = None
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
            l_self_modules_stages_modules_1_modules_3_modules_conv2b_parameters_weight_
        )
        batch_norm_32 = (
            l_self_modules_stages_modules_1_modules_3_modules_conv2b_parameters_weight_
        ) = None
        out_45 = torch.conv2d(
            silu_29,
            weight_32,
            l_self_modules_stages_modules_1_modules_3_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        silu_29 = (
            weight_32
        ) = (
            l_self_modules_stages_modules_1_modules_3_modules_conv2b_parameters_bias_
        ) = None
        silu_30 = torch.nn.functional.silu(out_45, inplace=False)
        out_45 = None
        reshape_33 = l_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_58 = (
            l_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_gain_
            * 0.1580497968320339
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_gain_ = None
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
            l_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_weight_
        )
        batch_norm_33 = (
            l_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_weight_
        ) = None
        out_46 = torch.conv2d(
            silu_30,
            weight_33,
            l_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_30 = (
            weight_33
        ) = (
            l_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_bias_
        ) = None
        mean_6 = out_46.mean((2, 3))
        y_18 = mean_6.view(1, 1, -1)
        mean_6 = None
        y_19 = torch.conv1d(
            y_18,
            l_self_modules_stages_modules_1_modules_3_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_18 = l_self_modules_stages_modules_1_modules_3_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_6 = y_19.sigmoid()
        y_19 = None
        y_20 = sigmoid_6.view(1, -1, 1, 1)
        sigmoid_6 = None
        expand_as_6 = y_20.expand_as(out_46)
        y_20 = None
        mul_59 = out_46 * expand_as_6
        out_46 = expand_as_6 = None
        out_47 = 2.0 * mul_59
        mul_59 = None
        mul_61 = out_47 * 0.2
        out_47 = None
        out_48 = mul_61 + out_41
        mul_61 = out_41 = None
        silu_31 = torch.nn.functional.silu(out_48, inplace=False)
        out_49 = silu_31 * 0.9284766908852592
        silu_31 = None
        reshape_34 = l_self_modules_stages_modules_1_modules_4_modules_conv1_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_63 = (
            l_self_modules_stages_modules_1_modules_4_modules_conv1_parameters_gain_
            * 0.07902489841601695
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv1_parameters_gain_ = None
        view_48 = mul_63.view(-1)
        mul_63 = None
        batch_norm_34 = torch.nn.functional.batch_norm(
            reshape_34,
            None,
            None,
            weight=view_48,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_34 = view_48 = None
        weight_34 = batch_norm_34.reshape_as(
            l_self_modules_stages_modules_1_modules_4_modules_conv1_parameters_weight_
        )
        batch_norm_34 = (
            l_self_modules_stages_modules_1_modules_4_modules_conv1_parameters_weight_
        ) = None
        out_50 = torch.conv2d(
            out_49,
            weight_34,
            l_self_modules_stages_modules_1_modules_4_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_49 = (
            weight_34
        ) = (
            l_self_modules_stages_modules_1_modules_4_modules_conv1_parameters_bias_
        ) = None
        silu_32 = torch.nn.functional.silu(out_50, inplace=True)
        out_50 = None
        reshape_35 = l_self_modules_stages_modules_1_modules_4_modules_conv2_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_64 = (
            l_self_modules_stages_modules_1_modules_4_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv2_parameters_gain_ = None
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
            l_self_modules_stages_modules_1_modules_4_modules_conv2_parameters_weight_
        )
        batch_norm_35 = (
            l_self_modules_stages_modules_1_modules_4_modules_conv2_parameters_weight_
        ) = None
        out_51 = torch.conv2d(
            silu_32,
            weight_35,
            l_self_modules_stages_modules_1_modules_4_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        silu_32 = (
            weight_35
        ) = (
            l_self_modules_stages_modules_1_modules_4_modules_conv2_parameters_bias_
        ) = None
        silu_33 = torch.nn.functional.silu(out_51, inplace=True)
        out_51 = None
        reshape_36 = l_self_modules_stages_modules_1_modules_4_modules_conv2b_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_65 = (
            l_self_modules_stages_modules_1_modules_4_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv2b_parameters_gain_ = None
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
            l_self_modules_stages_modules_1_modules_4_modules_conv2b_parameters_weight_
        )
        batch_norm_36 = (
            l_self_modules_stages_modules_1_modules_4_modules_conv2b_parameters_weight_
        ) = None
        out_52 = torch.conv2d(
            silu_33,
            weight_36,
            l_self_modules_stages_modules_1_modules_4_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        silu_33 = (
            weight_36
        ) = (
            l_self_modules_stages_modules_1_modules_4_modules_conv2b_parameters_bias_
        ) = None
        silu_34 = torch.nn.functional.silu(out_52, inplace=False)
        out_52 = None
        reshape_37 = l_self_modules_stages_modules_1_modules_4_modules_conv3_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_66 = (
            l_self_modules_stages_modules_1_modules_4_modules_conv3_parameters_gain_
            * 0.1580497968320339
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv3_parameters_gain_ = None
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
            l_self_modules_stages_modules_1_modules_4_modules_conv3_parameters_weight_
        )
        batch_norm_37 = (
            l_self_modules_stages_modules_1_modules_4_modules_conv3_parameters_weight_
        ) = None
        out_53 = torch.conv2d(
            silu_34,
            weight_37,
            l_self_modules_stages_modules_1_modules_4_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_34 = (
            weight_37
        ) = (
            l_self_modules_stages_modules_1_modules_4_modules_conv3_parameters_bias_
        ) = None
        mean_7 = out_53.mean((2, 3))
        y_21 = mean_7.view(1, 1, -1)
        mean_7 = None
        y_22 = torch.conv1d(
            y_21,
            l_self_modules_stages_modules_1_modules_4_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_21 = l_self_modules_stages_modules_1_modules_4_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_7 = y_22.sigmoid()
        y_22 = None
        y_23 = sigmoid_7.view(1, -1, 1, 1)
        sigmoid_7 = None
        expand_as_7 = y_23.expand_as(out_53)
        y_23 = None
        mul_67 = out_53 * expand_as_7
        out_53 = expand_as_7 = None
        out_54 = 2.0 * mul_67
        mul_67 = None
        mul_69 = out_54 * 0.2
        out_54 = None
        out_55 = mul_69 + out_48
        mul_69 = out_48 = None
        silu_35 = torch.nn.functional.silu(out_55, inplace=False)
        out_56 = silu_35 * 0.9128709291752768
        silu_35 = None
        reshape_38 = l_self_modules_stages_modules_1_modules_5_modules_conv1_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_71 = (
            l_self_modules_stages_modules_1_modules_5_modules_conv1_parameters_gain_
            * 0.07902489841601695
        )
        l_self_modules_stages_modules_1_modules_5_modules_conv1_parameters_gain_ = None
        view_54 = mul_71.view(-1)
        mul_71 = None
        batch_norm_38 = torch.nn.functional.batch_norm(
            reshape_38,
            None,
            None,
            weight=view_54,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_38 = view_54 = None
        weight_38 = batch_norm_38.reshape_as(
            l_self_modules_stages_modules_1_modules_5_modules_conv1_parameters_weight_
        )
        batch_norm_38 = (
            l_self_modules_stages_modules_1_modules_5_modules_conv1_parameters_weight_
        ) = None
        out_57 = torch.conv2d(
            out_56,
            weight_38,
            l_self_modules_stages_modules_1_modules_5_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_56 = (
            weight_38
        ) = (
            l_self_modules_stages_modules_1_modules_5_modules_conv1_parameters_bias_
        ) = None
        silu_36 = torch.nn.functional.silu(out_57, inplace=True)
        out_57 = None
        reshape_39 = l_self_modules_stages_modules_1_modules_5_modules_conv2_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_72 = (
            l_self_modules_stages_modules_1_modules_5_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_1_modules_5_modules_conv2_parameters_gain_ = None
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
            l_self_modules_stages_modules_1_modules_5_modules_conv2_parameters_weight_
        )
        batch_norm_39 = (
            l_self_modules_stages_modules_1_modules_5_modules_conv2_parameters_weight_
        ) = None
        out_58 = torch.conv2d(
            silu_36,
            weight_39,
            l_self_modules_stages_modules_1_modules_5_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        silu_36 = (
            weight_39
        ) = (
            l_self_modules_stages_modules_1_modules_5_modules_conv2_parameters_bias_
        ) = None
        silu_37 = torch.nn.functional.silu(out_58, inplace=True)
        out_58 = None
        reshape_40 = l_self_modules_stages_modules_1_modules_5_modules_conv2b_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_73 = (
            l_self_modules_stages_modules_1_modules_5_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_1_modules_5_modules_conv2b_parameters_gain_ = None
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
            l_self_modules_stages_modules_1_modules_5_modules_conv2b_parameters_weight_
        )
        batch_norm_40 = (
            l_self_modules_stages_modules_1_modules_5_modules_conv2b_parameters_weight_
        ) = None
        out_59 = torch.conv2d(
            silu_37,
            weight_40,
            l_self_modules_stages_modules_1_modules_5_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        silu_37 = (
            weight_40
        ) = (
            l_self_modules_stages_modules_1_modules_5_modules_conv2b_parameters_bias_
        ) = None
        silu_38 = torch.nn.functional.silu(out_59, inplace=False)
        out_59 = None
        reshape_41 = l_self_modules_stages_modules_1_modules_5_modules_conv3_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_74 = (
            l_self_modules_stages_modules_1_modules_5_modules_conv3_parameters_gain_
            * 0.1580497968320339
        )
        l_self_modules_stages_modules_1_modules_5_modules_conv3_parameters_gain_ = None
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
            l_self_modules_stages_modules_1_modules_5_modules_conv3_parameters_weight_
        )
        batch_norm_41 = (
            l_self_modules_stages_modules_1_modules_5_modules_conv3_parameters_weight_
        ) = None
        out_60 = torch.conv2d(
            silu_38,
            weight_41,
            l_self_modules_stages_modules_1_modules_5_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_38 = (
            weight_41
        ) = (
            l_self_modules_stages_modules_1_modules_5_modules_conv3_parameters_bias_
        ) = None
        mean_8 = out_60.mean((2, 3))
        y_24 = mean_8.view(1, 1, -1)
        mean_8 = None
        y_25 = torch.conv1d(
            y_24,
            l_self_modules_stages_modules_1_modules_5_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_24 = l_self_modules_stages_modules_1_modules_5_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_8 = y_25.sigmoid()
        y_25 = None
        y_26 = sigmoid_8.view(1, -1, 1, 1)
        sigmoid_8 = None
        expand_as_8 = y_26.expand_as(out_60)
        y_26 = None
        mul_75 = out_60 * expand_as_8
        out_60 = expand_as_8 = None
        out_61 = 2.0 * mul_75
        mul_75 = None
        mul_77 = out_61 * 0.2
        out_61 = None
        out_62 = mul_77 + out_55
        mul_77 = out_55 = None
        silu_39 = torch.nn.functional.silu(out_62, inplace=False)
        out_62 = None
        out_63 = silu_39 * 0.8980265101338745
        silu_39 = None
        avg_pool2d_1 = torch._C._nn.avg_pool2d(out_63, 2, 2, 0, True, False, None)
        reshape_42 = l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_79 = (
            l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_gain_
            * 0.07902489841601695
        )
        l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_gain_ = (
            None
        )
        view_60 = mul_79.view(-1)
        mul_79 = None
        batch_norm_42 = torch.nn.functional.batch_norm(
            reshape_42,
            None,
            None,
            weight=view_60,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_42 = view_60 = None
        weight_42 = batch_norm_42.reshape_as(
            l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_
        )
        batch_norm_42 = l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        shortcut_2 = torch.conv2d(
            avg_pool2d_1,
            weight_42,
            l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        avg_pool2d_1 = (
            weight_42
        ) = l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_bias_ = (None)
        reshape_43 = l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_80 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_gain_
            * 0.07902489841601695
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_gain_ = None
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
            l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_weight_
        )
        batch_norm_43 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_weight_
        ) = None
        out_64 = torch.conv2d(
            out_63,
            weight_43,
            l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_63 = (
            weight_43
        ) = (
            l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_bias_
        ) = None
        silu_40 = torch.nn.functional.silu(out_64, inplace=True)
        out_64 = None
        reshape_44 = l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_81 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_gain_ = None
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
            l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_weight_
        )
        batch_norm_44 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_weight_
        ) = None
        out_65 = torch.conv2d(
            silu_40,
            weight_44,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            6,
        )
        silu_40 = (
            weight_44
        ) = (
            l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_bias_
        ) = None
        silu_41 = torch.nn.functional.silu(out_65, inplace=True)
        out_65 = None
        reshape_45 = l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_82 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_gain_ = None
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
            l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_weight_
        )
        batch_norm_45 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_weight_
        ) = None
        out_66 = torch.conv2d(
            silu_41,
            weight_45,
            l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_41 = (
            weight_45
        ) = (
            l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_bias_
        ) = None
        silu_42 = torch.nn.functional.silu(out_66, inplace=False)
        out_66 = None
        reshape_46 = l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_83 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_gain_ = None
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
            l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_weight_
        )
        batch_norm_46 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_weight_
        ) = None
        out_67 = torch.conv2d(
            silu_42,
            weight_46,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_42 = (
            weight_46
        ) = (
            l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_bias_
        ) = None
        mean_9 = out_67.mean((2, 3))
        y_27 = mean_9.view(1, 1, -1)
        mean_9 = None
        y_28 = torch.conv1d(
            y_27,
            l_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_27 = l_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_9 = y_28.sigmoid()
        y_28 = None
        y_29 = sigmoid_9.view(1, -1, 1, 1)
        sigmoid_9 = None
        expand_as_9 = y_29.expand_as(out_67)
        y_29 = None
        mul_84 = out_67 * expand_as_9
        out_67 = expand_as_9 = None
        out_68 = 2.0 * mul_84
        mul_84 = None
        mul_86 = out_68 * 0.2
        out_68 = None
        out_69 = mul_86 + shortcut_2
        mul_86 = shortcut_2 = None
        silu_43 = torch.nn.functional.silu(out_69, inplace=False)
        out_70 = silu_43 * 0.9805806756909201
        silu_43 = None
        reshape_47 = l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_88 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_gain_ = None
        view_67 = mul_88.view(-1)
        mul_88 = None
        batch_norm_47 = torch.nn.functional.batch_norm(
            reshape_47,
            None,
            None,
            weight=view_67,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_47 = view_67 = None
        weight_47 = batch_norm_47.reshape_as(
            l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_weight_
        )
        batch_norm_47 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_weight_
        ) = None
        out_71 = torch.conv2d(
            out_70,
            weight_47,
            l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_70 = (
            weight_47
        ) = (
            l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_bias_
        ) = None
        silu_44 = torch.nn.functional.silu(out_71, inplace=True)
        out_71 = None
        reshape_48 = l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_89 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_gain_ = None
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
            l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_weight_
        )
        batch_norm_48 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_weight_
        ) = None
        out_72 = torch.conv2d(
            silu_44,
            weight_48,
            l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_44 = (
            weight_48
        ) = (
            l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_bias_
        ) = None
        silu_45 = torch.nn.functional.silu(out_72, inplace=True)
        out_72 = None
        reshape_49 = l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_90 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_gain_ = None
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
            l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_weight_
        )
        batch_norm_49 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_weight_
        ) = None
        out_73 = torch.conv2d(
            silu_45,
            weight_49,
            l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_45 = (
            weight_49
        ) = (
            l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_bias_
        ) = None
        silu_46 = torch.nn.functional.silu(out_73, inplace=False)
        out_73 = None
        reshape_50 = l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_91 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_gain_ = None
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
            l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_weight_
        )
        batch_norm_50 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_weight_
        ) = None
        out_74 = torch.conv2d(
            silu_46,
            weight_50,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_46 = (
            weight_50
        ) = (
            l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_bias_
        ) = None
        mean_10 = out_74.mean((2, 3))
        y_30 = mean_10.view(1, 1, -1)
        mean_10 = None
        y_31 = torch.conv1d(
            y_30,
            l_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_30 = l_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_10 = y_31.sigmoid()
        y_31 = None
        y_32 = sigmoid_10.view(1, -1, 1, 1)
        sigmoid_10 = None
        expand_as_10 = y_32.expand_as(out_74)
        y_32 = None
        mul_92 = out_74 * expand_as_10
        out_74 = expand_as_10 = None
        out_75 = 2.0 * mul_92
        mul_92 = None
        mul_94 = out_75 * 0.2
        out_75 = None
        out_76 = mul_94 + out_69
        mul_94 = out_69 = None
        silu_47 = torch.nn.functional.silu(out_76, inplace=False)
        out_77 = silu_47 * 0.9622504486493761
        silu_47 = None
        reshape_51 = l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_96 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_gain_ = None
        view_73 = mul_96.view(-1)
        mul_96 = None
        batch_norm_51 = torch.nn.functional.batch_norm(
            reshape_51,
            None,
            None,
            weight=view_73,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_51 = view_73 = None
        weight_51 = batch_norm_51.reshape_as(
            l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_weight_
        )
        batch_norm_51 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_weight_
        ) = None
        out_78 = torch.conv2d(
            out_77,
            weight_51,
            l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_77 = (
            weight_51
        ) = (
            l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_bias_
        ) = None
        silu_48 = torch.nn.functional.silu(out_78, inplace=True)
        out_78 = None
        reshape_52 = l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_97 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_gain_ = None
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
            l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_weight_
        )
        batch_norm_52 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_weight_
        ) = None
        out_79 = torch.conv2d(
            silu_48,
            weight_52,
            l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_48 = (
            weight_52
        ) = (
            l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_bias_
        ) = None
        silu_49 = torch.nn.functional.silu(out_79, inplace=True)
        out_79 = None
        reshape_53 = l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_98 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_gain_ = None
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
            l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_weight_
        )
        batch_norm_53 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_weight_
        ) = None
        out_80 = torch.conv2d(
            silu_49,
            weight_53,
            l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_49 = (
            weight_53
        ) = (
            l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_bias_
        ) = None
        silu_50 = torch.nn.functional.silu(out_80, inplace=False)
        out_80 = None
        reshape_54 = l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_99 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_gain_ = None
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
            l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_weight_
        )
        batch_norm_54 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_weight_
        ) = None
        out_81 = torch.conv2d(
            silu_50,
            weight_54,
            l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_50 = (
            weight_54
        ) = (
            l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_bias_
        ) = None
        mean_11 = out_81.mean((2, 3))
        y_33 = mean_11.view(1, 1, -1)
        mean_11 = None
        y_34 = torch.conv1d(
            y_33,
            l_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_33 = l_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_11 = y_34.sigmoid()
        y_34 = None
        y_35 = sigmoid_11.view(1, -1, 1, 1)
        sigmoid_11 = None
        expand_as_11 = y_35.expand_as(out_81)
        y_35 = None
        mul_100 = out_81 * expand_as_11
        out_81 = expand_as_11 = None
        out_82 = 2.0 * mul_100
        mul_100 = None
        mul_102 = out_82 * 0.2
        out_82 = None
        out_83 = mul_102 + out_76
        mul_102 = out_76 = None
        silu_51 = torch.nn.functional.silu(out_83, inplace=False)
        out_84 = silu_51 * 0.9449111825230679
        silu_51 = None
        reshape_55 = l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_104 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_gain_ = None
        view_79 = mul_104.view(-1)
        mul_104 = None
        batch_norm_55 = torch.nn.functional.batch_norm(
            reshape_55,
            None,
            None,
            weight=view_79,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_55 = view_79 = None
        weight_55 = batch_norm_55.reshape_as(
            l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_weight_
        )
        batch_norm_55 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_weight_
        ) = None
        out_85 = torch.conv2d(
            out_84,
            weight_55,
            l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_84 = (
            weight_55
        ) = (
            l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_bias_
        ) = None
        silu_52 = torch.nn.functional.silu(out_85, inplace=True)
        out_85 = None
        reshape_56 = l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_105 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_gain_ = None
        view_80 = mul_105.view(-1)
        mul_105 = None
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
            l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_weight_
        )
        batch_norm_56 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_weight_
        ) = None
        out_86 = torch.conv2d(
            silu_52,
            weight_56,
            l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_52 = (
            weight_56
        ) = (
            l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_bias_
        ) = None
        silu_53 = torch.nn.functional.silu(out_86, inplace=True)
        out_86 = None
        reshape_57 = l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_106 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_gain_ = None
        view_81 = mul_106.view(-1)
        mul_106 = None
        batch_norm_57 = torch.nn.functional.batch_norm(
            reshape_57,
            None,
            None,
            weight=view_81,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_57 = view_81 = None
        weight_57 = batch_norm_57.reshape_as(
            l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_weight_
        )
        batch_norm_57 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_weight_
        ) = None
        out_87 = torch.conv2d(
            silu_53,
            weight_57,
            l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_53 = (
            weight_57
        ) = (
            l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_bias_
        ) = None
        silu_54 = torch.nn.functional.silu(out_87, inplace=False)
        out_87 = None
        reshape_58 = l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_107 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_gain_ = None
        view_82 = mul_107.view(-1)
        mul_107 = None
        batch_norm_58 = torch.nn.functional.batch_norm(
            reshape_58,
            None,
            None,
            weight=view_82,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_58 = view_82 = None
        weight_58 = batch_norm_58.reshape_as(
            l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_weight_
        )
        batch_norm_58 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_weight_
        ) = None
        out_88 = torch.conv2d(
            silu_54,
            weight_58,
            l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_54 = (
            weight_58
        ) = (
            l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_bias_
        ) = None
        mean_12 = out_88.mean((2, 3))
        y_36 = mean_12.view(1, 1, -1)
        mean_12 = None
        y_37 = torch.conv1d(
            y_36,
            l_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_36 = l_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_12 = y_37.sigmoid()
        y_37 = None
        y_38 = sigmoid_12.view(1, -1, 1, 1)
        sigmoid_12 = None
        expand_as_12 = y_38.expand_as(out_88)
        y_38 = None
        mul_108 = out_88 * expand_as_12
        out_88 = expand_as_12 = None
        out_89 = 2.0 * mul_108
        mul_108 = None
        mul_110 = out_89 * 0.2
        out_89 = None
        out_90 = mul_110 + out_83
        mul_110 = out_83 = None
        silu_55 = torch.nn.functional.silu(out_90, inplace=False)
        out_91 = silu_55 * 0.9284766908852592
        silu_55 = None
        reshape_59 = l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_112 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_gain_ = None
        view_85 = mul_112.view(-1)
        mul_112 = None
        batch_norm_59 = torch.nn.functional.batch_norm(
            reshape_59,
            None,
            None,
            weight=view_85,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_59 = view_85 = None
        weight_59 = batch_norm_59.reshape_as(
            l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_weight_
        )
        batch_norm_59 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_weight_
        ) = None
        out_92 = torch.conv2d(
            out_91,
            weight_59,
            l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_91 = (
            weight_59
        ) = (
            l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_bias_
        ) = None
        silu_56 = torch.nn.functional.silu(out_92, inplace=True)
        out_92 = None
        reshape_60 = l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_113 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_gain_ = None
        view_86 = mul_113.view(-1)
        mul_113 = None
        batch_norm_60 = torch.nn.functional.batch_norm(
            reshape_60,
            None,
            None,
            weight=view_86,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_60 = view_86 = None
        weight_60 = batch_norm_60.reshape_as(
            l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_weight_
        )
        batch_norm_60 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_weight_
        ) = None
        out_93 = torch.conv2d(
            silu_56,
            weight_60,
            l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_56 = (
            weight_60
        ) = (
            l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_bias_
        ) = None
        silu_57 = torch.nn.functional.silu(out_93, inplace=True)
        out_93 = None
        reshape_61 = l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_114 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_gain_ = None
        view_87 = mul_114.view(-1)
        mul_114 = None
        batch_norm_61 = torch.nn.functional.batch_norm(
            reshape_61,
            None,
            None,
            weight=view_87,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_61 = view_87 = None
        weight_61 = batch_norm_61.reshape_as(
            l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_weight_
        )
        batch_norm_61 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_weight_
        ) = None
        out_94 = torch.conv2d(
            silu_57,
            weight_61,
            l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_57 = (
            weight_61
        ) = (
            l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_bias_
        ) = None
        silu_58 = torch.nn.functional.silu(out_94, inplace=False)
        out_94 = None
        reshape_62 = l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_115 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_gain_ = None
        view_88 = mul_115.view(-1)
        mul_115 = None
        batch_norm_62 = torch.nn.functional.batch_norm(
            reshape_62,
            None,
            None,
            weight=view_88,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_62 = view_88 = None
        weight_62 = batch_norm_62.reshape_as(
            l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_weight_
        )
        batch_norm_62 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_weight_
        ) = None
        out_95 = torch.conv2d(
            silu_58,
            weight_62,
            l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_58 = (
            weight_62
        ) = (
            l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_bias_
        ) = None
        mean_13 = out_95.mean((2, 3))
        y_39 = mean_13.view(1, 1, -1)
        mean_13 = None
        y_40 = torch.conv1d(
            y_39,
            l_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_39 = l_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_13 = y_40.sigmoid()
        y_40 = None
        y_41 = sigmoid_13.view(1, -1, 1, 1)
        sigmoid_13 = None
        expand_as_13 = y_41.expand_as(out_95)
        y_41 = None
        mul_116 = out_95 * expand_as_13
        out_95 = expand_as_13 = None
        out_96 = 2.0 * mul_116
        mul_116 = None
        mul_118 = out_96 * 0.2
        out_96 = None
        out_97 = mul_118 + out_90
        mul_118 = out_90 = None
        silu_59 = torch.nn.functional.silu(out_97, inplace=False)
        out_98 = silu_59 * 0.9128709291752768
        silu_59 = None
        reshape_63 = l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_120 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_gain_ = None
        view_91 = mul_120.view(-1)
        mul_120 = None
        batch_norm_63 = torch.nn.functional.batch_norm(
            reshape_63,
            None,
            None,
            weight=view_91,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_63 = view_91 = None
        weight_63 = batch_norm_63.reshape_as(
            l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_weight_
        )
        batch_norm_63 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_weight_
        ) = None
        out_99 = torch.conv2d(
            out_98,
            weight_63,
            l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_98 = (
            weight_63
        ) = (
            l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_bias_
        ) = None
        silu_60 = torch.nn.functional.silu(out_99, inplace=True)
        out_99 = None
        reshape_64 = l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_121 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_gain_ = None
        view_92 = mul_121.view(-1)
        mul_121 = None
        batch_norm_64 = torch.nn.functional.batch_norm(
            reshape_64,
            None,
            None,
            weight=view_92,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_64 = view_92 = None
        weight_64 = batch_norm_64.reshape_as(
            l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_weight_
        )
        batch_norm_64 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_weight_
        ) = None
        out_100 = torch.conv2d(
            silu_60,
            weight_64,
            l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_60 = (
            weight_64
        ) = (
            l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_bias_
        ) = None
        silu_61 = torch.nn.functional.silu(out_100, inplace=True)
        out_100 = None
        reshape_65 = l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_122 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_gain_ = None
        view_93 = mul_122.view(-1)
        mul_122 = None
        batch_norm_65 = torch.nn.functional.batch_norm(
            reshape_65,
            None,
            None,
            weight=view_93,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_65 = view_93 = None
        weight_65 = batch_norm_65.reshape_as(
            l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_weight_
        )
        batch_norm_65 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_weight_
        ) = None
        out_101 = torch.conv2d(
            silu_61,
            weight_65,
            l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_61 = (
            weight_65
        ) = (
            l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_bias_
        ) = None
        silu_62 = torch.nn.functional.silu(out_101, inplace=False)
        out_101 = None
        reshape_66 = l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_123 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_gain_ = None
        view_94 = mul_123.view(-1)
        mul_123 = None
        batch_norm_66 = torch.nn.functional.batch_norm(
            reshape_66,
            None,
            None,
            weight=view_94,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_66 = view_94 = None
        weight_66 = batch_norm_66.reshape_as(
            l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_weight_
        )
        batch_norm_66 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_weight_
        ) = None
        out_102 = torch.conv2d(
            silu_62,
            weight_66,
            l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_62 = (
            weight_66
        ) = (
            l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_bias_
        ) = None
        mean_14 = out_102.mean((2, 3))
        y_42 = mean_14.view(1, 1, -1)
        mean_14 = None
        y_43 = torch.conv1d(
            y_42,
            l_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_42 = l_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_14 = y_43.sigmoid()
        y_43 = None
        y_44 = sigmoid_14.view(1, -1, 1, 1)
        sigmoid_14 = None
        expand_as_14 = y_44.expand_as(out_102)
        y_44 = None
        mul_124 = out_102 * expand_as_14
        out_102 = expand_as_14 = None
        out_103 = 2.0 * mul_124
        mul_124 = None
        mul_126 = out_103 * 0.2
        out_103 = None
        out_104 = mul_126 + out_97
        mul_126 = out_97 = None
        silu_63 = torch.nn.functional.silu(out_104, inplace=False)
        out_105 = silu_63 * 0.8980265101338745
        silu_63 = None
        reshape_67 = l_self_modules_stages_modules_2_modules_6_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_128 = (
            l_self_modules_stages_modules_2_modules_6_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv1_parameters_gain_ = None
        view_97 = mul_128.view(-1)
        mul_128 = None
        batch_norm_67 = torch.nn.functional.batch_norm(
            reshape_67,
            None,
            None,
            weight=view_97,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_67 = view_97 = None
        weight_67 = batch_norm_67.reshape_as(
            l_self_modules_stages_modules_2_modules_6_modules_conv1_parameters_weight_
        )
        batch_norm_67 = (
            l_self_modules_stages_modules_2_modules_6_modules_conv1_parameters_weight_
        ) = None
        out_106 = torch.conv2d(
            out_105,
            weight_67,
            l_self_modules_stages_modules_2_modules_6_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_105 = (
            weight_67
        ) = (
            l_self_modules_stages_modules_2_modules_6_modules_conv1_parameters_bias_
        ) = None
        silu_64 = torch.nn.functional.silu(out_106, inplace=True)
        out_106 = None
        reshape_68 = l_self_modules_stages_modules_2_modules_6_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_129 = (
            l_self_modules_stages_modules_2_modules_6_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv2_parameters_gain_ = None
        view_98 = mul_129.view(-1)
        mul_129 = None
        batch_norm_68 = torch.nn.functional.batch_norm(
            reshape_68,
            None,
            None,
            weight=view_98,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_68 = view_98 = None
        weight_68 = batch_norm_68.reshape_as(
            l_self_modules_stages_modules_2_modules_6_modules_conv2_parameters_weight_
        )
        batch_norm_68 = (
            l_self_modules_stages_modules_2_modules_6_modules_conv2_parameters_weight_
        ) = None
        out_107 = torch.conv2d(
            silu_64,
            weight_68,
            l_self_modules_stages_modules_2_modules_6_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_64 = (
            weight_68
        ) = (
            l_self_modules_stages_modules_2_modules_6_modules_conv2_parameters_bias_
        ) = None
        silu_65 = torch.nn.functional.silu(out_107, inplace=True)
        out_107 = None
        reshape_69 = l_self_modules_stages_modules_2_modules_6_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_130 = (
            l_self_modules_stages_modules_2_modules_6_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv2b_parameters_gain_ = None
        view_99 = mul_130.view(-1)
        mul_130 = None
        batch_norm_69 = torch.nn.functional.batch_norm(
            reshape_69,
            None,
            None,
            weight=view_99,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_69 = view_99 = None
        weight_69 = batch_norm_69.reshape_as(
            l_self_modules_stages_modules_2_modules_6_modules_conv2b_parameters_weight_
        )
        batch_norm_69 = (
            l_self_modules_stages_modules_2_modules_6_modules_conv2b_parameters_weight_
        ) = None
        out_108 = torch.conv2d(
            silu_65,
            weight_69,
            l_self_modules_stages_modules_2_modules_6_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_65 = (
            weight_69
        ) = (
            l_self_modules_stages_modules_2_modules_6_modules_conv2b_parameters_bias_
        ) = None
        silu_66 = torch.nn.functional.silu(out_108, inplace=False)
        out_108 = None
        reshape_70 = l_self_modules_stages_modules_2_modules_6_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_131 = (
            l_self_modules_stages_modules_2_modules_6_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv3_parameters_gain_ = None
        view_100 = mul_131.view(-1)
        mul_131 = None
        batch_norm_70 = torch.nn.functional.batch_norm(
            reshape_70,
            None,
            None,
            weight=view_100,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_70 = view_100 = None
        weight_70 = batch_norm_70.reshape_as(
            l_self_modules_stages_modules_2_modules_6_modules_conv3_parameters_weight_
        )
        batch_norm_70 = (
            l_self_modules_stages_modules_2_modules_6_modules_conv3_parameters_weight_
        ) = None
        out_109 = torch.conv2d(
            silu_66,
            weight_70,
            l_self_modules_stages_modules_2_modules_6_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_66 = (
            weight_70
        ) = (
            l_self_modules_stages_modules_2_modules_6_modules_conv3_parameters_bias_
        ) = None
        mean_15 = out_109.mean((2, 3))
        y_45 = mean_15.view(1, 1, -1)
        mean_15 = None
        y_46 = torch.conv1d(
            y_45,
            l_self_modules_stages_modules_2_modules_6_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_45 = l_self_modules_stages_modules_2_modules_6_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_15 = y_46.sigmoid()
        y_46 = None
        y_47 = sigmoid_15.view(1, -1, 1, 1)
        sigmoid_15 = None
        expand_as_15 = y_47.expand_as(out_109)
        y_47 = None
        mul_132 = out_109 * expand_as_15
        out_109 = expand_as_15 = None
        out_110 = 2.0 * mul_132
        mul_132 = None
        mul_134 = out_110 * 0.2
        out_110 = None
        out_111 = mul_134 + out_104
        mul_134 = out_104 = None
        silu_67 = torch.nn.functional.silu(out_111, inplace=False)
        out_112 = silu_67 * 0.8838834764831842
        silu_67 = None
        reshape_71 = l_self_modules_stages_modules_2_modules_7_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_136 = (
            l_self_modules_stages_modules_2_modules_7_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_2_modules_7_modules_conv1_parameters_gain_ = None
        view_103 = mul_136.view(-1)
        mul_136 = None
        batch_norm_71 = torch.nn.functional.batch_norm(
            reshape_71,
            None,
            None,
            weight=view_103,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_71 = view_103 = None
        weight_71 = batch_norm_71.reshape_as(
            l_self_modules_stages_modules_2_modules_7_modules_conv1_parameters_weight_
        )
        batch_norm_71 = (
            l_self_modules_stages_modules_2_modules_7_modules_conv1_parameters_weight_
        ) = None
        out_113 = torch.conv2d(
            out_112,
            weight_71,
            l_self_modules_stages_modules_2_modules_7_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_112 = (
            weight_71
        ) = (
            l_self_modules_stages_modules_2_modules_7_modules_conv1_parameters_bias_
        ) = None
        silu_68 = torch.nn.functional.silu(out_113, inplace=True)
        out_113 = None
        reshape_72 = l_self_modules_stages_modules_2_modules_7_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_137 = (
            l_self_modules_stages_modules_2_modules_7_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_7_modules_conv2_parameters_gain_ = None
        view_104 = mul_137.view(-1)
        mul_137 = None
        batch_norm_72 = torch.nn.functional.batch_norm(
            reshape_72,
            None,
            None,
            weight=view_104,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_72 = view_104 = None
        weight_72 = batch_norm_72.reshape_as(
            l_self_modules_stages_modules_2_modules_7_modules_conv2_parameters_weight_
        )
        batch_norm_72 = (
            l_self_modules_stages_modules_2_modules_7_modules_conv2_parameters_weight_
        ) = None
        out_114 = torch.conv2d(
            silu_68,
            weight_72,
            l_self_modules_stages_modules_2_modules_7_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_68 = (
            weight_72
        ) = (
            l_self_modules_stages_modules_2_modules_7_modules_conv2_parameters_bias_
        ) = None
        silu_69 = torch.nn.functional.silu(out_114, inplace=True)
        out_114 = None
        reshape_73 = l_self_modules_stages_modules_2_modules_7_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_138 = (
            l_self_modules_stages_modules_2_modules_7_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_7_modules_conv2b_parameters_gain_ = None
        view_105 = mul_138.view(-1)
        mul_138 = None
        batch_norm_73 = torch.nn.functional.batch_norm(
            reshape_73,
            None,
            None,
            weight=view_105,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_73 = view_105 = None
        weight_73 = batch_norm_73.reshape_as(
            l_self_modules_stages_modules_2_modules_7_modules_conv2b_parameters_weight_
        )
        batch_norm_73 = (
            l_self_modules_stages_modules_2_modules_7_modules_conv2b_parameters_weight_
        ) = None
        out_115 = torch.conv2d(
            silu_69,
            weight_73,
            l_self_modules_stages_modules_2_modules_7_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_69 = (
            weight_73
        ) = (
            l_self_modules_stages_modules_2_modules_7_modules_conv2b_parameters_bias_
        ) = None
        silu_70 = torch.nn.functional.silu(out_115, inplace=False)
        out_115 = None
        reshape_74 = l_self_modules_stages_modules_2_modules_7_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_139 = (
            l_self_modules_stages_modules_2_modules_7_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_2_modules_7_modules_conv3_parameters_gain_ = None
        view_106 = mul_139.view(-1)
        mul_139 = None
        batch_norm_74 = torch.nn.functional.batch_norm(
            reshape_74,
            None,
            None,
            weight=view_106,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_74 = view_106 = None
        weight_74 = batch_norm_74.reshape_as(
            l_self_modules_stages_modules_2_modules_7_modules_conv3_parameters_weight_
        )
        batch_norm_74 = (
            l_self_modules_stages_modules_2_modules_7_modules_conv3_parameters_weight_
        ) = None
        out_116 = torch.conv2d(
            silu_70,
            weight_74,
            l_self_modules_stages_modules_2_modules_7_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_70 = (
            weight_74
        ) = (
            l_self_modules_stages_modules_2_modules_7_modules_conv3_parameters_bias_
        ) = None
        mean_16 = out_116.mean((2, 3))
        y_48 = mean_16.view(1, 1, -1)
        mean_16 = None
        y_49 = torch.conv1d(
            y_48,
            l_self_modules_stages_modules_2_modules_7_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_48 = l_self_modules_stages_modules_2_modules_7_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_16 = y_49.sigmoid()
        y_49 = None
        y_50 = sigmoid_16.view(1, -1, 1, 1)
        sigmoid_16 = None
        expand_as_16 = y_50.expand_as(out_116)
        y_50 = None
        mul_140 = out_116 * expand_as_16
        out_116 = expand_as_16 = None
        out_117 = 2.0 * mul_140
        mul_140 = None
        mul_142 = out_117 * 0.2
        out_117 = None
        out_118 = mul_142 + out_111
        mul_142 = out_111 = None
        silu_71 = torch.nn.functional.silu(out_118, inplace=False)
        out_119 = silu_71 * 0.8703882797784891
        silu_71 = None
        reshape_75 = l_self_modules_stages_modules_2_modules_8_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_144 = (
            l_self_modules_stages_modules_2_modules_8_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv1_parameters_gain_ = None
        view_109 = mul_144.view(-1)
        mul_144 = None
        batch_norm_75 = torch.nn.functional.batch_norm(
            reshape_75,
            None,
            None,
            weight=view_109,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_75 = view_109 = None
        weight_75 = batch_norm_75.reshape_as(
            l_self_modules_stages_modules_2_modules_8_modules_conv1_parameters_weight_
        )
        batch_norm_75 = (
            l_self_modules_stages_modules_2_modules_8_modules_conv1_parameters_weight_
        ) = None
        out_120 = torch.conv2d(
            out_119,
            weight_75,
            l_self_modules_stages_modules_2_modules_8_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_119 = (
            weight_75
        ) = (
            l_self_modules_stages_modules_2_modules_8_modules_conv1_parameters_bias_
        ) = None
        silu_72 = torch.nn.functional.silu(out_120, inplace=True)
        out_120 = None
        reshape_76 = l_self_modules_stages_modules_2_modules_8_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_145 = (
            l_self_modules_stages_modules_2_modules_8_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv2_parameters_gain_ = None
        view_110 = mul_145.view(-1)
        mul_145 = None
        batch_norm_76 = torch.nn.functional.batch_norm(
            reshape_76,
            None,
            None,
            weight=view_110,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_76 = view_110 = None
        weight_76 = batch_norm_76.reshape_as(
            l_self_modules_stages_modules_2_modules_8_modules_conv2_parameters_weight_
        )
        batch_norm_76 = (
            l_self_modules_stages_modules_2_modules_8_modules_conv2_parameters_weight_
        ) = None
        out_121 = torch.conv2d(
            silu_72,
            weight_76,
            l_self_modules_stages_modules_2_modules_8_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_72 = (
            weight_76
        ) = (
            l_self_modules_stages_modules_2_modules_8_modules_conv2_parameters_bias_
        ) = None
        silu_73 = torch.nn.functional.silu(out_121, inplace=True)
        out_121 = None
        reshape_77 = l_self_modules_stages_modules_2_modules_8_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_146 = (
            l_self_modules_stages_modules_2_modules_8_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv2b_parameters_gain_ = None
        view_111 = mul_146.view(-1)
        mul_146 = None
        batch_norm_77 = torch.nn.functional.batch_norm(
            reshape_77,
            None,
            None,
            weight=view_111,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_77 = view_111 = None
        weight_77 = batch_norm_77.reshape_as(
            l_self_modules_stages_modules_2_modules_8_modules_conv2b_parameters_weight_
        )
        batch_norm_77 = (
            l_self_modules_stages_modules_2_modules_8_modules_conv2b_parameters_weight_
        ) = None
        out_122 = torch.conv2d(
            silu_73,
            weight_77,
            l_self_modules_stages_modules_2_modules_8_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_73 = (
            weight_77
        ) = (
            l_self_modules_stages_modules_2_modules_8_modules_conv2b_parameters_bias_
        ) = None
        silu_74 = torch.nn.functional.silu(out_122, inplace=False)
        out_122 = None
        reshape_78 = l_self_modules_stages_modules_2_modules_8_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_147 = (
            l_self_modules_stages_modules_2_modules_8_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv3_parameters_gain_ = None
        view_112 = mul_147.view(-1)
        mul_147 = None
        batch_norm_78 = torch.nn.functional.batch_norm(
            reshape_78,
            None,
            None,
            weight=view_112,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_78 = view_112 = None
        weight_78 = batch_norm_78.reshape_as(
            l_self_modules_stages_modules_2_modules_8_modules_conv3_parameters_weight_
        )
        batch_norm_78 = (
            l_self_modules_stages_modules_2_modules_8_modules_conv3_parameters_weight_
        ) = None
        out_123 = torch.conv2d(
            silu_74,
            weight_78,
            l_self_modules_stages_modules_2_modules_8_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_74 = (
            weight_78
        ) = (
            l_self_modules_stages_modules_2_modules_8_modules_conv3_parameters_bias_
        ) = None
        mean_17 = out_123.mean((2, 3))
        y_51 = mean_17.view(1, 1, -1)
        mean_17 = None
        y_52 = torch.conv1d(
            y_51,
            l_self_modules_stages_modules_2_modules_8_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_51 = l_self_modules_stages_modules_2_modules_8_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_17 = y_52.sigmoid()
        y_52 = None
        y_53 = sigmoid_17.view(1, -1, 1, 1)
        sigmoid_17 = None
        expand_as_17 = y_53.expand_as(out_123)
        y_53 = None
        mul_148 = out_123 * expand_as_17
        out_123 = expand_as_17 = None
        out_124 = 2.0 * mul_148
        mul_148 = None
        mul_150 = out_124 * 0.2
        out_124 = None
        out_125 = mul_150 + out_118
        mul_150 = out_118 = None
        silu_75 = torch.nn.functional.silu(out_125, inplace=False)
        out_126 = silu_75 * 0.8574929257125441
        silu_75 = None
        reshape_79 = l_self_modules_stages_modules_2_modules_9_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_152 = (
            l_self_modules_stages_modules_2_modules_9_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_2_modules_9_modules_conv1_parameters_gain_ = None
        view_115 = mul_152.view(-1)
        mul_152 = None
        batch_norm_79 = torch.nn.functional.batch_norm(
            reshape_79,
            None,
            None,
            weight=view_115,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_79 = view_115 = None
        weight_79 = batch_norm_79.reshape_as(
            l_self_modules_stages_modules_2_modules_9_modules_conv1_parameters_weight_
        )
        batch_norm_79 = (
            l_self_modules_stages_modules_2_modules_9_modules_conv1_parameters_weight_
        ) = None
        out_127 = torch.conv2d(
            out_126,
            weight_79,
            l_self_modules_stages_modules_2_modules_9_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_126 = (
            weight_79
        ) = (
            l_self_modules_stages_modules_2_modules_9_modules_conv1_parameters_bias_
        ) = None
        silu_76 = torch.nn.functional.silu(out_127, inplace=True)
        out_127 = None
        reshape_80 = l_self_modules_stages_modules_2_modules_9_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_153 = (
            l_self_modules_stages_modules_2_modules_9_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_9_modules_conv2_parameters_gain_ = None
        view_116 = mul_153.view(-1)
        mul_153 = None
        batch_norm_80 = torch.nn.functional.batch_norm(
            reshape_80,
            None,
            None,
            weight=view_116,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_80 = view_116 = None
        weight_80 = batch_norm_80.reshape_as(
            l_self_modules_stages_modules_2_modules_9_modules_conv2_parameters_weight_
        )
        batch_norm_80 = (
            l_self_modules_stages_modules_2_modules_9_modules_conv2_parameters_weight_
        ) = None
        out_128 = torch.conv2d(
            silu_76,
            weight_80,
            l_self_modules_stages_modules_2_modules_9_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_76 = (
            weight_80
        ) = (
            l_self_modules_stages_modules_2_modules_9_modules_conv2_parameters_bias_
        ) = None
        silu_77 = torch.nn.functional.silu(out_128, inplace=True)
        out_128 = None
        reshape_81 = l_self_modules_stages_modules_2_modules_9_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_154 = (
            l_self_modules_stages_modules_2_modules_9_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_9_modules_conv2b_parameters_gain_ = None
        view_117 = mul_154.view(-1)
        mul_154 = None
        batch_norm_81 = torch.nn.functional.batch_norm(
            reshape_81,
            None,
            None,
            weight=view_117,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_81 = view_117 = None
        weight_81 = batch_norm_81.reshape_as(
            l_self_modules_stages_modules_2_modules_9_modules_conv2b_parameters_weight_
        )
        batch_norm_81 = (
            l_self_modules_stages_modules_2_modules_9_modules_conv2b_parameters_weight_
        ) = None
        out_129 = torch.conv2d(
            silu_77,
            weight_81,
            l_self_modules_stages_modules_2_modules_9_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_77 = (
            weight_81
        ) = (
            l_self_modules_stages_modules_2_modules_9_modules_conv2b_parameters_bias_
        ) = None
        silu_78 = torch.nn.functional.silu(out_129, inplace=False)
        out_129 = None
        reshape_82 = l_self_modules_stages_modules_2_modules_9_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_155 = (
            l_self_modules_stages_modules_2_modules_9_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_2_modules_9_modules_conv3_parameters_gain_ = None
        view_118 = mul_155.view(-1)
        mul_155 = None
        batch_norm_82 = torch.nn.functional.batch_norm(
            reshape_82,
            None,
            None,
            weight=view_118,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_82 = view_118 = None
        weight_82 = batch_norm_82.reshape_as(
            l_self_modules_stages_modules_2_modules_9_modules_conv3_parameters_weight_
        )
        batch_norm_82 = (
            l_self_modules_stages_modules_2_modules_9_modules_conv3_parameters_weight_
        ) = None
        out_130 = torch.conv2d(
            silu_78,
            weight_82,
            l_self_modules_stages_modules_2_modules_9_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_78 = (
            weight_82
        ) = (
            l_self_modules_stages_modules_2_modules_9_modules_conv3_parameters_bias_
        ) = None
        mean_18 = out_130.mean((2, 3))
        y_54 = mean_18.view(1, 1, -1)
        mean_18 = None
        y_55 = torch.conv1d(
            y_54,
            l_self_modules_stages_modules_2_modules_9_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_54 = l_self_modules_stages_modules_2_modules_9_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_18 = y_55.sigmoid()
        y_55 = None
        y_56 = sigmoid_18.view(1, -1, 1, 1)
        sigmoid_18 = None
        expand_as_18 = y_56.expand_as(out_130)
        y_56 = None
        mul_156 = out_130 * expand_as_18
        out_130 = expand_as_18 = None
        out_131 = 2.0 * mul_156
        mul_156 = None
        mul_158 = out_131 * 0.2
        out_131 = None
        out_132 = mul_158 + out_125
        mul_158 = out_125 = None
        silu_79 = torch.nn.functional.silu(out_132, inplace=False)
        out_133 = silu_79 * 0.8451542547285165
        silu_79 = None
        reshape_83 = l_self_modules_stages_modules_2_modules_10_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_160 = (
            l_self_modules_stages_modules_2_modules_10_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv1_parameters_gain_ = None
        view_121 = mul_160.view(-1)
        mul_160 = None
        batch_norm_83 = torch.nn.functional.batch_norm(
            reshape_83,
            None,
            None,
            weight=view_121,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_83 = view_121 = None
        weight_83 = batch_norm_83.reshape_as(
            l_self_modules_stages_modules_2_modules_10_modules_conv1_parameters_weight_
        )
        batch_norm_83 = (
            l_self_modules_stages_modules_2_modules_10_modules_conv1_parameters_weight_
        ) = None
        out_134 = torch.conv2d(
            out_133,
            weight_83,
            l_self_modules_stages_modules_2_modules_10_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_133 = (
            weight_83
        ) = (
            l_self_modules_stages_modules_2_modules_10_modules_conv1_parameters_bias_
        ) = None
        silu_80 = torch.nn.functional.silu(out_134, inplace=True)
        out_134 = None
        reshape_84 = l_self_modules_stages_modules_2_modules_10_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_161 = (
            l_self_modules_stages_modules_2_modules_10_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv2_parameters_gain_ = None
        view_122 = mul_161.view(-1)
        mul_161 = None
        batch_norm_84 = torch.nn.functional.batch_norm(
            reshape_84,
            None,
            None,
            weight=view_122,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_84 = view_122 = None
        weight_84 = batch_norm_84.reshape_as(
            l_self_modules_stages_modules_2_modules_10_modules_conv2_parameters_weight_
        )
        batch_norm_84 = (
            l_self_modules_stages_modules_2_modules_10_modules_conv2_parameters_weight_
        ) = None
        out_135 = torch.conv2d(
            silu_80,
            weight_84,
            l_self_modules_stages_modules_2_modules_10_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_80 = (
            weight_84
        ) = (
            l_self_modules_stages_modules_2_modules_10_modules_conv2_parameters_bias_
        ) = None
        silu_81 = torch.nn.functional.silu(out_135, inplace=True)
        out_135 = None
        reshape_85 = l_self_modules_stages_modules_2_modules_10_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_162 = (
            l_self_modules_stages_modules_2_modules_10_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv2b_parameters_gain_ = (
            None
        )
        view_123 = mul_162.view(-1)
        mul_162 = None
        batch_norm_85 = torch.nn.functional.batch_norm(
            reshape_85,
            None,
            None,
            weight=view_123,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_85 = view_123 = None
        weight_85 = batch_norm_85.reshape_as(
            l_self_modules_stages_modules_2_modules_10_modules_conv2b_parameters_weight_
        )
        batch_norm_85 = (
            l_self_modules_stages_modules_2_modules_10_modules_conv2b_parameters_weight_
        ) = None
        out_136 = torch.conv2d(
            silu_81,
            weight_85,
            l_self_modules_stages_modules_2_modules_10_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_81 = (
            weight_85
        ) = (
            l_self_modules_stages_modules_2_modules_10_modules_conv2b_parameters_bias_
        ) = None
        silu_82 = torch.nn.functional.silu(out_136, inplace=False)
        out_136 = None
        reshape_86 = l_self_modules_stages_modules_2_modules_10_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_163 = (
            l_self_modules_stages_modules_2_modules_10_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv3_parameters_gain_ = None
        view_124 = mul_163.view(-1)
        mul_163 = None
        batch_norm_86 = torch.nn.functional.batch_norm(
            reshape_86,
            None,
            None,
            weight=view_124,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_86 = view_124 = None
        weight_86 = batch_norm_86.reshape_as(
            l_self_modules_stages_modules_2_modules_10_modules_conv3_parameters_weight_
        )
        batch_norm_86 = (
            l_self_modules_stages_modules_2_modules_10_modules_conv3_parameters_weight_
        ) = None
        out_137 = torch.conv2d(
            silu_82,
            weight_86,
            l_self_modules_stages_modules_2_modules_10_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_82 = (
            weight_86
        ) = (
            l_self_modules_stages_modules_2_modules_10_modules_conv3_parameters_bias_
        ) = None
        mean_19 = out_137.mean((2, 3))
        y_57 = mean_19.view(1, 1, -1)
        mean_19 = None
        y_58 = torch.conv1d(
            y_57,
            l_self_modules_stages_modules_2_modules_10_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_57 = l_self_modules_stages_modules_2_modules_10_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_19 = y_58.sigmoid()
        y_58 = None
        y_59 = sigmoid_19.view(1, -1, 1, 1)
        sigmoid_19 = None
        expand_as_19 = y_59.expand_as(out_137)
        y_59 = None
        mul_164 = out_137 * expand_as_19
        out_137 = expand_as_19 = None
        out_138 = 2.0 * mul_164
        mul_164 = None
        mul_166 = out_138 * 0.2
        out_138 = None
        out_139 = mul_166 + out_132
        mul_166 = out_132 = None
        silu_83 = torch.nn.functional.silu(out_139, inplace=False)
        out_140 = silu_83 * 0.8333333333333333
        silu_83 = None
        reshape_87 = l_self_modules_stages_modules_2_modules_11_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_168 = (
            l_self_modules_stages_modules_2_modules_11_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_2_modules_11_modules_conv1_parameters_gain_ = None
        view_127 = mul_168.view(-1)
        mul_168 = None
        batch_norm_87 = torch.nn.functional.batch_norm(
            reshape_87,
            None,
            None,
            weight=view_127,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_87 = view_127 = None
        weight_87 = batch_norm_87.reshape_as(
            l_self_modules_stages_modules_2_modules_11_modules_conv1_parameters_weight_
        )
        batch_norm_87 = (
            l_self_modules_stages_modules_2_modules_11_modules_conv1_parameters_weight_
        ) = None
        out_141 = torch.conv2d(
            out_140,
            weight_87,
            l_self_modules_stages_modules_2_modules_11_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_140 = (
            weight_87
        ) = (
            l_self_modules_stages_modules_2_modules_11_modules_conv1_parameters_bias_
        ) = None
        silu_84 = torch.nn.functional.silu(out_141, inplace=True)
        out_141 = None
        reshape_88 = l_self_modules_stages_modules_2_modules_11_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_169 = (
            l_self_modules_stages_modules_2_modules_11_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_11_modules_conv2_parameters_gain_ = None
        view_128 = mul_169.view(-1)
        mul_169 = None
        batch_norm_88 = torch.nn.functional.batch_norm(
            reshape_88,
            None,
            None,
            weight=view_128,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_88 = view_128 = None
        weight_88 = batch_norm_88.reshape_as(
            l_self_modules_stages_modules_2_modules_11_modules_conv2_parameters_weight_
        )
        batch_norm_88 = (
            l_self_modules_stages_modules_2_modules_11_modules_conv2_parameters_weight_
        ) = None
        out_142 = torch.conv2d(
            silu_84,
            weight_88,
            l_self_modules_stages_modules_2_modules_11_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_84 = (
            weight_88
        ) = (
            l_self_modules_stages_modules_2_modules_11_modules_conv2_parameters_bias_
        ) = None
        silu_85 = torch.nn.functional.silu(out_142, inplace=True)
        out_142 = None
        reshape_89 = l_self_modules_stages_modules_2_modules_11_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_170 = (
            l_self_modules_stages_modules_2_modules_11_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_11_modules_conv2b_parameters_gain_ = (
            None
        )
        view_129 = mul_170.view(-1)
        mul_170 = None
        batch_norm_89 = torch.nn.functional.batch_norm(
            reshape_89,
            None,
            None,
            weight=view_129,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_89 = view_129 = None
        weight_89 = batch_norm_89.reshape_as(
            l_self_modules_stages_modules_2_modules_11_modules_conv2b_parameters_weight_
        )
        batch_norm_89 = (
            l_self_modules_stages_modules_2_modules_11_modules_conv2b_parameters_weight_
        ) = None
        out_143 = torch.conv2d(
            silu_85,
            weight_89,
            l_self_modules_stages_modules_2_modules_11_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_85 = (
            weight_89
        ) = (
            l_self_modules_stages_modules_2_modules_11_modules_conv2b_parameters_bias_
        ) = None
        silu_86 = torch.nn.functional.silu(out_143, inplace=False)
        out_143 = None
        reshape_90 = l_self_modules_stages_modules_2_modules_11_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_171 = (
            l_self_modules_stages_modules_2_modules_11_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_2_modules_11_modules_conv3_parameters_gain_ = None
        view_130 = mul_171.view(-1)
        mul_171 = None
        batch_norm_90 = torch.nn.functional.batch_norm(
            reshape_90,
            None,
            None,
            weight=view_130,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_90 = view_130 = None
        weight_90 = batch_norm_90.reshape_as(
            l_self_modules_stages_modules_2_modules_11_modules_conv3_parameters_weight_
        )
        batch_norm_90 = (
            l_self_modules_stages_modules_2_modules_11_modules_conv3_parameters_weight_
        ) = None
        out_144 = torch.conv2d(
            silu_86,
            weight_90,
            l_self_modules_stages_modules_2_modules_11_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_86 = (
            weight_90
        ) = (
            l_self_modules_stages_modules_2_modules_11_modules_conv3_parameters_bias_
        ) = None
        mean_20 = out_144.mean((2, 3))
        y_60 = mean_20.view(1, 1, -1)
        mean_20 = None
        y_61 = torch.conv1d(
            y_60,
            l_self_modules_stages_modules_2_modules_11_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_60 = l_self_modules_stages_modules_2_modules_11_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_20 = y_61.sigmoid()
        y_61 = None
        y_62 = sigmoid_20.view(1, -1, 1, 1)
        sigmoid_20 = None
        expand_as_20 = y_62.expand_as(out_144)
        y_62 = None
        mul_172 = out_144 * expand_as_20
        out_144 = expand_as_20 = None
        out_145 = 2.0 * mul_172
        mul_172 = None
        mul_174 = out_145 * 0.2
        out_145 = None
        out_146 = mul_174 + out_139
        mul_174 = out_139 = None
        silu_87 = torch.nn.functional.silu(out_146, inplace=False)
        out_147 = silu_87 * 0.8219949365267863
        silu_87 = None
        reshape_91 = l_self_modules_stages_modules_2_modules_12_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_176 = (
            l_self_modules_stages_modules_2_modules_12_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv1_parameters_gain_ = None
        view_133 = mul_176.view(-1)
        mul_176 = None
        batch_norm_91 = torch.nn.functional.batch_norm(
            reshape_91,
            None,
            None,
            weight=view_133,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_91 = view_133 = None
        weight_91 = batch_norm_91.reshape_as(
            l_self_modules_stages_modules_2_modules_12_modules_conv1_parameters_weight_
        )
        batch_norm_91 = (
            l_self_modules_stages_modules_2_modules_12_modules_conv1_parameters_weight_
        ) = None
        out_148 = torch.conv2d(
            out_147,
            weight_91,
            l_self_modules_stages_modules_2_modules_12_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_147 = (
            weight_91
        ) = (
            l_self_modules_stages_modules_2_modules_12_modules_conv1_parameters_bias_
        ) = None
        silu_88 = torch.nn.functional.silu(out_148, inplace=True)
        out_148 = None
        reshape_92 = l_self_modules_stages_modules_2_modules_12_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_177 = (
            l_self_modules_stages_modules_2_modules_12_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv2_parameters_gain_ = None
        view_134 = mul_177.view(-1)
        mul_177 = None
        batch_norm_92 = torch.nn.functional.batch_norm(
            reshape_92,
            None,
            None,
            weight=view_134,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_92 = view_134 = None
        weight_92 = batch_norm_92.reshape_as(
            l_self_modules_stages_modules_2_modules_12_modules_conv2_parameters_weight_
        )
        batch_norm_92 = (
            l_self_modules_stages_modules_2_modules_12_modules_conv2_parameters_weight_
        ) = None
        out_149 = torch.conv2d(
            silu_88,
            weight_92,
            l_self_modules_stages_modules_2_modules_12_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_88 = (
            weight_92
        ) = (
            l_self_modules_stages_modules_2_modules_12_modules_conv2_parameters_bias_
        ) = None
        silu_89 = torch.nn.functional.silu(out_149, inplace=True)
        out_149 = None
        reshape_93 = l_self_modules_stages_modules_2_modules_12_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_178 = (
            l_self_modules_stages_modules_2_modules_12_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv2b_parameters_gain_ = (
            None
        )
        view_135 = mul_178.view(-1)
        mul_178 = None
        batch_norm_93 = torch.nn.functional.batch_norm(
            reshape_93,
            None,
            None,
            weight=view_135,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_93 = view_135 = None
        weight_93 = batch_norm_93.reshape_as(
            l_self_modules_stages_modules_2_modules_12_modules_conv2b_parameters_weight_
        )
        batch_norm_93 = (
            l_self_modules_stages_modules_2_modules_12_modules_conv2b_parameters_weight_
        ) = None
        out_150 = torch.conv2d(
            silu_89,
            weight_93,
            l_self_modules_stages_modules_2_modules_12_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_89 = (
            weight_93
        ) = (
            l_self_modules_stages_modules_2_modules_12_modules_conv2b_parameters_bias_
        ) = None
        silu_90 = torch.nn.functional.silu(out_150, inplace=False)
        out_150 = None
        reshape_94 = l_self_modules_stages_modules_2_modules_12_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_179 = (
            l_self_modules_stages_modules_2_modules_12_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv3_parameters_gain_ = None
        view_136 = mul_179.view(-1)
        mul_179 = None
        batch_norm_94 = torch.nn.functional.batch_norm(
            reshape_94,
            None,
            None,
            weight=view_136,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_94 = view_136 = None
        weight_94 = batch_norm_94.reshape_as(
            l_self_modules_stages_modules_2_modules_12_modules_conv3_parameters_weight_
        )
        batch_norm_94 = (
            l_self_modules_stages_modules_2_modules_12_modules_conv3_parameters_weight_
        ) = None
        out_151 = torch.conv2d(
            silu_90,
            weight_94,
            l_self_modules_stages_modules_2_modules_12_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_90 = (
            weight_94
        ) = (
            l_self_modules_stages_modules_2_modules_12_modules_conv3_parameters_bias_
        ) = None
        mean_21 = out_151.mean((2, 3))
        y_63 = mean_21.view(1, 1, -1)
        mean_21 = None
        y_64 = torch.conv1d(
            y_63,
            l_self_modules_stages_modules_2_modules_12_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_63 = l_self_modules_stages_modules_2_modules_12_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_21 = y_64.sigmoid()
        y_64 = None
        y_65 = sigmoid_21.view(1, -1, 1, 1)
        sigmoid_21 = None
        expand_as_21 = y_65.expand_as(out_151)
        y_65 = None
        mul_180 = out_151 * expand_as_21
        out_151 = expand_as_21 = None
        out_152 = 2.0 * mul_180
        mul_180 = None
        mul_182 = out_152 * 0.2
        out_152 = None
        out_153 = mul_182 + out_146
        mul_182 = out_146 = None
        silu_91 = torch.nn.functional.silu(out_153, inplace=False)
        out_154 = silu_91 * 0.8111071056538125
        silu_91 = None
        reshape_95 = l_self_modules_stages_modules_2_modules_13_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_184 = (
            l_self_modules_stages_modules_2_modules_13_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_2_modules_13_modules_conv1_parameters_gain_ = None
        view_139 = mul_184.view(-1)
        mul_184 = None
        batch_norm_95 = torch.nn.functional.batch_norm(
            reshape_95,
            None,
            None,
            weight=view_139,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_95 = view_139 = None
        weight_95 = batch_norm_95.reshape_as(
            l_self_modules_stages_modules_2_modules_13_modules_conv1_parameters_weight_
        )
        batch_norm_95 = (
            l_self_modules_stages_modules_2_modules_13_modules_conv1_parameters_weight_
        ) = None
        out_155 = torch.conv2d(
            out_154,
            weight_95,
            l_self_modules_stages_modules_2_modules_13_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_154 = (
            weight_95
        ) = (
            l_self_modules_stages_modules_2_modules_13_modules_conv1_parameters_bias_
        ) = None
        silu_92 = torch.nn.functional.silu(out_155, inplace=True)
        out_155 = None
        reshape_96 = l_self_modules_stages_modules_2_modules_13_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_185 = (
            l_self_modules_stages_modules_2_modules_13_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_13_modules_conv2_parameters_gain_ = None
        view_140 = mul_185.view(-1)
        mul_185 = None
        batch_norm_96 = torch.nn.functional.batch_norm(
            reshape_96,
            None,
            None,
            weight=view_140,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_96 = view_140 = None
        weight_96 = batch_norm_96.reshape_as(
            l_self_modules_stages_modules_2_modules_13_modules_conv2_parameters_weight_
        )
        batch_norm_96 = (
            l_self_modules_stages_modules_2_modules_13_modules_conv2_parameters_weight_
        ) = None
        out_156 = torch.conv2d(
            silu_92,
            weight_96,
            l_self_modules_stages_modules_2_modules_13_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_92 = (
            weight_96
        ) = (
            l_self_modules_stages_modules_2_modules_13_modules_conv2_parameters_bias_
        ) = None
        silu_93 = torch.nn.functional.silu(out_156, inplace=True)
        out_156 = None
        reshape_97 = l_self_modules_stages_modules_2_modules_13_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_186 = (
            l_self_modules_stages_modules_2_modules_13_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_13_modules_conv2b_parameters_gain_ = (
            None
        )
        view_141 = mul_186.view(-1)
        mul_186 = None
        batch_norm_97 = torch.nn.functional.batch_norm(
            reshape_97,
            None,
            None,
            weight=view_141,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_97 = view_141 = None
        weight_97 = batch_norm_97.reshape_as(
            l_self_modules_stages_modules_2_modules_13_modules_conv2b_parameters_weight_
        )
        batch_norm_97 = (
            l_self_modules_stages_modules_2_modules_13_modules_conv2b_parameters_weight_
        ) = None
        out_157 = torch.conv2d(
            silu_93,
            weight_97,
            l_self_modules_stages_modules_2_modules_13_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_93 = (
            weight_97
        ) = (
            l_self_modules_stages_modules_2_modules_13_modules_conv2b_parameters_bias_
        ) = None
        silu_94 = torch.nn.functional.silu(out_157, inplace=False)
        out_157 = None
        reshape_98 = l_self_modules_stages_modules_2_modules_13_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_187 = (
            l_self_modules_stages_modules_2_modules_13_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_2_modules_13_modules_conv3_parameters_gain_ = None
        view_142 = mul_187.view(-1)
        mul_187 = None
        batch_norm_98 = torch.nn.functional.batch_norm(
            reshape_98,
            None,
            None,
            weight=view_142,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_98 = view_142 = None
        weight_98 = batch_norm_98.reshape_as(
            l_self_modules_stages_modules_2_modules_13_modules_conv3_parameters_weight_
        )
        batch_norm_98 = (
            l_self_modules_stages_modules_2_modules_13_modules_conv3_parameters_weight_
        ) = None
        out_158 = torch.conv2d(
            silu_94,
            weight_98,
            l_self_modules_stages_modules_2_modules_13_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_94 = (
            weight_98
        ) = (
            l_self_modules_stages_modules_2_modules_13_modules_conv3_parameters_bias_
        ) = None
        mean_22 = out_158.mean((2, 3))
        y_66 = mean_22.view(1, 1, -1)
        mean_22 = None
        y_67 = torch.conv1d(
            y_66,
            l_self_modules_stages_modules_2_modules_13_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_66 = l_self_modules_stages_modules_2_modules_13_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_22 = y_67.sigmoid()
        y_67 = None
        y_68 = sigmoid_22.view(1, -1, 1, 1)
        sigmoid_22 = None
        expand_as_22 = y_68.expand_as(out_158)
        y_68 = None
        mul_188 = out_158 * expand_as_22
        out_158 = expand_as_22 = None
        out_159 = 2.0 * mul_188
        mul_188 = None
        mul_190 = out_159 * 0.2
        out_159 = None
        out_160 = mul_190 + out_153
        mul_190 = out_153 = None
        silu_95 = torch.nn.functional.silu(out_160, inplace=False)
        out_161 = silu_95 * 0.8006407690254355
        silu_95 = None
        reshape_99 = l_self_modules_stages_modules_2_modules_14_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_192 = (
            l_self_modules_stages_modules_2_modules_14_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv1_parameters_gain_ = None
        view_145 = mul_192.view(-1)
        mul_192 = None
        batch_norm_99 = torch.nn.functional.batch_norm(
            reshape_99,
            None,
            None,
            weight=view_145,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_99 = view_145 = None
        weight_99 = batch_norm_99.reshape_as(
            l_self_modules_stages_modules_2_modules_14_modules_conv1_parameters_weight_
        )
        batch_norm_99 = (
            l_self_modules_stages_modules_2_modules_14_modules_conv1_parameters_weight_
        ) = None
        out_162 = torch.conv2d(
            out_161,
            weight_99,
            l_self_modules_stages_modules_2_modules_14_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_161 = (
            weight_99
        ) = (
            l_self_modules_stages_modules_2_modules_14_modules_conv1_parameters_bias_
        ) = None
        silu_96 = torch.nn.functional.silu(out_162, inplace=True)
        out_162 = None
        reshape_100 = l_self_modules_stages_modules_2_modules_14_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_193 = (
            l_self_modules_stages_modules_2_modules_14_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv2_parameters_gain_ = None
        view_146 = mul_193.view(-1)
        mul_193 = None
        batch_norm_100 = torch.nn.functional.batch_norm(
            reshape_100,
            None,
            None,
            weight=view_146,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_100 = view_146 = None
        weight_100 = batch_norm_100.reshape_as(
            l_self_modules_stages_modules_2_modules_14_modules_conv2_parameters_weight_
        )
        batch_norm_100 = (
            l_self_modules_stages_modules_2_modules_14_modules_conv2_parameters_weight_
        ) = None
        out_163 = torch.conv2d(
            silu_96,
            weight_100,
            l_self_modules_stages_modules_2_modules_14_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_96 = (
            weight_100
        ) = (
            l_self_modules_stages_modules_2_modules_14_modules_conv2_parameters_bias_
        ) = None
        silu_97 = torch.nn.functional.silu(out_163, inplace=True)
        out_163 = None
        reshape_101 = l_self_modules_stages_modules_2_modules_14_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_194 = (
            l_self_modules_stages_modules_2_modules_14_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv2b_parameters_gain_ = (
            None
        )
        view_147 = mul_194.view(-1)
        mul_194 = None
        batch_norm_101 = torch.nn.functional.batch_norm(
            reshape_101,
            None,
            None,
            weight=view_147,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_101 = view_147 = None
        weight_101 = batch_norm_101.reshape_as(
            l_self_modules_stages_modules_2_modules_14_modules_conv2b_parameters_weight_
        )
        batch_norm_101 = (
            l_self_modules_stages_modules_2_modules_14_modules_conv2b_parameters_weight_
        ) = None
        out_164 = torch.conv2d(
            silu_97,
            weight_101,
            l_self_modules_stages_modules_2_modules_14_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_97 = (
            weight_101
        ) = (
            l_self_modules_stages_modules_2_modules_14_modules_conv2b_parameters_bias_
        ) = None
        silu_98 = torch.nn.functional.silu(out_164, inplace=False)
        out_164 = None
        reshape_102 = l_self_modules_stages_modules_2_modules_14_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_195 = (
            l_self_modules_stages_modules_2_modules_14_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv3_parameters_gain_ = None
        view_148 = mul_195.view(-1)
        mul_195 = None
        batch_norm_102 = torch.nn.functional.batch_norm(
            reshape_102,
            None,
            None,
            weight=view_148,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_102 = view_148 = None
        weight_102 = batch_norm_102.reshape_as(
            l_self_modules_stages_modules_2_modules_14_modules_conv3_parameters_weight_
        )
        batch_norm_102 = (
            l_self_modules_stages_modules_2_modules_14_modules_conv3_parameters_weight_
        ) = None
        out_165 = torch.conv2d(
            silu_98,
            weight_102,
            l_self_modules_stages_modules_2_modules_14_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_98 = (
            weight_102
        ) = (
            l_self_modules_stages_modules_2_modules_14_modules_conv3_parameters_bias_
        ) = None
        mean_23 = out_165.mean((2, 3))
        y_69 = mean_23.view(1, 1, -1)
        mean_23 = None
        y_70 = torch.conv1d(
            y_69,
            l_self_modules_stages_modules_2_modules_14_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_69 = l_self_modules_stages_modules_2_modules_14_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_23 = y_70.sigmoid()
        y_70 = None
        y_71 = sigmoid_23.view(1, -1, 1, 1)
        sigmoid_23 = None
        expand_as_23 = y_71.expand_as(out_165)
        y_71 = None
        mul_196 = out_165 * expand_as_23
        out_165 = expand_as_23 = None
        out_166 = 2.0 * mul_196
        mul_196 = None
        mul_198 = out_166 * 0.2
        out_166 = None
        out_167 = mul_198 + out_160
        mul_198 = out_160 = None
        silu_99 = torch.nn.functional.silu(out_167, inplace=False)
        out_168 = silu_99 * 0.7905694150420947
        silu_99 = None
        reshape_103 = l_self_modules_stages_modules_2_modules_15_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_200 = (
            l_self_modules_stages_modules_2_modules_15_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_2_modules_15_modules_conv1_parameters_gain_ = None
        view_151 = mul_200.view(-1)
        mul_200 = None
        batch_norm_103 = torch.nn.functional.batch_norm(
            reshape_103,
            None,
            None,
            weight=view_151,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_103 = view_151 = None
        weight_103 = batch_norm_103.reshape_as(
            l_self_modules_stages_modules_2_modules_15_modules_conv1_parameters_weight_
        )
        batch_norm_103 = (
            l_self_modules_stages_modules_2_modules_15_modules_conv1_parameters_weight_
        ) = None
        out_169 = torch.conv2d(
            out_168,
            weight_103,
            l_self_modules_stages_modules_2_modules_15_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_168 = (
            weight_103
        ) = (
            l_self_modules_stages_modules_2_modules_15_modules_conv1_parameters_bias_
        ) = None
        silu_100 = torch.nn.functional.silu(out_169, inplace=True)
        out_169 = None
        reshape_104 = l_self_modules_stages_modules_2_modules_15_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_201 = (
            l_self_modules_stages_modules_2_modules_15_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_15_modules_conv2_parameters_gain_ = None
        view_152 = mul_201.view(-1)
        mul_201 = None
        batch_norm_104 = torch.nn.functional.batch_norm(
            reshape_104,
            None,
            None,
            weight=view_152,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_104 = view_152 = None
        weight_104 = batch_norm_104.reshape_as(
            l_self_modules_stages_modules_2_modules_15_modules_conv2_parameters_weight_
        )
        batch_norm_104 = (
            l_self_modules_stages_modules_2_modules_15_modules_conv2_parameters_weight_
        ) = None
        out_170 = torch.conv2d(
            silu_100,
            weight_104,
            l_self_modules_stages_modules_2_modules_15_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_100 = (
            weight_104
        ) = (
            l_self_modules_stages_modules_2_modules_15_modules_conv2_parameters_bias_
        ) = None
        silu_101 = torch.nn.functional.silu(out_170, inplace=True)
        out_170 = None
        reshape_105 = l_self_modules_stages_modules_2_modules_15_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_202 = (
            l_self_modules_stages_modules_2_modules_15_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_15_modules_conv2b_parameters_gain_ = (
            None
        )
        view_153 = mul_202.view(-1)
        mul_202 = None
        batch_norm_105 = torch.nn.functional.batch_norm(
            reshape_105,
            None,
            None,
            weight=view_153,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_105 = view_153 = None
        weight_105 = batch_norm_105.reshape_as(
            l_self_modules_stages_modules_2_modules_15_modules_conv2b_parameters_weight_
        )
        batch_norm_105 = (
            l_self_modules_stages_modules_2_modules_15_modules_conv2b_parameters_weight_
        ) = None
        out_171 = torch.conv2d(
            silu_101,
            weight_105,
            l_self_modules_stages_modules_2_modules_15_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_101 = (
            weight_105
        ) = (
            l_self_modules_stages_modules_2_modules_15_modules_conv2b_parameters_bias_
        ) = None
        silu_102 = torch.nn.functional.silu(out_171, inplace=False)
        out_171 = None
        reshape_106 = l_self_modules_stages_modules_2_modules_15_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_203 = (
            l_self_modules_stages_modules_2_modules_15_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_2_modules_15_modules_conv3_parameters_gain_ = None
        view_154 = mul_203.view(-1)
        mul_203 = None
        batch_norm_106 = torch.nn.functional.batch_norm(
            reshape_106,
            None,
            None,
            weight=view_154,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_106 = view_154 = None
        weight_106 = batch_norm_106.reshape_as(
            l_self_modules_stages_modules_2_modules_15_modules_conv3_parameters_weight_
        )
        batch_norm_106 = (
            l_self_modules_stages_modules_2_modules_15_modules_conv3_parameters_weight_
        ) = None
        out_172 = torch.conv2d(
            silu_102,
            weight_106,
            l_self_modules_stages_modules_2_modules_15_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_102 = (
            weight_106
        ) = (
            l_self_modules_stages_modules_2_modules_15_modules_conv3_parameters_bias_
        ) = None
        mean_24 = out_172.mean((2, 3))
        y_72 = mean_24.view(1, 1, -1)
        mean_24 = None
        y_73 = torch.conv1d(
            y_72,
            l_self_modules_stages_modules_2_modules_15_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_72 = l_self_modules_stages_modules_2_modules_15_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_24 = y_73.sigmoid()
        y_73 = None
        y_74 = sigmoid_24.view(1, -1, 1, 1)
        sigmoid_24 = None
        expand_as_24 = y_74.expand_as(out_172)
        y_74 = None
        mul_204 = out_172 * expand_as_24
        out_172 = expand_as_24 = None
        out_173 = 2.0 * mul_204
        mul_204 = None
        mul_206 = out_173 * 0.2
        out_173 = None
        out_174 = mul_206 + out_167
        mul_206 = out_167 = None
        silu_103 = torch.nn.functional.silu(out_174, inplace=False)
        out_175 = silu_103 * 0.7808688094430302
        silu_103 = None
        reshape_107 = l_self_modules_stages_modules_2_modules_16_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_208 = (
            l_self_modules_stages_modules_2_modules_16_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_2_modules_16_modules_conv1_parameters_gain_ = None
        view_157 = mul_208.view(-1)
        mul_208 = None
        batch_norm_107 = torch.nn.functional.batch_norm(
            reshape_107,
            None,
            None,
            weight=view_157,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_107 = view_157 = None
        weight_107 = batch_norm_107.reshape_as(
            l_self_modules_stages_modules_2_modules_16_modules_conv1_parameters_weight_
        )
        batch_norm_107 = (
            l_self_modules_stages_modules_2_modules_16_modules_conv1_parameters_weight_
        ) = None
        out_176 = torch.conv2d(
            out_175,
            weight_107,
            l_self_modules_stages_modules_2_modules_16_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_175 = (
            weight_107
        ) = (
            l_self_modules_stages_modules_2_modules_16_modules_conv1_parameters_bias_
        ) = None
        silu_104 = torch.nn.functional.silu(out_176, inplace=True)
        out_176 = None
        reshape_108 = l_self_modules_stages_modules_2_modules_16_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_209 = (
            l_self_modules_stages_modules_2_modules_16_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_16_modules_conv2_parameters_gain_ = None
        view_158 = mul_209.view(-1)
        mul_209 = None
        batch_norm_108 = torch.nn.functional.batch_norm(
            reshape_108,
            None,
            None,
            weight=view_158,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_108 = view_158 = None
        weight_108 = batch_norm_108.reshape_as(
            l_self_modules_stages_modules_2_modules_16_modules_conv2_parameters_weight_
        )
        batch_norm_108 = (
            l_self_modules_stages_modules_2_modules_16_modules_conv2_parameters_weight_
        ) = None
        out_177 = torch.conv2d(
            silu_104,
            weight_108,
            l_self_modules_stages_modules_2_modules_16_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_104 = (
            weight_108
        ) = (
            l_self_modules_stages_modules_2_modules_16_modules_conv2_parameters_bias_
        ) = None
        silu_105 = torch.nn.functional.silu(out_177, inplace=True)
        out_177 = None
        reshape_109 = l_self_modules_stages_modules_2_modules_16_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_210 = (
            l_self_modules_stages_modules_2_modules_16_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_16_modules_conv2b_parameters_gain_ = (
            None
        )
        view_159 = mul_210.view(-1)
        mul_210 = None
        batch_norm_109 = torch.nn.functional.batch_norm(
            reshape_109,
            None,
            None,
            weight=view_159,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_109 = view_159 = None
        weight_109 = batch_norm_109.reshape_as(
            l_self_modules_stages_modules_2_modules_16_modules_conv2b_parameters_weight_
        )
        batch_norm_109 = (
            l_self_modules_stages_modules_2_modules_16_modules_conv2b_parameters_weight_
        ) = None
        out_178 = torch.conv2d(
            silu_105,
            weight_109,
            l_self_modules_stages_modules_2_modules_16_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_105 = (
            weight_109
        ) = (
            l_self_modules_stages_modules_2_modules_16_modules_conv2b_parameters_bias_
        ) = None
        silu_106 = torch.nn.functional.silu(out_178, inplace=False)
        out_178 = None
        reshape_110 = l_self_modules_stages_modules_2_modules_16_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_211 = (
            l_self_modules_stages_modules_2_modules_16_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_2_modules_16_modules_conv3_parameters_gain_ = None
        view_160 = mul_211.view(-1)
        mul_211 = None
        batch_norm_110 = torch.nn.functional.batch_norm(
            reshape_110,
            None,
            None,
            weight=view_160,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_110 = view_160 = None
        weight_110 = batch_norm_110.reshape_as(
            l_self_modules_stages_modules_2_modules_16_modules_conv3_parameters_weight_
        )
        batch_norm_110 = (
            l_self_modules_stages_modules_2_modules_16_modules_conv3_parameters_weight_
        ) = None
        out_179 = torch.conv2d(
            silu_106,
            weight_110,
            l_self_modules_stages_modules_2_modules_16_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_106 = (
            weight_110
        ) = (
            l_self_modules_stages_modules_2_modules_16_modules_conv3_parameters_bias_
        ) = None
        mean_25 = out_179.mean((2, 3))
        y_75 = mean_25.view(1, 1, -1)
        mean_25 = None
        y_76 = torch.conv1d(
            y_75,
            l_self_modules_stages_modules_2_modules_16_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_75 = l_self_modules_stages_modules_2_modules_16_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_25 = y_76.sigmoid()
        y_76 = None
        y_77 = sigmoid_25.view(1, -1, 1, 1)
        sigmoid_25 = None
        expand_as_25 = y_77.expand_as(out_179)
        y_77 = None
        mul_212 = out_179 * expand_as_25
        out_179 = expand_as_25 = None
        out_180 = 2.0 * mul_212
        mul_212 = None
        mul_214 = out_180 * 0.2
        out_180 = None
        out_181 = mul_214 + out_174
        mul_214 = out_174 = None
        silu_107 = torch.nn.functional.silu(out_181, inplace=False)
        out_182 = silu_107 * 0.7715167498104594
        silu_107 = None
        reshape_111 = l_self_modules_stages_modules_2_modules_17_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_216 = (
            l_self_modules_stages_modules_2_modules_17_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_2_modules_17_modules_conv1_parameters_gain_ = None
        view_163 = mul_216.view(-1)
        mul_216 = None
        batch_norm_111 = torch.nn.functional.batch_norm(
            reshape_111,
            None,
            None,
            weight=view_163,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_111 = view_163 = None
        weight_111 = batch_norm_111.reshape_as(
            l_self_modules_stages_modules_2_modules_17_modules_conv1_parameters_weight_
        )
        batch_norm_111 = (
            l_self_modules_stages_modules_2_modules_17_modules_conv1_parameters_weight_
        ) = None
        out_183 = torch.conv2d(
            out_182,
            weight_111,
            l_self_modules_stages_modules_2_modules_17_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_182 = (
            weight_111
        ) = (
            l_self_modules_stages_modules_2_modules_17_modules_conv1_parameters_bias_
        ) = None
        silu_108 = torch.nn.functional.silu(out_183, inplace=True)
        out_183 = None
        reshape_112 = l_self_modules_stages_modules_2_modules_17_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_217 = (
            l_self_modules_stages_modules_2_modules_17_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_17_modules_conv2_parameters_gain_ = None
        view_164 = mul_217.view(-1)
        mul_217 = None
        batch_norm_112 = torch.nn.functional.batch_norm(
            reshape_112,
            None,
            None,
            weight=view_164,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_112 = view_164 = None
        weight_112 = batch_norm_112.reshape_as(
            l_self_modules_stages_modules_2_modules_17_modules_conv2_parameters_weight_
        )
        batch_norm_112 = (
            l_self_modules_stages_modules_2_modules_17_modules_conv2_parameters_weight_
        ) = None
        out_184 = torch.conv2d(
            silu_108,
            weight_112,
            l_self_modules_stages_modules_2_modules_17_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_108 = (
            weight_112
        ) = (
            l_self_modules_stages_modules_2_modules_17_modules_conv2_parameters_bias_
        ) = None
        silu_109 = torch.nn.functional.silu(out_184, inplace=True)
        out_184 = None
        reshape_113 = l_self_modules_stages_modules_2_modules_17_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_218 = (
            l_self_modules_stages_modules_2_modules_17_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_2_modules_17_modules_conv2b_parameters_gain_ = (
            None
        )
        view_165 = mul_218.view(-1)
        mul_218 = None
        batch_norm_113 = torch.nn.functional.batch_norm(
            reshape_113,
            None,
            None,
            weight=view_165,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_113 = view_165 = None
        weight_113 = batch_norm_113.reshape_as(
            l_self_modules_stages_modules_2_modules_17_modules_conv2b_parameters_weight_
        )
        batch_norm_113 = (
            l_self_modules_stages_modules_2_modules_17_modules_conv2b_parameters_weight_
        ) = None
        out_185 = torch.conv2d(
            silu_109,
            weight_113,
            l_self_modules_stages_modules_2_modules_17_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_109 = (
            weight_113
        ) = (
            l_self_modules_stages_modules_2_modules_17_modules_conv2b_parameters_bias_
        ) = None
        silu_110 = torch.nn.functional.silu(out_185, inplace=False)
        out_185 = None
        reshape_114 = l_self_modules_stages_modules_2_modules_17_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_219 = (
            l_self_modules_stages_modules_2_modules_17_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_2_modules_17_modules_conv3_parameters_gain_ = None
        view_166 = mul_219.view(-1)
        mul_219 = None
        batch_norm_114 = torch.nn.functional.batch_norm(
            reshape_114,
            None,
            None,
            weight=view_166,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_114 = view_166 = None
        weight_114 = batch_norm_114.reshape_as(
            l_self_modules_stages_modules_2_modules_17_modules_conv3_parameters_weight_
        )
        batch_norm_114 = (
            l_self_modules_stages_modules_2_modules_17_modules_conv3_parameters_weight_
        ) = None
        out_186 = torch.conv2d(
            silu_110,
            weight_114,
            l_self_modules_stages_modules_2_modules_17_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_110 = (
            weight_114
        ) = (
            l_self_modules_stages_modules_2_modules_17_modules_conv3_parameters_bias_
        ) = None
        mean_26 = out_186.mean((2, 3))
        y_78 = mean_26.view(1, 1, -1)
        mean_26 = None
        y_79 = torch.conv1d(
            y_78,
            l_self_modules_stages_modules_2_modules_17_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_78 = l_self_modules_stages_modules_2_modules_17_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_26 = y_79.sigmoid()
        y_79 = None
        y_80 = sigmoid_26.view(1, -1, 1, 1)
        sigmoid_26 = None
        expand_as_26 = y_80.expand_as(out_186)
        y_80 = None
        mul_220 = out_186 * expand_as_26
        out_186 = expand_as_26 = None
        out_187 = 2.0 * mul_220
        mul_220 = None
        mul_222 = out_187 * 0.2
        out_187 = None
        out_188 = mul_222 + out_181
        mul_222 = out_181 = None
        silu_111 = torch.nn.functional.silu(out_188, inplace=False)
        out_188 = None
        out_189 = silu_111 * 0.7624928516630232
        silu_111 = None
        avg_pool2d_2 = torch._C._nn.avg_pool2d(out_189, 2, 2, 0, True, False, None)
        reshape_115 = l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_224 = (
            l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_gain_ = (
            None
        )
        view_169 = mul_224.view(-1)
        mul_224 = None
        batch_norm_115 = torch.nn.functional.batch_norm(
            reshape_115,
            None,
            None,
            weight=view_169,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_115 = view_169 = None
        weight_115 = batch_norm_115.reshape_as(
            l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_weight_
        )
        batch_norm_115 = l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        shortcut_3 = torch.conv2d(
            avg_pool2d_2,
            weight_115,
            l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        avg_pool2d_2 = (
            weight_115
        ) = l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_bias_ = (None)
        reshape_116 = l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_225 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_gain_ = None
        view_170 = mul_225.view(-1)
        mul_225 = None
        batch_norm_116 = torch.nn.functional.batch_norm(
            reshape_116,
            None,
            None,
            weight=view_170,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_116 = view_170 = None
        weight_116 = batch_norm_116.reshape_as(
            l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_weight_
        )
        batch_norm_116 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_weight_
        ) = None
        out_190 = torch.conv2d(
            out_189,
            weight_116,
            l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_189 = (
            weight_116
        ) = (
            l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_bias_
        ) = None
        silu_112 = torch.nn.functional.silu(out_190, inplace=True)
        out_190 = None
        reshape_117 = l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_226 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_gain_ = None
        view_171 = mul_226.view(-1)
        mul_226 = None
        batch_norm_117 = torch.nn.functional.batch_norm(
            reshape_117,
            None,
            None,
            weight=view_171,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_117 = view_171 = None
        weight_117 = batch_norm_117.reshape_as(
            l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_weight_
        )
        batch_norm_117 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_weight_
        ) = None
        out_191 = torch.conv2d(
            silu_112,
            weight_117,
            l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            6,
        )
        silu_112 = (
            weight_117
        ) = (
            l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_bias_
        ) = None
        silu_113 = torch.nn.functional.silu(out_191, inplace=True)
        out_191 = None
        reshape_118 = l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_227 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_gain_ = None
        view_172 = mul_227.view(-1)
        mul_227 = None
        batch_norm_118 = torch.nn.functional.batch_norm(
            reshape_118,
            None,
            None,
            weight=view_172,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_118 = view_172 = None
        weight_118 = batch_norm_118.reshape_as(
            l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_weight_
        )
        batch_norm_118 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_weight_
        ) = None
        out_192 = torch.conv2d(
            silu_113,
            weight_118,
            l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_113 = (
            weight_118
        ) = (
            l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_bias_
        ) = None
        silu_114 = torch.nn.functional.silu(out_192, inplace=False)
        out_192 = None
        reshape_119 = l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_228 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_gain_ = None
        view_173 = mul_228.view(-1)
        mul_228 = None
        batch_norm_119 = torch.nn.functional.batch_norm(
            reshape_119,
            None,
            None,
            weight=view_173,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_119 = view_173 = None
        weight_119 = batch_norm_119.reshape_as(
            l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_weight_
        )
        batch_norm_119 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_weight_
        ) = None
        out_193 = torch.conv2d(
            silu_114,
            weight_119,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_114 = (
            weight_119
        ) = (
            l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_bias_
        ) = None
        mean_27 = out_193.mean((2, 3))
        y_81 = mean_27.view(1, 1, -1)
        mean_27 = None
        y_82 = torch.conv1d(
            y_81,
            l_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_81 = l_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_27 = y_82.sigmoid()
        y_82 = None
        y_83 = sigmoid_27.view(1, -1, 1, 1)
        sigmoid_27 = None
        expand_as_27 = y_83.expand_as(out_193)
        y_83 = None
        mul_229 = out_193 * expand_as_27
        out_193 = expand_as_27 = None
        out_194 = 2.0 * mul_229
        mul_229 = None
        mul_231 = out_194 * 0.2
        out_194 = None
        out_195 = mul_231 + shortcut_3
        mul_231 = shortcut_3 = None
        silu_115 = torch.nn.functional.silu(out_195, inplace=False)
        out_196 = silu_115 * 0.9805806756909201
        silu_115 = None
        reshape_120 = l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_233 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_gain_ = None
        view_176 = mul_233.view(-1)
        mul_233 = None
        batch_norm_120 = torch.nn.functional.batch_norm(
            reshape_120,
            None,
            None,
            weight=view_176,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_120 = view_176 = None
        weight_120 = batch_norm_120.reshape_as(
            l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_weight_
        )
        batch_norm_120 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_weight_
        ) = None
        out_197 = torch.conv2d(
            out_196,
            weight_120,
            l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_196 = (
            weight_120
        ) = (
            l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_bias_
        ) = None
        silu_116 = torch.nn.functional.silu(out_197, inplace=True)
        out_197 = None
        reshape_121 = l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_234 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_gain_ = None
        view_177 = mul_234.view(-1)
        mul_234 = None
        batch_norm_121 = torch.nn.functional.batch_norm(
            reshape_121,
            None,
            None,
            weight=view_177,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_121 = view_177 = None
        weight_121 = batch_norm_121.reshape_as(
            l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_weight_
        )
        batch_norm_121 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_weight_
        ) = None
        out_198 = torch.conv2d(
            silu_116,
            weight_121,
            l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_116 = (
            weight_121
        ) = (
            l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_bias_
        ) = None
        silu_117 = torch.nn.functional.silu(out_198, inplace=True)
        out_198 = None
        reshape_122 = l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_235 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_gain_ = None
        view_178 = mul_235.view(-1)
        mul_235 = None
        batch_norm_122 = torch.nn.functional.batch_norm(
            reshape_122,
            None,
            None,
            weight=view_178,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_122 = view_178 = None
        weight_122 = batch_norm_122.reshape_as(
            l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_weight_
        )
        batch_norm_122 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_weight_
        ) = None
        out_199 = torch.conv2d(
            silu_117,
            weight_122,
            l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_117 = (
            weight_122
        ) = (
            l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_bias_
        ) = None
        silu_118 = torch.nn.functional.silu(out_199, inplace=False)
        out_199 = None
        reshape_123 = l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_236 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_gain_ = None
        view_179 = mul_236.view(-1)
        mul_236 = None
        batch_norm_123 = torch.nn.functional.batch_norm(
            reshape_123,
            None,
            None,
            weight=view_179,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_123 = view_179 = None
        weight_123 = batch_norm_123.reshape_as(
            l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_weight_
        )
        batch_norm_123 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_weight_
        ) = None
        out_200 = torch.conv2d(
            silu_118,
            weight_123,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_118 = (
            weight_123
        ) = (
            l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_bias_
        ) = None
        mean_28 = out_200.mean((2, 3))
        y_84 = mean_28.view(1, 1, -1)
        mean_28 = None
        y_85 = torch.conv1d(
            y_84,
            l_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_84 = l_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_28 = y_85.sigmoid()
        y_85 = None
        y_86 = sigmoid_28.view(1, -1, 1, 1)
        sigmoid_28 = None
        expand_as_28 = y_86.expand_as(out_200)
        y_86 = None
        mul_237 = out_200 * expand_as_28
        out_200 = expand_as_28 = None
        out_201 = 2.0 * mul_237
        mul_237 = None
        mul_239 = out_201 * 0.2
        out_201 = None
        out_202 = mul_239 + out_195
        mul_239 = out_195 = None
        silu_119 = torch.nn.functional.silu(out_202, inplace=False)
        out_203 = silu_119 * 0.9622504486493761
        silu_119 = None
        reshape_124 = l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_241 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_gain_ = None
        view_182 = mul_241.view(-1)
        mul_241 = None
        batch_norm_124 = torch.nn.functional.batch_norm(
            reshape_124,
            None,
            None,
            weight=view_182,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_124 = view_182 = None
        weight_124 = batch_norm_124.reshape_as(
            l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_weight_
        )
        batch_norm_124 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_weight_
        ) = None
        out_204 = torch.conv2d(
            out_203,
            weight_124,
            l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_203 = (
            weight_124
        ) = (
            l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_bias_
        ) = None
        silu_120 = torch.nn.functional.silu(out_204, inplace=True)
        out_204 = None
        reshape_125 = l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_242 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_gain_ = None
        view_183 = mul_242.view(-1)
        mul_242 = None
        batch_norm_125 = torch.nn.functional.batch_norm(
            reshape_125,
            None,
            None,
            weight=view_183,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_125 = view_183 = None
        weight_125 = batch_norm_125.reshape_as(
            l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_weight_
        )
        batch_norm_125 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_weight_
        ) = None
        out_205 = torch.conv2d(
            silu_120,
            weight_125,
            l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_120 = (
            weight_125
        ) = (
            l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_bias_
        ) = None
        silu_121 = torch.nn.functional.silu(out_205, inplace=True)
        out_205 = None
        reshape_126 = l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_243 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_gain_ = None
        view_184 = mul_243.view(-1)
        mul_243 = None
        batch_norm_126 = torch.nn.functional.batch_norm(
            reshape_126,
            None,
            None,
            weight=view_184,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_126 = view_184 = None
        weight_126 = batch_norm_126.reshape_as(
            l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_weight_
        )
        batch_norm_126 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_weight_
        ) = None
        out_206 = torch.conv2d(
            silu_121,
            weight_126,
            l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_121 = (
            weight_126
        ) = (
            l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_bias_
        ) = None
        silu_122 = torch.nn.functional.silu(out_206, inplace=False)
        out_206 = None
        reshape_127 = l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_244 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_gain_ = None
        view_185 = mul_244.view(-1)
        mul_244 = None
        batch_norm_127 = torch.nn.functional.batch_norm(
            reshape_127,
            None,
            None,
            weight=view_185,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_127 = view_185 = None
        weight_127 = batch_norm_127.reshape_as(
            l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_weight_
        )
        batch_norm_127 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_weight_
        ) = None
        out_207 = torch.conv2d(
            silu_122,
            weight_127,
            l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_122 = (
            weight_127
        ) = (
            l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_bias_
        ) = None
        mean_29 = out_207.mean((2, 3))
        y_87 = mean_29.view(1, 1, -1)
        mean_29 = None
        y_88 = torch.conv1d(
            y_87,
            l_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_87 = l_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_29 = y_88.sigmoid()
        y_88 = None
        y_89 = sigmoid_29.view(1, -1, 1, 1)
        sigmoid_29 = None
        expand_as_29 = y_89.expand_as(out_207)
        y_89 = None
        mul_245 = out_207 * expand_as_29
        out_207 = expand_as_29 = None
        out_208 = 2.0 * mul_245
        mul_245 = None
        mul_247 = out_208 * 0.2
        out_208 = None
        out_209 = mul_247 + out_202
        mul_247 = out_202 = None
        silu_123 = torch.nn.functional.silu(out_209, inplace=False)
        out_210 = silu_123 * 0.9449111825230679
        silu_123 = None
        reshape_128 = l_self_modules_stages_modules_3_modules_3_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_249 = (
            l_self_modules_stages_modules_3_modules_3_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_3_modules_3_modules_conv1_parameters_gain_ = None
        view_188 = mul_249.view(-1)
        mul_249 = None
        batch_norm_128 = torch.nn.functional.batch_norm(
            reshape_128,
            None,
            None,
            weight=view_188,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_128 = view_188 = None
        weight_128 = batch_norm_128.reshape_as(
            l_self_modules_stages_modules_3_modules_3_modules_conv1_parameters_weight_
        )
        batch_norm_128 = (
            l_self_modules_stages_modules_3_modules_3_modules_conv1_parameters_weight_
        ) = None
        out_211 = torch.conv2d(
            out_210,
            weight_128,
            l_self_modules_stages_modules_3_modules_3_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_210 = (
            weight_128
        ) = (
            l_self_modules_stages_modules_3_modules_3_modules_conv1_parameters_bias_
        ) = None
        silu_124 = torch.nn.functional.silu(out_211, inplace=True)
        out_211 = None
        reshape_129 = l_self_modules_stages_modules_3_modules_3_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_250 = (
            l_self_modules_stages_modules_3_modules_3_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_3_modules_3_modules_conv2_parameters_gain_ = None
        view_189 = mul_250.view(-1)
        mul_250 = None
        batch_norm_129 = torch.nn.functional.batch_norm(
            reshape_129,
            None,
            None,
            weight=view_189,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_129 = view_189 = None
        weight_129 = batch_norm_129.reshape_as(
            l_self_modules_stages_modules_3_modules_3_modules_conv2_parameters_weight_
        )
        batch_norm_129 = (
            l_self_modules_stages_modules_3_modules_3_modules_conv2_parameters_weight_
        ) = None
        out_212 = torch.conv2d(
            silu_124,
            weight_129,
            l_self_modules_stages_modules_3_modules_3_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_124 = (
            weight_129
        ) = (
            l_self_modules_stages_modules_3_modules_3_modules_conv2_parameters_bias_
        ) = None
        silu_125 = torch.nn.functional.silu(out_212, inplace=True)
        out_212 = None
        reshape_130 = l_self_modules_stages_modules_3_modules_3_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_251 = (
            l_self_modules_stages_modules_3_modules_3_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_3_modules_3_modules_conv2b_parameters_gain_ = None
        view_190 = mul_251.view(-1)
        mul_251 = None
        batch_norm_130 = torch.nn.functional.batch_norm(
            reshape_130,
            None,
            None,
            weight=view_190,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_130 = view_190 = None
        weight_130 = batch_norm_130.reshape_as(
            l_self_modules_stages_modules_3_modules_3_modules_conv2b_parameters_weight_
        )
        batch_norm_130 = (
            l_self_modules_stages_modules_3_modules_3_modules_conv2b_parameters_weight_
        ) = None
        out_213 = torch.conv2d(
            silu_125,
            weight_130,
            l_self_modules_stages_modules_3_modules_3_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_125 = (
            weight_130
        ) = (
            l_self_modules_stages_modules_3_modules_3_modules_conv2b_parameters_bias_
        ) = None
        silu_126 = torch.nn.functional.silu(out_213, inplace=False)
        out_213 = None
        reshape_131 = l_self_modules_stages_modules_3_modules_3_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_252 = (
            l_self_modules_stages_modules_3_modules_3_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_3_modules_3_modules_conv3_parameters_gain_ = None
        view_191 = mul_252.view(-1)
        mul_252 = None
        batch_norm_131 = torch.nn.functional.batch_norm(
            reshape_131,
            None,
            None,
            weight=view_191,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_131 = view_191 = None
        weight_131 = batch_norm_131.reshape_as(
            l_self_modules_stages_modules_3_modules_3_modules_conv3_parameters_weight_
        )
        batch_norm_131 = (
            l_self_modules_stages_modules_3_modules_3_modules_conv3_parameters_weight_
        ) = None
        out_214 = torch.conv2d(
            silu_126,
            weight_131,
            l_self_modules_stages_modules_3_modules_3_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_126 = (
            weight_131
        ) = (
            l_self_modules_stages_modules_3_modules_3_modules_conv3_parameters_bias_
        ) = None
        mean_30 = out_214.mean((2, 3))
        y_90 = mean_30.view(1, 1, -1)
        mean_30 = None
        y_91 = torch.conv1d(
            y_90,
            l_self_modules_stages_modules_3_modules_3_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_90 = l_self_modules_stages_modules_3_modules_3_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_30 = y_91.sigmoid()
        y_91 = None
        y_92 = sigmoid_30.view(1, -1, 1, 1)
        sigmoid_30 = None
        expand_as_30 = y_92.expand_as(out_214)
        y_92 = None
        mul_253 = out_214 * expand_as_30
        out_214 = expand_as_30 = None
        out_215 = 2.0 * mul_253
        mul_253 = None
        mul_255 = out_215 * 0.2
        out_215 = None
        out_216 = mul_255 + out_209
        mul_255 = out_209 = None
        silu_127 = torch.nn.functional.silu(out_216, inplace=False)
        out_217 = silu_127 * 0.9284766908852592
        silu_127 = None
        reshape_132 = l_self_modules_stages_modules_3_modules_4_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_257 = (
            l_self_modules_stages_modules_3_modules_4_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_3_modules_4_modules_conv1_parameters_gain_ = None
        view_194 = mul_257.view(-1)
        mul_257 = None
        batch_norm_132 = torch.nn.functional.batch_norm(
            reshape_132,
            None,
            None,
            weight=view_194,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_132 = view_194 = None
        weight_132 = batch_norm_132.reshape_as(
            l_self_modules_stages_modules_3_modules_4_modules_conv1_parameters_weight_
        )
        batch_norm_132 = (
            l_self_modules_stages_modules_3_modules_4_modules_conv1_parameters_weight_
        ) = None
        out_218 = torch.conv2d(
            out_217,
            weight_132,
            l_self_modules_stages_modules_3_modules_4_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_217 = (
            weight_132
        ) = (
            l_self_modules_stages_modules_3_modules_4_modules_conv1_parameters_bias_
        ) = None
        silu_128 = torch.nn.functional.silu(out_218, inplace=True)
        out_218 = None
        reshape_133 = l_self_modules_stages_modules_3_modules_4_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_258 = (
            l_self_modules_stages_modules_3_modules_4_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_3_modules_4_modules_conv2_parameters_gain_ = None
        view_195 = mul_258.view(-1)
        mul_258 = None
        batch_norm_133 = torch.nn.functional.batch_norm(
            reshape_133,
            None,
            None,
            weight=view_195,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_133 = view_195 = None
        weight_133 = batch_norm_133.reshape_as(
            l_self_modules_stages_modules_3_modules_4_modules_conv2_parameters_weight_
        )
        batch_norm_133 = (
            l_self_modules_stages_modules_3_modules_4_modules_conv2_parameters_weight_
        ) = None
        out_219 = torch.conv2d(
            silu_128,
            weight_133,
            l_self_modules_stages_modules_3_modules_4_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_128 = (
            weight_133
        ) = (
            l_self_modules_stages_modules_3_modules_4_modules_conv2_parameters_bias_
        ) = None
        silu_129 = torch.nn.functional.silu(out_219, inplace=True)
        out_219 = None
        reshape_134 = l_self_modules_stages_modules_3_modules_4_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_259 = (
            l_self_modules_stages_modules_3_modules_4_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_3_modules_4_modules_conv2b_parameters_gain_ = None
        view_196 = mul_259.view(-1)
        mul_259 = None
        batch_norm_134 = torch.nn.functional.batch_norm(
            reshape_134,
            None,
            None,
            weight=view_196,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_134 = view_196 = None
        weight_134 = batch_norm_134.reshape_as(
            l_self_modules_stages_modules_3_modules_4_modules_conv2b_parameters_weight_
        )
        batch_norm_134 = (
            l_self_modules_stages_modules_3_modules_4_modules_conv2b_parameters_weight_
        ) = None
        out_220 = torch.conv2d(
            silu_129,
            weight_134,
            l_self_modules_stages_modules_3_modules_4_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_129 = (
            weight_134
        ) = (
            l_self_modules_stages_modules_3_modules_4_modules_conv2b_parameters_bias_
        ) = None
        silu_130 = torch.nn.functional.silu(out_220, inplace=False)
        out_220 = None
        reshape_135 = l_self_modules_stages_modules_3_modules_4_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_260 = (
            l_self_modules_stages_modules_3_modules_4_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_3_modules_4_modules_conv3_parameters_gain_ = None
        view_197 = mul_260.view(-1)
        mul_260 = None
        batch_norm_135 = torch.nn.functional.batch_norm(
            reshape_135,
            None,
            None,
            weight=view_197,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_135 = view_197 = None
        weight_135 = batch_norm_135.reshape_as(
            l_self_modules_stages_modules_3_modules_4_modules_conv3_parameters_weight_
        )
        batch_norm_135 = (
            l_self_modules_stages_modules_3_modules_4_modules_conv3_parameters_weight_
        ) = None
        out_221 = torch.conv2d(
            silu_130,
            weight_135,
            l_self_modules_stages_modules_3_modules_4_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_130 = (
            weight_135
        ) = (
            l_self_modules_stages_modules_3_modules_4_modules_conv3_parameters_bias_
        ) = None
        mean_31 = out_221.mean((2, 3))
        y_93 = mean_31.view(1, 1, -1)
        mean_31 = None
        y_94 = torch.conv1d(
            y_93,
            l_self_modules_stages_modules_3_modules_4_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_93 = l_self_modules_stages_modules_3_modules_4_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_31 = y_94.sigmoid()
        y_94 = None
        y_95 = sigmoid_31.view(1, -1, 1, 1)
        sigmoid_31 = None
        expand_as_31 = y_95.expand_as(out_221)
        y_95 = None
        mul_261 = out_221 * expand_as_31
        out_221 = expand_as_31 = None
        out_222 = 2.0 * mul_261
        mul_261 = None
        mul_263 = out_222 * 0.2
        out_222 = None
        out_223 = mul_263 + out_216
        mul_263 = out_216 = None
        silu_131 = torch.nn.functional.silu(out_223, inplace=False)
        out_224 = silu_131 * 0.9128709291752768
        silu_131 = None
        reshape_136 = l_self_modules_stages_modules_3_modules_5_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_265 = (
            l_self_modules_stages_modules_3_modules_5_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_3_modules_5_modules_conv1_parameters_gain_ = None
        view_200 = mul_265.view(-1)
        mul_265 = None
        batch_norm_136 = torch.nn.functional.batch_norm(
            reshape_136,
            None,
            None,
            weight=view_200,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_136 = view_200 = None
        weight_136 = batch_norm_136.reshape_as(
            l_self_modules_stages_modules_3_modules_5_modules_conv1_parameters_weight_
        )
        batch_norm_136 = (
            l_self_modules_stages_modules_3_modules_5_modules_conv1_parameters_weight_
        ) = None
        out_225 = torch.conv2d(
            out_224,
            weight_136,
            l_self_modules_stages_modules_3_modules_5_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_224 = (
            weight_136
        ) = (
            l_self_modules_stages_modules_3_modules_5_modules_conv1_parameters_bias_
        ) = None
        silu_132 = torch.nn.functional.silu(out_225, inplace=True)
        out_225 = None
        reshape_137 = l_self_modules_stages_modules_3_modules_5_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_266 = (
            l_self_modules_stages_modules_3_modules_5_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_3_modules_5_modules_conv2_parameters_gain_ = None
        view_201 = mul_266.view(-1)
        mul_266 = None
        batch_norm_137 = torch.nn.functional.batch_norm(
            reshape_137,
            None,
            None,
            weight=view_201,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_137 = view_201 = None
        weight_137 = batch_norm_137.reshape_as(
            l_self_modules_stages_modules_3_modules_5_modules_conv2_parameters_weight_
        )
        batch_norm_137 = (
            l_self_modules_stages_modules_3_modules_5_modules_conv2_parameters_weight_
        ) = None
        out_226 = torch.conv2d(
            silu_132,
            weight_137,
            l_self_modules_stages_modules_3_modules_5_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_132 = (
            weight_137
        ) = (
            l_self_modules_stages_modules_3_modules_5_modules_conv2_parameters_bias_
        ) = None
        silu_133 = torch.nn.functional.silu(out_226, inplace=True)
        out_226 = None
        reshape_138 = l_self_modules_stages_modules_3_modules_5_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_267 = (
            l_self_modules_stages_modules_3_modules_5_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_3_modules_5_modules_conv2b_parameters_gain_ = None
        view_202 = mul_267.view(-1)
        mul_267 = None
        batch_norm_138 = torch.nn.functional.batch_norm(
            reshape_138,
            None,
            None,
            weight=view_202,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_138 = view_202 = None
        weight_138 = batch_norm_138.reshape_as(
            l_self_modules_stages_modules_3_modules_5_modules_conv2b_parameters_weight_
        )
        batch_norm_138 = (
            l_self_modules_stages_modules_3_modules_5_modules_conv2b_parameters_weight_
        ) = None
        out_227 = torch.conv2d(
            silu_133,
            weight_138,
            l_self_modules_stages_modules_3_modules_5_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_133 = (
            weight_138
        ) = (
            l_self_modules_stages_modules_3_modules_5_modules_conv2b_parameters_bias_
        ) = None
        silu_134 = torch.nn.functional.silu(out_227, inplace=False)
        out_227 = None
        reshape_139 = l_self_modules_stages_modules_3_modules_5_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_268 = (
            l_self_modules_stages_modules_3_modules_5_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_3_modules_5_modules_conv3_parameters_gain_ = None
        view_203 = mul_268.view(-1)
        mul_268 = None
        batch_norm_139 = torch.nn.functional.batch_norm(
            reshape_139,
            None,
            None,
            weight=view_203,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_139 = view_203 = None
        weight_139 = batch_norm_139.reshape_as(
            l_self_modules_stages_modules_3_modules_5_modules_conv3_parameters_weight_
        )
        batch_norm_139 = (
            l_self_modules_stages_modules_3_modules_5_modules_conv3_parameters_weight_
        ) = None
        out_228 = torch.conv2d(
            silu_134,
            weight_139,
            l_self_modules_stages_modules_3_modules_5_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_134 = (
            weight_139
        ) = (
            l_self_modules_stages_modules_3_modules_5_modules_conv3_parameters_bias_
        ) = None
        mean_32 = out_228.mean((2, 3))
        y_96 = mean_32.view(1, 1, -1)
        mean_32 = None
        y_97 = torch.conv1d(
            y_96,
            l_self_modules_stages_modules_3_modules_5_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_96 = l_self_modules_stages_modules_3_modules_5_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_32 = y_97.sigmoid()
        y_97 = None
        y_98 = sigmoid_32.view(1, -1, 1, 1)
        sigmoid_32 = None
        expand_as_32 = y_98.expand_as(out_228)
        y_98 = None
        mul_269 = out_228 * expand_as_32
        out_228 = expand_as_32 = None
        out_229 = 2.0 * mul_269
        mul_269 = None
        mul_271 = out_229 * 0.2
        out_229 = None
        out_230 = mul_271 + out_223
        mul_271 = out_223 = None
        silu_135 = torch.nn.functional.silu(out_230, inplace=False)
        out_231 = silu_135 * 0.8980265101338745
        silu_135 = None
        reshape_140 = l_self_modules_stages_modules_3_modules_6_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_273 = (
            l_self_modules_stages_modules_3_modules_6_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_3_modules_6_modules_conv1_parameters_gain_ = None
        view_206 = mul_273.view(-1)
        mul_273 = None
        batch_norm_140 = torch.nn.functional.batch_norm(
            reshape_140,
            None,
            None,
            weight=view_206,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_140 = view_206 = None
        weight_140 = batch_norm_140.reshape_as(
            l_self_modules_stages_modules_3_modules_6_modules_conv1_parameters_weight_
        )
        batch_norm_140 = (
            l_self_modules_stages_modules_3_modules_6_modules_conv1_parameters_weight_
        ) = None
        out_232 = torch.conv2d(
            out_231,
            weight_140,
            l_self_modules_stages_modules_3_modules_6_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_231 = (
            weight_140
        ) = (
            l_self_modules_stages_modules_3_modules_6_modules_conv1_parameters_bias_
        ) = None
        silu_136 = torch.nn.functional.silu(out_232, inplace=True)
        out_232 = None
        reshape_141 = l_self_modules_stages_modules_3_modules_6_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_274 = (
            l_self_modules_stages_modules_3_modules_6_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_3_modules_6_modules_conv2_parameters_gain_ = None
        view_207 = mul_274.view(-1)
        mul_274 = None
        batch_norm_141 = torch.nn.functional.batch_norm(
            reshape_141,
            None,
            None,
            weight=view_207,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_141 = view_207 = None
        weight_141 = batch_norm_141.reshape_as(
            l_self_modules_stages_modules_3_modules_6_modules_conv2_parameters_weight_
        )
        batch_norm_141 = (
            l_self_modules_stages_modules_3_modules_6_modules_conv2_parameters_weight_
        ) = None
        out_233 = torch.conv2d(
            silu_136,
            weight_141,
            l_self_modules_stages_modules_3_modules_6_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_136 = (
            weight_141
        ) = (
            l_self_modules_stages_modules_3_modules_6_modules_conv2_parameters_bias_
        ) = None
        silu_137 = torch.nn.functional.silu(out_233, inplace=True)
        out_233 = None
        reshape_142 = l_self_modules_stages_modules_3_modules_6_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_275 = (
            l_self_modules_stages_modules_3_modules_6_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_3_modules_6_modules_conv2b_parameters_gain_ = None
        view_208 = mul_275.view(-1)
        mul_275 = None
        batch_norm_142 = torch.nn.functional.batch_norm(
            reshape_142,
            None,
            None,
            weight=view_208,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_142 = view_208 = None
        weight_142 = batch_norm_142.reshape_as(
            l_self_modules_stages_modules_3_modules_6_modules_conv2b_parameters_weight_
        )
        batch_norm_142 = (
            l_self_modules_stages_modules_3_modules_6_modules_conv2b_parameters_weight_
        ) = None
        out_234 = torch.conv2d(
            silu_137,
            weight_142,
            l_self_modules_stages_modules_3_modules_6_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_137 = (
            weight_142
        ) = (
            l_self_modules_stages_modules_3_modules_6_modules_conv2b_parameters_bias_
        ) = None
        silu_138 = torch.nn.functional.silu(out_234, inplace=False)
        out_234 = None
        reshape_143 = l_self_modules_stages_modules_3_modules_6_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_276 = (
            l_self_modules_stages_modules_3_modules_6_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_3_modules_6_modules_conv3_parameters_gain_ = None
        view_209 = mul_276.view(-1)
        mul_276 = None
        batch_norm_143 = torch.nn.functional.batch_norm(
            reshape_143,
            None,
            None,
            weight=view_209,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_143 = view_209 = None
        weight_143 = batch_norm_143.reshape_as(
            l_self_modules_stages_modules_3_modules_6_modules_conv3_parameters_weight_
        )
        batch_norm_143 = (
            l_self_modules_stages_modules_3_modules_6_modules_conv3_parameters_weight_
        ) = None
        out_235 = torch.conv2d(
            silu_138,
            weight_143,
            l_self_modules_stages_modules_3_modules_6_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_138 = (
            weight_143
        ) = (
            l_self_modules_stages_modules_3_modules_6_modules_conv3_parameters_bias_
        ) = None
        mean_33 = out_235.mean((2, 3))
        y_99 = mean_33.view(1, 1, -1)
        mean_33 = None
        y_100 = torch.conv1d(
            y_99,
            l_self_modules_stages_modules_3_modules_6_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_99 = l_self_modules_stages_modules_3_modules_6_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_33 = y_100.sigmoid()
        y_100 = None
        y_101 = sigmoid_33.view(1, -1, 1, 1)
        sigmoid_33 = None
        expand_as_33 = y_101.expand_as(out_235)
        y_101 = None
        mul_277 = out_235 * expand_as_33
        out_235 = expand_as_33 = None
        out_236 = 2.0 * mul_277
        mul_277 = None
        mul_279 = out_236 * 0.2
        out_236 = None
        out_237 = mul_279 + out_230
        mul_279 = out_230 = None
        silu_139 = torch.nn.functional.silu(out_237, inplace=False)
        out_238 = silu_139 * 0.8838834764831842
        silu_139 = None
        reshape_144 = l_self_modules_stages_modules_3_modules_7_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_281 = (
            l_self_modules_stages_modules_3_modules_7_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_3_modules_7_modules_conv1_parameters_gain_ = None
        view_212 = mul_281.view(-1)
        mul_281 = None
        batch_norm_144 = torch.nn.functional.batch_norm(
            reshape_144,
            None,
            None,
            weight=view_212,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_144 = view_212 = None
        weight_144 = batch_norm_144.reshape_as(
            l_self_modules_stages_modules_3_modules_7_modules_conv1_parameters_weight_
        )
        batch_norm_144 = (
            l_self_modules_stages_modules_3_modules_7_modules_conv1_parameters_weight_
        ) = None
        out_239 = torch.conv2d(
            out_238,
            weight_144,
            l_self_modules_stages_modules_3_modules_7_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_238 = (
            weight_144
        ) = (
            l_self_modules_stages_modules_3_modules_7_modules_conv1_parameters_bias_
        ) = None
        silu_140 = torch.nn.functional.silu(out_239, inplace=True)
        out_239 = None
        reshape_145 = l_self_modules_stages_modules_3_modules_7_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_282 = (
            l_self_modules_stages_modules_3_modules_7_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_3_modules_7_modules_conv2_parameters_gain_ = None
        view_213 = mul_282.view(-1)
        mul_282 = None
        batch_norm_145 = torch.nn.functional.batch_norm(
            reshape_145,
            None,
            None,
            weight=view_213,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_145 = view_213 = None
        weight_145 = batch_norm_145.reshape_as(
            l_self_modules_stages_modules_3_modules_7_modules_conv2_parameters_weight_
        )
        batch_norm_145 = (
            l_self_modules_stages_modules_3_modules_7_modules_conv2_parameters_weight_
        ) = None
        out_240 = torch.conv2d(
            silu_140,
            weight_145,
            l_self_modules_stages_modules_3_modules_7_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_140 = (
            weight_145
        ) = (
            l_self_modules_stages_modules_3_modules_7_modules_conv2_parameters_bias_
        ) = None
        silu_141 = torch.nn.functional.silu(out_240, inplace=True)
        out_240 = None
        reshape_146 = l_self_modules_stages_modules_3_modules_7_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_283 = (
            l_self_modules_stages_modules_3_modules_7_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_3_modules_7_modules_conv2b_parameters_gain_ = None
        view_214 = mul_283.view(-1)
        mul_283 = None
        batch_norm_146 = torch.nn.functional.batch_norm(
            reshape_146,
            None,
            None,
            weight=view_214,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_146 = view_214 = None
        weight_146 = batch_norm_146.reshape_as(
            l_self_modules_stages_modules_3_modules_7_modules_conv2b_parameters_weight_
        )
        batch_norm_146 = (
            l_self_modules_stages_modules_3_modules_7_modules_conv2b_parameters_weight_
        ) = None
        out_241 = torch.conv2d(
            silu_141,
            weight_146,
            l_self_modules_stages_modules_3_modules_7_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_141 = (
            weight_146
        ) = (
            l_self_modules_stages_modules_3_modules_7_modules_conv2b_parameters_bias_
        ) = None
        silu_142 = torch.nn.functional.silu(out_241, inplace=False)
        out_241 = None
        reshape_147 = l_self_modules_stages_modules_3_modules_7_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_284 = (
            l_self_modules_stages_modules_3_modules_7_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_3_modules_7_modules_conv3_parameters_gain_ = None
        view_215 = mul_284.view(-1)
        mul_284 = None
        batch_norm_147 = torch.nn.functional.batch_norm(
            reshape_147,
            None,
            None,
            weight=view_215,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_147 = view_215 = None
        weight_147 = batch_norm_147.reshape_as(
            l_self_modules_stages_modules_3_modules_7_modules_conv3_parameters_weight_
        )
        batch_norm_147 = (
            l_self_modules_stages_modules_3_modules_7_modules_conv3_parameters_weight_
        ) = None
        out_242 = torch.conv2d(
            silu_142,
            weight_147,
            l_self_modules_stages_modules_3_modules_7_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_142 = (
            weight_147
        ) = (
            l_self_modules_stages_modules_3_modules_7_modules_conv3_parameters_bias_
        ) = None
        mean_34 = out_242.mean((2, 3))
        y_102 = mean_34.view(1, 1, -1)
        mean_34 = None
        y_103 = torch.conv1d(
            y_102,
            l_self_modules_stages_modules_3_modules_7_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_102 = l_self_modules_stages_modules_3_modules_7_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_34 = y_103.sigmoid()
        y_103 = None
        y_104 = sigmoid_34.view(1, -1, 1, 1)
        sigmoid_34 = None
        expand_as_34 = y_104.expand_as(out_242)
        y_104 = None
        mul_285 = out_242 * expand_as_34
        out_242 = expand_as_34 = None
        out_243 = 2.0 * mul_285
        mul_285 = None
        mul_287 = out_243 * 0.2
        out_243 = None
        out_244 = mul_287 + out_237
        mul_287 = out_237 = None
        silu_143 = torch.nn.functional.silu(out_244, inplace=False)
        out_245 = silu_143 * 0.8703882797784891
        silu_143 = None
        reshape_148 = l_self_modules_stages_modules_3_modules_8_modules_conv1_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_289 = (
            l_self_modules_stages_modules_3_modules_8_modules_conv1_parameters_gain_
            * 0.04562504637317021
        )
        l_self_modules_stages_modules_3_modules_8_modules_conv1_parameters_gain_ = None
        view_218 = mul_289.view(-1)
        mul_289 = None
        batch_norm_148 = torch.nn.functional.batch_norm(
            reshape_148,
            None,
            None,
            weight=view_218,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_148 = view_218 = None
        weight_148 = batch_norm_148.reshape_as(
            l_self_modules_stages_modules_3_modules_8_modules_conv1_parameters_weight_
        )
        batch_norm_148 = (
            l_self_modules_stages_modules_3_modules_8_modules_conv1_parameters_weight_
        ) = None
        out_246 = torch.conv2d(
            out_245,
            weight_148,
            l_self_modules_stages_modules_3_modules_8_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_245 = (
            weight_148
        ) = (
            l_self_modules_stages_modules_3_modules_8_modules_conv1_parameters_bias_
        ) = None
        silu_144 = torch.nn.functional.silu(out_246, inplace=True)
        out_246 = None
        reshape_149 = l_self_modules_stages_modules_3_modules_8_modules_conv2_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_290 = (
            l_self_modules_stages_modules_3_modules_8_modules_conv2_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_3_modules_8_modules_conv2_parameters_gain_ = None
        view_219 = mul_290.view(-1)
        mul_290 = None
        batch_norm_149 = torch.nn.functional.batch_norm(
            reshape_149,
            None,
            None,
            weight=view_219,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_149 = view_219 = None
        weight_149 = batch_norm_149.reshape_as(
            l_self_modules_stages_modules_3_modules_8_modules_conv2_parameters_weight_
        )
        batch_norm_149 = (
            l_self_modules_stages_modules_3_modules_8_modules_conv2_parameters_weight_
        ) = None
        out_247 = torch.conv2d(
            silu_144,
            weight_149,
            l_self_modules_stages_modules_3_modules_8_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_144 = (
            weight_149
        ) = (
            l_self_modules_stages_modules_3_modules_8_modules_conv2_parameters_bias_
        ) = None
        silu_145 = torch.nn.functional.silu(out_247, inplace=True)
        out_247 = None
        reshape_150 = l_self_modules_stages_modules_3_modules_8_modules_conv2b_parameters_weight_.reshape(
            1, 384, -1
        )
        mul_291 = (
            l_self_modules_stages_modules_3_modules_8_modules_conv2b_parameters_gain_
            * 0.07450538873672485
        )
        l_self_modules_stages_modules_3_modules_8_modules_conv2b_parameters_gain_ = None
        view_220 = mul_291.view(-1)
        mul_291 = None
        batch_norm_150 = torch.nn.functional.batch_norm(
            reshape_150,
            None,
            None,
            weight=view_220,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_150 = view_220 = None
        weight_150 = batch_norm_150.reshape_as(
            l_self_modules_stages_modules_3_modules_8_modules_conv2b_parameters_weight_
        )
        batch_norm_150 = (
            l_self_modules_stages_modules_3_modules_8_modules_conv2b_parameters_weight_
        ) = None
        out_248 = torch.conv2d(
            silu_145,
            weight_150,
            l_self_modules_stages_modules_3_modules_8_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        silu_145 = (
            weight_150
        ) = (
            l_self_modules_stages_modules_3_modules_8_modules_conv2b_parameters_bias_
        ) = None
        silu_146 = torch.nn.functional.silu(out_248, inplace=False)
        out_248 = None
        reshape_151 = l_self_modules_stages_modules_3_modules_8_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_292 = (
            l_self_modules_stages_modules_3_modules_8_modules_conv3_parameters_gain_
            * 0.09125009274634042
        )
        l_self_modules_stages_modules_3_modules_8_modules_conv3_parameters_gain_ = None
        view_221 = mul_292.view(-1)
        mul_292 = None
        batch_norm_151 = torch.nn.functional.batch_norm(
            reshape_151,
            None,
            None,
            weight=view_221,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_151 = view_221 = None
        weight_151 = batch_norm_151.reshape_as(
            l_self_modules_stages_modules_3_modules_8_modules_conv3_parameters_weight_
        )
        batch_norm_151 = (
            l_self_modules_stages_modules_3_modules_8_modules_conv3_parameters_weight_
        ) = None
        out_249 = torch.conv2d(
            silu_146,
            weight_151,
            l_self_modules_stages_modules_3_modules_8_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_146 = (
            weight_151
        ) = (
            l_self_modules_stages_modules_3_modules_8_modules_conv3_parameters_bias_
        ) = None
        mean_35 = out_249.mean((2, 3))
        y_105 = mean_35.view(1, 1, -1)
        mean_35 = None
        y_106 = torch.conv1d(
            y_105,
            l_self_modules_stages_modules_3_modules_8_modules_attn_last_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_105 = l_self_modules_stages_modules_3_modules_8_modules_attn_last_modules_conv_parameters_weight_ = (None)
        sigmoid_35 = y_106.sigmoid()
        y_106 = None
        y_107 = sigmoid_35.view(1, -1, 1, 1)
        sigmoid_35 = None
        expand_as_35 = y_107.expand_as(out_249)
        y_107 = None
        mul_293 = out_249 * expand_as_35
        out_249 = expand_as_35 = None
        out_250 = 2.0 * mul_293
        mul_293 = None
        mul_295 = out_250 * 0.2
        out_250 = None
        out_251 = mul_295 + out_244
        mul_295 = out_244 = None
        reshape_152 = l_self_modules_final_conv_parameters_weight_.reshape(1, 3072, -1)
        mul_296 = l_self_modules_final_conv_parameters_gain_ * 0.04562504637317021
        l_self_modules_final_conv_parameters_gain_ = None
        view_224 = mul_296.view(-1)
        mul_296 = None
        batch_norm_152 = torch.nn.functional.batch_norm(
            reshape_152,
            None,
            None,
            weight=view_224,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_152 = view_224 = None
        weight_152 = batch_norm_152.reshape_as(
            l_self_modules_final_conv_parameters_weight_
        )
        batch_norm_152 = l_self_modules_final_conv_parameters_weight_ = None
        x = torch.conv2d(
            out_251,
            weight_152,
            l_self_modules_final_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_251 = weight_152 = l_self_modules_final_conv_parameters_bias_ = None
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
