import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_0_modules_0_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_0_modules_1_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_0_modules_2_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_2_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_3_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_4_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_5_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_6_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_7_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_8_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_9_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_10_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_11_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_12_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_13_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_14_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_15_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_16_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_17_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_3_modules_3_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_3_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_3_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_3_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_3_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_3_modules_4_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_4_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_4_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_4_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_4_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_3_modules_5_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_5_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_5_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_5_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_5_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_3_modules_6_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_6_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_6_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_6_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_6_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_3_modules_7_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_7_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_7_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_7_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_7_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_3_modules_8_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_8_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_8_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_8_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_8_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_9_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_9_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_9_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_9_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_9_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_9_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_9_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_9_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_9_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_9_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_9_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_9_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_9_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_9_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_9_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_9_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_9_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_10_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_10_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_10_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_10_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_10_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_10_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_10_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_10_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_10_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_10_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_10_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_10_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_10_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_10_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_10_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_10_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_10_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_11_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_11_modules_conv1_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_11_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_11_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_11_modules_conv2_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_11_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_11_modules_conv2b_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_11_modules_conv2b_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_11_modules_conv2b_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_11_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_11_modules_conv3_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_11_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_11_modules_attn_last_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_11_modules_attn_last_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_11_modules_attn_last_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_11_modules_attn_last_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_11_parameters_skipinit_gain_: torch.nn.parameter.Parameter,
        L_self_modules_final_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_final_conv_parameters_gain_: torch.nn.parameter.Parameter,
        L_self_modules_final_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_x_ = L_x_
        l_self_modules_stem_modules_conv1_parameters_weight_ = (
            L_self_modules_stem_modules_conv1_parameters_weight_
        )
        l_self_modules_stem_modules_conv1_parameters_gain_ = (
            L_self_modules_stem_modules_conv1_parameters_gain_
        )
        l_self_modules_stem_modules_conv1_parameters_bias_ = (
            L_self_modules_stem_modules_conv1_parameters_bias_
        )
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
        l_self_modules_stages_modules_0_modules_0_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_0_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_0_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_0_modules_0_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_0_modules_1_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_0_modules_1_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_0_modules_2_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_2_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_2_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_2_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_2_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_0_modules_2_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_0_modules_3_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_0_modules_3_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_3_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_0_modules_3_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_0_modules_3_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_3_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_3_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_0_modules_3_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_3_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_0_modules_3_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_0_modules_3_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_3_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_3_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_0_modules_3_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_3_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_0_modules_3_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_0_modules_3_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_3_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_3_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_0_modules_3_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_3_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_0_modules_3_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_0_modules_3_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_3_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_3_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_3_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_3_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_3_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_3_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_3_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_3_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_3_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_3_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_0_modules_3_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_1_modules_0_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_1_modules_1_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_1_modules_2_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_2_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_2_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_2_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_2_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_1_modules_2_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_1_modules_3_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_3_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_3_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_3_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_3_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_1_modules_3_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_1_modules_4_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_4_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_4_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_4_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_4_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_1_modules_4_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_1_modules_5_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_5_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_5_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_5_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_5_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_1_modules_5_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_1_modules_6_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_6_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_6_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_6_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_6_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_6_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_6_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_6_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_6_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_6_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_6_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_6_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_6_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_6_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_6_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_6_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_6_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_6_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_6_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_6_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_6_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_6_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_6_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_6_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_6_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_6_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_6_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_6_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_6_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_6_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_6_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_6_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_6_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_1_modules_6_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_1_modules_7_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_7_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_7_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_7_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_7_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_7_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_7_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_7_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_7_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_7_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_7_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_7_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_7_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_7_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_7_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_7_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_7_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_7_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_7_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_7_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_7_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_1_modules_7_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_1_modules_7_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_7_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_7_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_7_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_7_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_7_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_7_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_7_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_7_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_7_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_7_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_1_modules_7_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_2_modules_0_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_2_modules_1_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_2_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_2_modules_2_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_3_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_2_modules_3_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_4_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_2_modules_4_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_5_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_2_modules_5_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_2_modules_6_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_6_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_6_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_6_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_6_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_2_modules_6_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_2_modules_7_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_7_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_7_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_7_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_7_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_2_modules_7_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_2_modules_8_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_8_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_8_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_8_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_8_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_2_modules_8_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_2_modules_9_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_9_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_9_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_9_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_9_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_2_modules_9_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_2_modules_10_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_10_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_10_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_10_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_10_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_2_modules_10_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_2_modules_11_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_11_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_11_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_11_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_11_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_2_modules_11_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_2_modules_12_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_12_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_12_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_12_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_12_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_2_modules_12_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_2_modules_13_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_13_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_13_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_13_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_13_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_2_modules_13_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_2_modules_14_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_14_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_14_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_14_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_14_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_2_modules_14_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_2_modules_15_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_15_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_15_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_15_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_15_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_2_modules_15_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_2_modules_16_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_16_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_16_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_16_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_16_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_16_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_16_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_16_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_16_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_2_modules_16_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_2_modules_17_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_17_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_17_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_17_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_17_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_17_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_17_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_17_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_17_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_2_modules_17_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_18_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_18_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_18_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_18_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_18_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_18_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_18_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_18_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_18_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_18_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_18_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_18_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_18_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_18_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_18_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_18_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_18_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_18_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_18_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_18_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_18_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_18_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_18_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_18_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_18_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_18_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_18_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_18_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_18_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_18_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_18_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_18_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_18_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_2_modules_18_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_19_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_19_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_19_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_19_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_19_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_19_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_19_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_19_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_19_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_19_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_19_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_19_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_19_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_19_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_19_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_19_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_19_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_19_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_19_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_19_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_19_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_19_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_19_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_19_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_19_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_19_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_19_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_19_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_19_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_19_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_19_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_19_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_19_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_2_modules_19_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_20_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_20_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_20_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_20_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_20_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_20_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_20_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_20_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_20_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_20_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_20_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_20_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_20_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_20_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_20_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_20_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_20_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_20_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_20_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_20_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_20_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_20_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_20_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_20_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_20_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_20_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_20_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_20_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_20_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_20_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_20_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_20_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_20_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_2_modules_20_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_21_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_21_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_21_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_21_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_21_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_21_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_21_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_21_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_21_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_21_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_21_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_21_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_21_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_21_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_21_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_21_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_21_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_21_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_21_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_21_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_21_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_21_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_21_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_21_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_21_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_21_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_21_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_21_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_21_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_21_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_21_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_21_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_21_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_2_modules_21_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_22_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_22_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_22_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_22_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_22_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_22_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_22_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_22_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_22_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_22_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_22_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_22_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_22_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_22_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_22_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_22_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_22_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_22_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_22_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_22_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_22_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_22_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_22_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_22_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_22_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_22_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_22_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_22_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_22_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_22_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_22_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_22_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_22_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_2_modules_22_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_23_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_23_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_23_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_23_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_23_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_23_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_23_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_23_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_23_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_23_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_23_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_23_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_23_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_23_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_23_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_23_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_23_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_23_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_23_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_23_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_23_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_2_modules_23_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_2_modules_23_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_23_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_23_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_23_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_23_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_23_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_23_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_23_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_23_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_23_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_23_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_2_modules_23_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_3_modules_0_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_3_modules_1_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_2_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_3_modules_2_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_3_modules_3_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_3_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_3_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_3_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_3_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_3_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_3_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_3_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_3_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_3_modules_3_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_3_modules_4_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_4_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_4_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_4_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_4_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_4_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_4_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_4_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_4_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_3_modules_4_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_3_modules_5_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_5_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_5_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_5_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_5_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_5_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_5_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_5_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_5_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_3_modules_5_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_3_modules_6_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_6_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_6_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_6_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_6_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_6_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_6_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_6_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_6_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_3_modules_6_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_3_modules_7_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_7_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_7_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_7_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_7_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_7_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_7_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_7_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_7_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_3_modules_7_parameters_skipinit_gain_
        )
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
        l_self_modules_stages_modules_3_modules_8_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_8_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_8_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_8_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_8_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_8_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_8_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_8_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_8_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_3_modules_8_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_3_modules_9_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_9_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_9_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_9_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_9_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_9_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_9_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_9_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_9_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_9_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_9_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_9_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_9_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_9_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_9_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_9_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_9_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_9_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_9_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_9_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_9_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_9_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_9_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_9_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_9_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_9_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_9_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_9_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_9_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_9_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_9_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_9_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_9_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_3_modules_9_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_3_modules_10_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_10_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_10_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_10_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_10_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_10_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_10_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_10_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_10_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_10_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_10_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_10_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_10_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_10_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_10_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_10_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_10_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_10_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_10_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_10_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_10_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_10_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_10_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_10_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_10_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_10_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_10_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_10_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_10_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_10_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_10_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_10_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_10_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_3_modules_10_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_3_modules_11_modules_conv1_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_11_modules_conv1_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_11_modules_conv1_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_11_modules_conv1_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_11_modules_conv1_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_11_modules_conv1_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_11_modules_conv2_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_11_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_11_modules_conv2_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_11_modules_conv2_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_11_modules_conv2_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_11_modules_conv2_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_11_modules_conv2b_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_11_modules_conv2b_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_11_modules_conv2b_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_11_modules_conv2b_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_11_modules_conv2b_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_11_modules_conv2b_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_11_modules_conv3_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_11_modules_conv3_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_11_modules_conv3_parameters_gain_ = (
            L_self_modules_stages_modules_3_modules_11_modules_conv3_parameters_gain_
        )
        l_self_modules_stages_modules_3_modules_11_modules_conv3_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_11_modules_conv3_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_11_modules_attn_last_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_11_modules_attn_last_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_11_modules_attn_last_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_11_modules_attn_last_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_11_modules_attn_last_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_11_modules_attn_last_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_11_modules_attn_last_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_11_modules_attn_last_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_11_parameters_skipinit_gain_ = (
            L_self_modules_stages_modules_3_modules_11_parameters_skipinit_gain_
        )
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
        x = torch._C._nn.pad(l_x_, (0, 1, 0, 1), "constant", 0)
        l_x_ = None
        reshape = l_self_modules_stem_modules_conv1_parameters_weight_.reshape(
            1, 16, -1
        )
        mul = l_self_modules_stem_modules_conv1_parameters_gain_ * 0.19245008972987526
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
            x,
            weight,
            l_self_modules_stem_modules_conv1_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x = weight = l_self_modules_stem_modules_conv1_parameters_bias_ = None
        gelu = torch._C._nn.gelu(input_1)
        input_1 = None
        input_2 = gelu.mul_(1.7015043497085571)
        gelu = None
        reshape_1 = l_self_modules_stem_modules_conv2_parameters_weight_.reshape(
            1, 32, -1
        )
        mul_1 = l_self_modules_stem_modules_conv2_parameters_gain_ * 0.08333333333333333
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
        gelu_1 = torch._C._nn.gelu(input_3)
        input_3 = None
        input_4 = gelu_1.mul_(1.7015043497085571)
        gelu_1 = None
        reshape_2 = l_self_modules_stem_modules_conv3_parameters_weight_.reshape(
            1, 64, -1
        )
        mul_2 = l_self_modules_stem_modules_conv3_parameters_gain_ * 0.05892556509887896
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
        gelu_2 = torch._C._nn.gelu(input_5)
        input_5 = None
        input_6 = gelu_2.mul_(1.7015043497085571)
        gelu_2 = None
        x_1 = torch._C._nn.pad(input_6, (0, 1, 0, 1), "constant", 0)
        input_6 = None
        reshape_3 = l_self_modules_stem_modules_conv4_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_3 = (
            l_self_modules_stem_modules_conv4_parameters_gain_ * 0.041666666666666664
        )
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
            x_1,
            weight_3,
            l_self_modules_stem_modules_conv4_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_1 = weight_3 = l_self_modules_stem_modules_conv4_parameters_bias_ = None
        gelu_3 = torch._C._nn.gelu(input_7)
        input_7 = None
        mul__3 = gelu_3.mul_(1.7015043497085571)
        gelu_3 = None
        out = mul__3 * 1.0
        mul__3 = None
        reshape_4 = l_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_5 = (
            l_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_gain_
            * 0.08838834764831845
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
            1, 128, -1
        )
        mul_6 = (
            l_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_gain_
            * 0.08838834764831845
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
        gelu_4 = torch._C._nn.gelu(out_1)
        out_1 = None
        mul__4 = gelu_4.mul_(1.7015043497085571)
        gelu_4 = None
        reshape_6 = l_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_7 = (
            l_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_gain_
            * 0.02946278254943948
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
            mul__4,
            weight_6,
            l_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        mul__4 = (
            weight_6
        ) = (
            l_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_bias_
        ) = None
        gelu_5 = torch._C._nn.gelu(out_2)
        out_2 = None
        mul__5 = gelu_5.mul_(1.7015043497085571)
        gelu_5 = None
        reshape_7 = l_self_modules_stages_modules_0_modules_0_modules_conv2b_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_8 = (
            l_self_modules_stages_modules_0_modules_0_modules_conv2b_parameters_gain_
            * 0.02946278254943948
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
            mul__5,
            weight_7,
            l_self_modules_stages_modules_0_modules_0_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        mul__5 = (
            weight_7
        ) = (
            l_self_modules_stages_modules_0_modules_0_modules_conv2b_parameters_bias_
        ) = None
        gelu_6 = torch._C._nn.gelu(out_3)
        out_3 = None
        mul__6 = gelu_6.mul_(1.7015043497085571)
        gelu_6 = None
        reshape_8 = l_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_9 = (
            l_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_gain_
            * 0.08838834764831845
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
            mul__6,
            weight_8,
            l_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__6 = (
            weight_8
        ) = (
            l_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_bias_
        ) = None
        x_se = out_4.mean((2, 3), keepdim=True)
        x_se_1 = torch.conv2d(
            x_se,
            l_self_modules_stages_modules_0_modules_0_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se = l_self_modules_stages_modules_0_modules_0_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_2 = torch.nn.functional.relu(x_se_1, inplace=True)
        x_se_1 = None
        x_se_3 = torch.conv2d(
            x_se_2,
            l_self_modules_stages_modules_0_modules_0_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_2 = l_self_modules_stages_modules_0_modules_0_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid = x_se_3.sigmoid()
        x_se_3 = None
        mul_10 = out_4 * sigmoid
        out_4 = sigmoid = None
        out_5 = 2.0 * mul_10
        mul_10 = None
        mul__7 = out_5.mul_(
            l_self_modules_stages_modules_0_modules_0_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_0_modules_0_parameters_skipinit_gain_ = (
            mul__7
        ) = None
        mul_12 = out_5 * 0.2
        out_5 = None
        out_6 = mul_12 + shortcut
        mul_12 = shortcut = None
        gelu_7 = torch._C._nn.gelu(out_6)
        mul__8 = gelu_7.mul_(1.7015043497085571)
        gelu_7 = None
        out_7 = mul__8 * 0.9805806756909201
        mul__8 = None
        reshape_9 = l_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_14 = (
            l_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_gain_
            * 0.0625
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_gain_ = None
        view_9 = mul_14.view(-1)
        mul_14 = None
        batch_norm_9 = torch.nn.functional.batch_norm(
            reshape_9, None, None, weight=view_9, training=True, momentum=0.0, eps=1e-05
        )
        reshape_9 = view_9 = None
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
        gelu_8 = torch._C._nn.gelu(out_8)
        out_8 = None
        mul__9 = gelu_8.mul_(1.7015043497085571)
        gelu_8 = None
        reshape_10 = l_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_15 = (
            l_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_gain_ = None
        view_10 = mul_15.view(-1)
        mul_15 = None
        batch_norm_10 = torch.nn.functional.batch_norm(
            reshape_10,
            None,
            None,
            weight=view_10,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_10 = view_10 = None
        weight_10 = batch_norm_10.reshape_as(
            l_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_weight_
        )
        batch_norm_10 = (
            l_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_weight_
        ) = None
        out_9 = torch.conv2d(
            mul__9,
            weight_10,
            l_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        mul__9 = (
            weight_10
        ) = (
            l_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_bias_
        ) = None
        gelu_9 = torch._C._nn.gelu(out_9)
        out_9 = None
        mul__10 = gelu_9.mul_(1.7015043497085571)
        gelu_9 = None
        reshape_11 = l_self_modules_stages_modules_0_modules_1_modules_conv2b_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_16 = (
            l_self_modules_stages_modules_0_modules_1_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv2b_parameters_gain_ = None
        view_11 = mul_16.view(-1)
        mul_16 = None
        batch_norm_11 = torch.nn.functional.batch_norm(
            reshape_11,
            None,
            None,
            weight=view_11,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_11 = view_11 = None
        weight_11 = batch_norm_11.reshape_as(
            l_self_modules_stages_modules_0_modules_1_modules_conv2b_parameters_weight_
        )
        batch_norm_11 = (
            l_self_modules_stages_modules_0_modules_1_modules_conv2b_parameters_weight_
        ) = None
        out_10 = torch.conv2d(
            mul__10,
            weight_11,
            l_self_modules_stages_modules_0_modules_1_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        mul__10 = (
            weight_11
        ) = (
            l_self_modules_stages_modules_0_modules_1_modules_conv2b_parameters_bias_
        ) = None
        gelu_10 = torch._C._nn.gelu(out_10)
        out_10 = None
        mul__11 = gelu_10.mul_(1.7015043497085571)
        gelu_10 = None
        reshape_12 = l_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_17 = (
            l_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_gain_
            * 0.08838834764831845
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_gain_ = None
        view_12 = mul_17.view(-1)
        mul_17 = None
        batch_norm_12 = torch.nn.functional.batch_norm(
            reshape_12,
            None,
            None,
            weight=view_12,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_12 = view_12 = None
        weight_12 = batch_norm_12.reshape_as(
            l_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_weight_
        )
        batch_norm_12 = (
            l_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_weight_
        ) = None
        out_11 = torch.conv2d(
            mul__11,
            weight_12,
            l_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__11 = (
            weight_12
        ) = (
            l_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_bias_
        ) = None
        x_se_4 = out_11.mean((2, 3), keepdim=True)
        x_se_5 = torch.conv2d(
            x_se_4,
            l_self_modules_stages_modules_0_modules_1_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_4 = l_self_modules_stages_modules_0_modules_1_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_6 = torch.nn.functional.relu(x_se_5, inplace=True)
        x_se_5 = None
        x_se_7 = torch.conv2d(
            x_se_6,
            l_self_modules_stages_modules_0_modules_1_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_6 = l_self_modules_stages_modules_0_modules_1_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_1 = x_se_7.sigmoid()
        x_se_7 = None
        mul_18 = out_11 * sigmoid_1
        out_11 = sigmoid_1 = None
        out_12 = 2.0 * mul_18
        mul_18 = None
        mul__12 = out_12.mul_(
            l_self_modules_stages_modules_0_modules_1_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_0_modules_1_parameters_skipinit_gain_ = (
            mul__12
        ) = None
        mul_20 = out_12 * 0.2
        out_12 = None
        out_13 = mul_20 + out_6
        mul_20 = out_6 = None
        gelu_11 = torch._C._nn.gelu(out_13)
        mul__13 = gelu_11.mul_(1.7015043497085571)
        gelu_11 = None
        out_14 = mul__13 * 0.9622504486493761
        mul__13 = None
        reshape_13 = l_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_22 = (
            l_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_gain_
            * 0.0625
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_gain_ = None
        view_13 = mul_22.view(-1)
        mul_22 = None
        batch_norm_13 = torch.nn.functional.batch_norm(
            reshape_13,
            None,
            None,
            weight=view_13,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_13 = view_13 = None
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
        gelu_12 = torch._C._nn.gelu(out_15)
        out_15 = None
        mul__14 = gelu_12.mul_(1.7015043497085571)
        gelu_12 = None
        reshape_14 = l_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_23 = (
            l_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_gain_ = None
        view_14 = mul_23.view(-1)
        mul_23 = None
        batch_norm_14 = torch.nn.functional.batch_norm(
            reshape_14,
            None,
            None,
            weight=view_14,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_14 = view_14 = None
        weight_14 = batch_norm_14.reshape_as(
            l_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_weight_
        )
        batch_norm_14 = (
            l_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_weight_
        ) = None
        out_16 = torch.conv2d(
            mul__14,
            weight_14,
            l_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        mul__14 = (
            weight_14
        ) = (
            l_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_bias_
        ) = None
        gelu_13 = torch._C._nn.gelu(out_16)
        out_16 = None
        mul__15 = gelu_13.mul_(1.7015043497085571)
        gelu_13 = None
        reshape_15 = l_self_modules_stages_modules_0_modules_2_modules_conv2b_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_24 = (
            l_self_modules_stages_modules_0_modules_2_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv2b_parameters_gain_ = None
        view_15 = mul_24.view(-1)
        mul_24 = None
        batch_norm_15 = torch.nn.functional.batch_norm(
            reshape_15,
            None,
            None,
            weight=view_15,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_15 = view_15 = None
        weight_15 = batch_norm_15.reshape_as(
            l_self_modules_stages_modules_0_modules_2_modules_conv2b_parameters_weight_
        )
        batch_norm_15 = (
            l_self_modules_stages_modules_0_modules_2_modules_conv2b_parameters_weight_
        ) = None
        out_17 = torch.conv2d(
            mul__15,
            weight_15,
            l_self_modules_stages_modules_0_modules_2_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        mul__15 = (
            weight_15
        ) = (
            l_self_modules_stages_modules_0_modules_2_modules_conv2b_parameters_bias_
        ) = None
        gelu_14 = torch._C._nn.gelu(out_17)
        out_17 = None
        mul__16 = gelu_14.mul_(1.7015043497085571)
        gelu_14 = None
        reshape_16 = l_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_25 = (
            l_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_gain_
            * 0.08838834764831845
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_gain_ = None
        view_16 = mul_25.view(-1)
        mul_25 = None
        batch_norm_16 = torch.nn.functional.batch_norm(
            reshape_16,
            None,
            None,
            weight=view_16,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_16 = view_16 = None
        weight_16 = batch_norm_16.reshape_as(
            l_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_weight_
        )
        batch_norm_16 = (
            l_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_weight_
        ) = None
        out_18 = torch.conv2d(
            mul__16,
            weight_16,
            l_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__16 = (
            weight_16
        ) = (
            l_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_bias_
        ) = None
        x_se_8 = out_18.mean((2, 3), keepdim=True)
        x_se_9 = torch.conv2d(
            x_se_8,
            l_self_modules_stages_modules_0_modules_2_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_2_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_8 = l_self_modules_stages_modules_0_modules_2_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_2_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_10 = torch.nn.functional.relu(x_se_9, inplace=True)
        x_se_9 = None
        x_se_11 = torch.conv2d(
            x_se_10,
            l_self_modules_stages_modules_0_modules_2_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_2_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_10 = l_self_modules_stages_modules_0_modules_2_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_2_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_2 = x_se_11.sigmoid()
        x_se_11 = None
        mul_26 = out_18 * sigmoid_2
        out_18 = sigmoid_2 = None
        out_19 = 2.0 * mul_26
        mul_26 = None
        mul__17 = out_19.mul_(
            l_self_modules_stages_modules_0_modules_2_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_0_modules_2_parameters_skipinit_gain_ = (
            mul__17
        ) = None
        mul_28 = out_19 * 0.2
        out_19 = None
        out_20 = mul_28 + out_13
        mul_28 = out_13 = None
        gelu_15 = torch._C._nn.gelu(out_20)
        mul__18 = gelu_15.mul_(1.7015043497085571)
        gelu_15 = None
        out_21 = mul__18 * 0.9449111825230679
        mul__18 = None
        reshape_17 = l_self_modules_stages_modules_0_modules_3_modules_conv1_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_30 = (
            l_self_modules_stages_modules_0_modules_3_modules_conv1_parameters_gain_
            * 0.0625
        )
        l_self_modules_stages_modules_0_modules_3_modules_conv1_parameters_gain_ = None
        view_17 = mul_30.view(-1)
        mul_30 = None
        batch_norm_17 = torch.nn.functional.batch_norm(
            reshape_17,
            None,
            None,
            weight=view_17,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_17 = view_17 = None
        weight_17 = batch_norm_17.reshape_as(
            l_self_modules_stages_modules_0_modules_3_modules_conv1_parameters_weight_
        )
        batch_norm_17 = (
            l_self_modules_stages_modules_0_modules_3_modules_conv1_parameters_weight_
        ) = None
        out_22 = torch.conv2d(
            out_21,
            weight_17,
            l_self_modules_stages_modules_0_modules_3_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_21 = (
            weight_17
        ) = (
            l_self_modules_stages_modules_0_modules_3_modules_conv1_parameters_bias_
        ) = None
        gelu_16 = torch._C._nn.gelu(out_22)
        out_22 = None
        mul__19 = gelu_16.mul_(1.7015043497085571)
        gelu_16 = None
        reshape_18 = l_self_modules_stages_modules_0_modules_3_modules_conv2_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_31 = (
            l_self_modules_stages_modules_0_modules_3_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_0_modules_3_modules_conv2_parameters_gain_ = None
        view_18 = mul_31.view(-1)
        mul_31 = None
        batch_norm_18 = torch.nn.functional.batch_norm(
            reshape_18,
            None,
            None,
            weight=view_18,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_18 = view_18 = None
        weight_18 = batch_norm_18.reshape_as(
            l_self_modules_stages_modules_0_modules_3_modules_conv2_parameters_weight_
        )
        batch_norm_18 = (
            l_self_modules_stages_modules_0_modules_3_modules_conv2_parameters_weight_
        ) = None
        out_23 = torch.conv2d(
            mul__19,
            weight_18,
            l_self_modules_stages_modules_0_modules_3_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        mul__19 = (
            weight_18
        ) = (
            l_self_modules_stages_modules_0_modules_3_modules_conv2_parameters_bias_
        ) = None
        gelu_17 = torch._C._nn.gelu(out_23)
        out_23 = None
        mul__20 = gelu_17.mul_(1.7015043497085571)
        gelu_17 = None
        reshape_19 = l_self_modules_stages_modules_0_modules_3_modules_conv2b_parameters_weight_.reshape(
            1, 128, -1
        )
        mul_32 = (
            l_self_modules_stages_modules_0_modules_3_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_0_modules_3_modules_conv2b_parameters_gain_ = None
        view_19 = mul_32.view(-1)
        mul_32 = None
        batch_norm_19 = torch.nn.functional.batch_norm(
            reshape_19,
            None,
            None,
            weight=view_19,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_19 = view_19 = None
        weight_19 = batch_norm_19.reshape_as(
            l_self_modules_stages_modules_0_modules_3_modules_conv2b_parameters_weight_
        )
        batch_norm_19 = (
            l_self_modules_stages_modules_0_modules_3_modules_conv2b_parameters_weight_
        ) = None
        out_24 = torch.conv2d(
            mul__20,
            weight_19,
            l_self_modules_stages_modules_0_modules_3_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        mul__20 = (
            weight_19
        ) = (
            l_self_modules_stages_modules_0_modules_3_modules_conv2b_parameters_bias_
        ) = None
        gelu_18 = torch._C._nn.gelu(out_24)
        out_24 = None
        mul__21 = gelu_18.mul_(1.7015043497085571)
        gelu_18 = None
        reshape_20 = l_self_modules_stages_modules_0_modules_3_modules_conv3_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_33 = (
            l_self_modules_stages_modules_0_modules_3_modules_conv3_parameters_gain_
            * 0.08838834764831845
        )
        l_self_modules_stages_modules_0_modules_3_modules_conv3_parameters_gain_ = None
        view_20 = mul_33.view(-1)
        mul_33 = None
        batch_norm_20 = torch.nn.functional.batch_norm(
            reshape_20,
            None,
            None,
            weight=view_20,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_20 = view_20 = None
        weight_20 = batch_norm_20.reshape_as(
            l_self_modules_stages_modules_0_modules_3_modules_conv3_parameters_weight_
        )
        batch_norm_20 = (
            l_self_modules_stages_modules_0_modules_3_modules_conv3_parameters_weight_
        ) = None
        out_25 = torch.conv2d(
            mul__21,
            weight_20,
            l_self_modules_stages_modules_0_modules_3_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__21 = (
            weight_20
        ) = (
            l_self_modules_stages_modules_0_modules_3_modules_conv3_parameters_bias_
        ) = None
        x_se_12 = out_25.mean((2, 3), keepdim=True)
        x_se_13 = torch.conv2d(
            x_se_12,
            l_self_modules_stages_modules_0_modules_3_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_3_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_12 = l_self_modules_stages_modules_0_modules_3_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_3_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_14 = torch.nn.functional.relu(x_se_13, inplace=True)
        x_se_13 = None
        x_se_15 = torch.conv2d(
            x_se_14,
            l_self_modules_stages_modules_0_modules_3_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_3_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_14 = l_self_modules_stages_modules_0_modules_3_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_3_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_3 = x_se_15.sigmoid()
        x_se_15 = None
        mul_34 = out_25 * sigmoid_3
        out_25 = sigmoid_3 = None
        out_26 = 2.0 * mul_34
        mul_34 = None
        mul__22 = out_26.mul_(
            l_self_modules_stages_modules_0_modules_3_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_0_modules_3_parameters_skipinit_gain_ = (
            mul__22
        ) = None
        mul_36 = out_26 * 0.2
        out_26 = None
        out_27 = mul_36 + out_20
        mul_36 = out_20 = None
        gelu_19 = torch._C._nn.gelu(out_27)
        out_27 = None
        mul__23 = gelu_19.mul_(1.7015043497085571)
        gelu_19 = None
        out_28 = mul__23 * 0.9284766908852592
        mul__23 = None
        avg_pool2d = torch._C._nn.avg_pool2d(out_28, 2, 2, 0, True, False, None)
        reshape_21 = l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_38 = (
            l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_gain_
            * 0.0625
        )
        l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_gain_ = (
            None
        )
        view_21 = mul_38.view(-1)
        mul_38 = None
        batch_norm_21 = torch.nn.functional.batch_norm(
            reshape_21,
            None,
            None,
            weight=view_21,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_21 = view_21 = None
        weight_21 = batch_norm_21.reshape_as(
            l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_
        )
        batch_norm_21 = l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        shortcut_1 = torch.conv2d(
            avg_pool2d,
            weight_21,
            l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        avg_pool2d = (
            weight_21
        ) = l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_bias_ = (None)
        reshape_22 = l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_39 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_gain_
            * 0.0625
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_gain_ = None
        view_22 = mul_39.view(-1)
        mul_39 = None
        batch_norm_22 = torch.nn.functional.batch_norm(
            reshape_22,
            None,
            None,
            weight=view_22,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_22 = view_22 = None
        weight_22 = batch_norm_22.reshape_as(
            l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_weight_
        )
        batch_norm_22 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_weight_
        ) = None
        out_29 = torch.conv2d(
            out_28,
            weight_22,
            l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_28 = (
            weight_22
        ) = (
            l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_bias_
        ) = None
        gelu_20 = torch._C._nn.gelu(out_29)
        out_29 = None
        mul__24 = gelu_20.mul_(1.7015043497085571)
        gelu_20 = None
        x_2 = torch._C._nn.pad(mul__24, (0, 1, 0, 1), "constant", 0)
        mul__24 = None
        reshape_23 = l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_40 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_gain_ = None
        view_23 = mul_40.view(-1)
        mul_40 = None
        batch_norm_23 = torch.nn.functional.batch_norm(
            reshape_23,
            None,
            None,
            weight=view_23,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_23 = view_23 = None
        weight_23 = batch_norm_23.reshape_as(
            l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_weight_
        )
        batch_norm_23 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_weight_
        ) = None
        out_30 = torch.conv2d(
            x_2,
            weight_23,
            l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            2,
        )
        x_2 = (
            weight_23
        ) = (
            l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_bias_
        ) = None
        gelu_21 = torch._C._nn.gelu(out_30)
        out_30 = None
        mul__25 = gelu_21.mul_(1.7015043497085571)
        gelu_21 = None
        reshape_24 = l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_41 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_gain_ = None
        view_24 = mul_41.view(-1)
        mul_41 = None
        batch_norm_24 = torch.nn.functional.batch_norm(
            reshape_24,
            None,
            None,
            weight=view_24,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_24 = view_24 = None
        weight_24 = batch_norm_24.reshape_as(
            l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_weight_
        )
        batch_norm_24 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_weight_
        ) = None
        out_31 = torch.conv2d(
            mul__25,
            weight_24,
            l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        mul__25 = (
            weight_24
        ) = (
            l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_bias_
        ) = None
        gelu_22 = torch._C._nn.gelu(out_31)
        out_31 = None
        mul__26 = gelu_22.mul_(1.7015043497085571)
        gelu_22 = None
        reshape_25 = l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_42 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_gain_
            * 0.0625
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_gain_ = None
        view_25 = mul_42.view(-1)
        mul_42 = None
        batch_norm_25 = torch.nn.functional.batch_norm(
            reshape_25,
            None,
            None,
            weight=view_25,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_25 = view_25 = None
        weight_25 = batch_norm_25.reshape_as(
            l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_weight_
        )
        batch_norm_25 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_weight_
        ) = None
        out_32 = torch.conv2d(
            mul__26,
            weight_25,
            l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__26 = (
            weight_25
        ) = (
            l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_bias_
        ) = None
        x_se_16 = out_32.mean((2, 3), keepdim=True)
        x_se_17 = torch.conv2d(
            x_se_16,
            l_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_16 = l_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_18 = torch.nn.functional.relu(x_se_17, inplace=True)
        x_se_17 = None
        x_se_19 = torch.conv2d(
            x_se_18,
            l_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_18 = l_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_4 = x_se_19.sigmoid()
        x_se_19 = None
        mul_43 = out_32 * sigmoid_4
        out_32 = sigmoid_4 = None
        out_33 = 2.0 * mul_43
        mul_43 = None
        mul__27 = out_33.mul_(
            l_self_modules_stages_modules_1_modules_0_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_1_modules_0_parameters_skipinit_gain_ = (
            mul__27
        ) = None
        mul_45 = out_33 * 0.2
        out_33 = None
        out_34 = mul_45 + shortcut_1
        mul_45 = shortcut_1 = None
        gelu_23 = torch._C._nn.gelu(out_34)
        mul__28 = gelu_23.mul_(1.7015043497085571)
        gelu_23 = None
        out_35 = mul__28 * 0.9805806756909201
        mul__28 = None
        reshape_26 = l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_47 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_gain_
            * 0.04419417382415922
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_gain_ = None
        view_26 = mul_47.view(-1)
        mul_47 = None
        batch_norm_26 = torch.nn.functional.batch_norm(
            reshape_26,
            None,
            None,
            weight=view_26,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_26 = view_26 = None
        weight_26 = batch_norm_26.reshape_as(
            l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_weight_
        )
        batch_norm_26 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_weight_
        ) = None
        out_36 = torch.conv2d(
            out_35,
            weight_26,
            l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_35 = (
            weight_26
        ) = (
            l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_bias_
        ) = None
        gelu_24 = torch._C._nn.gelu(out_36)
        out_36 = None
        mul__29 = gelu_24.mul_(1.7015043497085571)
        gelu_24 = None
        reshape_27 = l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_48 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_gain_ = None
        view_27 = mul_48.view(-1)
        mul_48 = None
        batch_norm_27 = torch.nn.functional.batch_norm(
            reshape_27,
            None,
            None,
            weight=view_27,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_27 = view_27 = None
        weight_27 = batch_norm_27.reshape_as(
            l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_weight_
        )
        batch_norm_27 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_weight_
        ) = None
        out_37 = torch.conv2d(
            mul__29,
            weight_27,
            l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        mul__29 = (
            weight_27
        ) = (
            l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_bias_
        ) = None
        gelu_25 = torch._C._nn.gelu(out_37)
        out_37 = None
        mul__30 = gelu_25.mul_(1.7015043497085571)
        gelu_25 = None
        reshape_28 = l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_49 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_gain_ = None
        view_28 = mul_49.view(-1)
        mul_49 = None
        batch_norm_28 = torch.nn.functional.batch_norm(
            reshape_28,
            None,
            None,
            weight=view_28,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_28 = view_28 = None
        weight_28 = batch_norm_28.reshape_as(
            l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_weight_
        )
        batch_norm_28 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_weight_
        ) = None
        out_38 = torch.conv2d(
            mul__30,
            weight_28,
            l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        mul__30 = (
            weight_28
        ) = (
            l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_bias_
        ) = None
        gelu_26 = torch._C._nn.gelu(out_38)
        out_38 = None
        mul__31 = gelu_26.mul_(1.7015043497085571)
        gelu_26 = None
        reshape_29 = l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_50 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_gain_
            * 0.0625
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_gain_ = None
        view_29 = mul_50.view(-1)
        mul_50 = None
        batch_norm_29 = torch.nn.functional.batch_norm(
            reshape_29,
            None,
            None,
            weight=view_29,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_29 = view_29 = None
        weight_29 = batch_norm_29.reshape_as(
            l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_weight_
        )
        batch_norm_29 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_weight_
        ) = None
        out_39 = torch.conv2d(
            mul__31,
            weight_29,
            l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__31 = (
            weight_29
        ) = (
            l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_bias_
        ) = None
        x_se_20 = out_39.mean((2, 3), keepdim=True)
        x_se_21 = torch.conv2d(
            x_se_20,
            l_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_20 = l_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_22 = torch.nn.functional.relu(x_se_21, inplace=True)
        x_se_21 = None
        x_se_23 = torch.conv2d(
            x_se_22,
            l_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_22 = l_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_5 = x_se_23.sigmoid()
        x_se_23 = None
        mul_51 = out_39 * sigmoid_5
        out_39 = sigmoid_5 = None
        out_40 = 2.0 * mul_51
        mul_51 = None
        mul__32 = out_40.mul_(
            l_self_modules_stages_modules_1_modules_1_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_1_modules_1_parameters_skipinit_gain_ = (
            mul__32
        ) = None
        mul_53 = out_40 * 0.2
        out_40 = None
        out_41 = mul_53 + out_34
        mul_53 = out_34 = None
        gelu_27 = torch._C._nn.gelu(out_41)
        mul__33 = gelu_27.mul_(1.7015043497085571)
        gelu_27 = None
        out_42 = mul__33 * 0.9622504486493761
        mul__33 = None
        reshape_30 = l_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_55 = (
            l_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_gain_
            * 0.04419417382415922
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_gain_ = None
        view_30 = mul_55.view(-1)
        mul_55 = None
        batch_norm_30 = torch.nn.functional.batch_norm(
            reshape_30,
            None,
            None,
            weight=view_30,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_30 = view_30 = None
        weight_30 = batch_norm_30.reshape_as(
            l_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_weight_
        )
        batch_norm_30 = (
            l_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_weight_
        ) = None
        out_43 = torch.conv2d(
            out_42,
            weight_30,
            l_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_42 = (
            weight_30
        ) = (
            l_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_bias_
        ) = None
        gelu_28 = torch._C._nn.gelu(out_43)
        out_43 = None
        mul__34 = gelu_28.mul_(1.7015043497085571)
        gelu_28 = None
        reshape_31 = l_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_56 = (
            l_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_gain_ = None
        view_31 = mul_56.view(-1)
        mul_56 = None
        batch_norm_31 = torch.nn.functional.batch_norm(
            reshape_31,
            None,
            None,
            weight=view_31,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_31 = view_31 = None
        weight_31 = batch_norm_31.reshape_as(
            l_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_weight_
        )
        batch_norm_31 = (
            l_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_weight_
        ) = None
        out_44 = torch.conv2d(
            mul__34,
            weight_31,
            l_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        mul__34 = (
            weight_31
        ) = (
            l_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_bias_
        ) = None
        gelu_29 = torch._C._nn.gelu(out_44)
        out_44 = None
        mul__35 = gelu_29.mul_(1.7015043497085571)
        gelu_29 = None
        reshape_32 = l_self_modules_stages_modules_1_modules_2_modules_conv2b_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_57 = (
            l_self_modules_stages_modules_1_modules_2_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv2b_parameters_gain_ = None
        view_32 = mul_57.view(-1)
        mul_57 = None
        batch_norm_32 = torch.nn.functional.batch_norm(
            reshape_32,
            None,
            None,
            weight=view_32,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_32 = view_32 = None
        weight_32 = batch_norm_32.reshape_as(
            l_self_modules_stages_modules_1_modules_2_modules_conv2b_parameters_weight_
        )
        batch_norm_32 = (
            l_self_modules_stages_modules_1_modules_2_modules_conv2b_parameters_weight_
        ) = None
        out_45 = torch.conv2d(
            mul__35,
            weight_32,
            l_self_modules_stages_modules_1_modules_2_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        mul__35 = (
            weight_32
        ) = (
            l_self_modules_stages_modules_1_modules_2_modules_conv2b_parameters_bias_
        ) = None
        gelu_30 = torch._C._nn.gelu(out_45)
        out_45 = None
        mul__36 = gelu_30.mul_(1.7015043497085571)
        gelu_30 = None
        reshape_33 = l_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_58 = (
            l_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_gain_
            * 0.0625
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_gain_ = None
        view_33 = mul_58.view(-1)
        mul_58 = None
        batch_norm_33 = torch.nn.functional.batch_norm(
            reshape_33,
            None,
            None,
            weight=view_33,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_33 = view_33 = None
        weight_33 = batch_norm_33.reshape_as(
            l_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_weight_
        )
        batch_norm_33 = (
            l_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_weight_
        ) = None
        out_46 = torch.conv2d(
            mul__36,
            weight_33,
            l_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__36 = (
            weight_33
        ) = (
            l_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_bias_
        ) = None
        x_se_24 = out_46.mean((2, 3), keepdim=True)
        x_se_25 = torch.conv2d(
            x_se_24,
            l_self_modules_stages_modules_1_modules_2_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_24 = l_self_modules_stages_modules_1_modules_2_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_26 = torch.nn.functional.relu(x_se_25, inplace=True)
        x_se_25 = None
        x_se_27 = torch.conv2d(
            x_se_26,
            l_self_modules_stages_modules_1_modules_2_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_26 = l_self_modules_stages_modules_1_modules_2_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_6 = x_se_27.sigmoid()
        x_se_27 = None
        mul_59 = out_46 * sigmoid_6
        out_46 = sigmoid_6 = None
        out_47 = 2.0 * mul_59
        mul_59 = None
        mul__37 = out_47.mul_(
            l_self_modules_stages_modules_1_modules_2_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_1_modules_2_parameters_skipinit_gain_ = (
            mul__37
        ) = None
        mul_61 = out_47 * 0.2
        out_47 = None
        out_48 = mul_61 + out_41
        mul_61 = out_41 = None
        gelu_31 = torch._C._nn.gelu(out_48)
        mul__38 = gelu_31.mul_(1.7015043497085571)
        gelu_31 = None
        out_49 = mul__38 * 0.9449111825230679
        mul__38 = None
        reshape_34 = l_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_63 = (
            l_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_gain_
            * 0.04419417382415922
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_gain_ = None
        view_34 = mul_63.view(-1)
        mul_63 = None
        batch_norm_34 = torch.nn.functional.batch_norm(
            reshape_34,
            None,
            None,
            weight=view_34,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_34 = view_34 = None
        weight_34 = batch_norm_34.reshape_as(
            l_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_weight_
        )
        batch_norm_34 = (
            l_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_weight_
        ) = None
        out_50 = torch.conv2d(
            out_49,
            weight_34,
            l_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_49 = (
            weight_34
        ) = (
            l_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_bias_
        ) = None
        gelu_32 = torch._C._nn.gelu(out_50)
        out_50 = None
        mul__39 = gelu_32.mul_(1.7015043497085571)
        gelu_32 = None
        reshape_35 = l_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_64 = (
            l_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_gain_ = None
        view_35 = mul_64.view(-1)
        mul_64 = None
        batch_norm_35 = torch.nn.functional.batch_norm(
            reshape_35,
            None,
            None,
            weight=view_35,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_35 = view_35 = None
        weight_35 = batch_norm_35.reshape_as(
            l_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_weight_
        )
        batch_norm_35 = (
            l_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_weight_
        ) = None
        out_51 = torch.conv2d(
            mul__39,
            weight_35,
            l_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        mul__39 = (
            weight_35
        ) = (
            l_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_bias_
        ) = None
        gelu_33 = torch._C._nn.gelu(out_51)
        out_51 = None
        mul__40 = gelu_33.mul_(1.7015043497085571)
        gelu_33 = None
        reshape_36 = l_self_modules_stages_modules_1_modules_3_modules_conv2b_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_65 = (
            l_self_modules_stages_modules_1_modules_3_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv2b_parameters_gain_ = None
        view_36 = mul_65.view(-1)
        mul_65 = None
        batch_norm_36 = torch.nn.functional.batch_norm(
            reshape_36,
            None,
            None,
            weight=view_36,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_36 = view_36 = None
        weight_36 = batch_norm_36.reshape_as(
            l_self_modules_stages_modules_1_modules_3_modules_conv2b_parameters_weight_
        )
        batch_norm_36 = (
            l_self_modules_stages_modules_1_modules_3_modules_conv2b_parameters_weight_
        ) = None
        out_52 = torch.conv2d(
            mul__40,
            weight_36,
            l_self_modules_stages_modules_1_modules_3_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        mul__40 = (
            weight_36
        ) = (
            l_self_modules_stages_modules_1_modules_3_modules_conv2b_parameters_bias_
        ) = None
        gelu_34 = torch._C._nn.gelu(out_52)
        out_52 = None
        mul__41 = gelu_34.mul_(1.7015043497085571)
        gelu_34 = None
        reshape_37 = l_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_66 = (
            l_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_gain_
            * 0.0625
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_gain_ = None
        view_37 = mul_66.view(-1)
        mul_66 = None
        batch_norm_37 = torch.nn.functional.batch_norm(
            reshape_37,
            None,
            None,
            weight=view_37,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_37 = view_37 = None
        weight_37 = batch_norm_37.reshape_as(
            l_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_weight_
        )
        batch_norm_37 = (
            l_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_weight_
        ) = None
        out_53 = torch.conv2d(
            mul__41,
            weight_37,
            l_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__41 = (
            weight_37
        ) = (
            l_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_bias_
        ) = None
        x_se_28 = out_53.mean((2, 3), keepdim=True)
        x_se_29 = torch.conv2d(
            x_se_28,
            l_self_modules_stages_modules_1_modules_3_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_3_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_28 = l_self_modules_stages_modules_1_modules_3_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_3_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_30 = torch.nn.functional.relu(x_se_29, inplace=True)
        x_se_29 = None
        x_se_31 = torch.conv2d(
            x_se_30,
            l_self_modules_stages_modules_1_modules_3_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_3_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_30 = l_self_modules_stages_modules_1_modules_3_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_3_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_7 = x_se_31.sigmoid()
        x_se_31 = None
        mul_67 = out_53 * sigmoid_7
        out_53 = sigmoid_7 = None
        out_54 = 2.0 * mul_67
        mul_67 = None
        mul__42 = out_54.mul_(
            l_self_modules_stages_modules_1_modules_3_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_1_modules_3_parameters_skipinit_gain_ = (
            mul__42
        ) = None
        mul_69 = out_54 * 0.2
        out_54 = None
        out_55 = mul_69 + out_48
        mul_69 = out_48 = None
        gelu_35 = torch._C._nn.gelu(out_55)
        mul__43 = gelu_35.mul_(1.7015043497085571)
        gelu_35 = None
        out_56 = mul__43 * 0.9284766908852592
        mul__43 = None
        reshape_38 = l_self_modules_stages_modules_1_modules_4_modules_conv1_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_71 = (
            l_self_modules_stages_modules_1_modules_4_modules_conv1_parameters_gain_
            * 0.04419417382415922
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv1_parameters_gain_ = None
        view_38 = mul_71.view(-1)
        mul_71 = None
        batch_norm_38 = torch.nn.functional.batch_norm(
            reshape_38,
            None,
            None,
            weight=view_38,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_38 = view_38 = None
        weight_38 = batch_norm_38.reshape_as(
            l_self_modules_stages_modules_1_modules_4_modules_conv1_parameters_weight_
        )
        batch_norm_38 = (
            l_self_modules_stages_modules_1_modules_4_modules_conv1_parameters_weight_
        ) = None
        out_57 = torch.conv2d(
            out_56,
            weight_38,
            l_self_modules_stages_modules_1_modules_4_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_56 = (
            weight_38
        ) = (
            l_self_modules_stages_modules_1_modules_4_modules_conv1_parameters_bias_
        ) = None
        gelu_36 = torch._C._nn.gelu(out_57)
        out_57 = None
        mul__44 = gelu_36.mul_(1.7015043497085571)
        gelu_36 = None
        reshape_39 = l_self_modules_stages_modules_1_modules_4_modules_conv2_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_72 = (
            l_self_modules_stages_modules_1_modules_4_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv2_parameters_gain_ = None
        view_39 = mul_72.view(-1)
        mul_72 = None
        batch_norm_39 = torch.nn.functional.batch_norm(
            reshape_39,
            None,
            None,
            weight=view_39,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_39 = view_39 = None
        weight_39 = batch_norm_39.reshape_as(
            l_self_modules_stages_modules_1_modules_4_modules_conv2_parameters_weight_
        )
        batch_norm_39 = (
            l_self_modules_stages_modules_1_modules_4_modules_conv2_parameters_weight_
        ) = None
        out_58 = torch.conv2d(
            mul__44,
            weight_39,
            l_self_modules_stages_modules_1_modules_4_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        mul__44 = (
            weight_39
        ) = (
            l_self_modules_stages_modules_1_modules_4_modules_conv2_parameters_bias_
        ) = None
        gelu_37 = torch._C._nn.gelu(out_58)
        out_58 = None
        mul__45 = gelu_37.mul_(1.7015043497085571)
        gelu_37 = None
        reshape_40 = l_self_modules_stages_modules_1_modules_4_modules_conv2b_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_73 = (
            l_self_modules_stages_modules_1_modules_4_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv2b_parameters_gain_ = None
        view_40 = mul_73.view(-1)
        mul_73 = None
        batch_norm_40 = torch.nn.functional.batch_norm(
            reshape_40,
            None,
            None,
            weight=view_40,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_40 = view_40 = None
        weight_40 = batch_norm_40.reshape_as(
            l_self_modules_stages_modules_1_modules_4_modules_conv2b_parameters_weight_
        )
        batch_norm_40 = (
            l_self_modules_stages_modules_1_modules_4_modules_conv2b_parameters_weight_
        ) = None
        out_59 = torch.conv2d(
            mul__45,
            weight_40,
            l_self_modules_stages_modules_1_modules_4_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        mul__45 = (
            weight_40
        ) = (
            l_self_modules_stages_modules_1_modules_4_modules_conv2b_parameters_bias_
        ) = None
        gelu_38 = torch._C._nn.gelu(out_59)
        out_59 = None
        mul__46 = gelu_38.mul_(1.7015043497085571)
        gelu_38 = None
        reshape_41 = l_self_modules_stages_modules_1_modules_4_modules_conv3_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_74 = (
            l_self_modules_stages_modules_1_modules_4_modules_conv3_parameters_gain_
            * 0.0625
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv3_parameters_gain_ = None
        view_41 = mul_74.view(-1)
        mul_74 = None
        batch_norm_41 = torch.nn.functional.batch_norm(
            reshape_41,
            None,
            None,
            weight=view_41,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_41 = view_41 = None
        weight_41 = batch_norm_41.reshape_as(
            l_self_modules_stages_modules_1_modules_4_modules_conv3_parameters_weight_
        )
        batch_norm_41 = (
            l_self_modules_stages_modules_1_modules_4_modules_conv3_parameters_weight_
        ) = None
        out_60 = torch.conv2d(
            mul__46,
            weight_41,
            l_self_modules_stages_modules_1_modules_4_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__46 = (
            weight_41
        ) = (
            l_self_modules_stages_modules_1_modules_4_modules_conv3_parameters_bias_
        ) = None
        x_se_32 = out_60.mean((2, 3), keepdim=True)
        x_se_33 = torch.conv2d(
            x_se_32,
            l_self_modules_stages_modules_1_modules_4_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_4_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_32 = l_self_modules_stages_modules_1_modules_4_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_4_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_34 = torch.nn.functional.relu(x_se_33, inplace=True)
        x_se_33 = None
        x_se_35 = torch.conv2d(
            x_se_34,
            l_self_modules_stages_modules_1_modules_4_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_4_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_34 = l_self_modules_stages_modules_1_modules_4_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_4_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_8 = x_se_35.sigmoid()
        x_se_35 = None
        mul_75 = out_60 * sigmoid_8
        out_60 = sigmoid_8 = None
        out_61 = 2.0 * mul_75
        mul_75 = None
        mul__47 = out_61.mul_(
            l_self_modules_stages_modules_1_modules_4_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_1_modules_4_parameters_skipinit_gain_ = (
            mul__47
        ) = None
        mul_77 = out_61 * 0.2
        out_61 = None
        out_62 = mul_77 + out_55
        mul_77 = out_55 = None
        gelu_39 = torch._C._nn.gelu(out_62)
        mul__48 = gelu_39.mul_(1.7015043497085571)
        gelu_39 = None
        out_63 = mul__48 * 0.9128709291752768
        mul__48 = None
        reshape_42 = l_self_modules_stages_modules_1_modules_5_modules_conv1_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_79 = (
            l_self_modules_stages_modules_1_modules_5_modules_conv1_parameters_gain_
            * 0.04419417382415922
        )
        l_self_modules_stages_modules_1_modules_5_modules_conv1_parameters_gain_ = None
        view_42 = mul_79.view(-1)
        mul_79 = None
        batch_norm_42 = torch.nn.functional.batch_norm(
            reshape_42,
            None,
            None,
            weight=view_42,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_42 = view_42 = None
        weight_42 = batch_norm_42.reshape_as(
            l_self_modules_stages_modules_1_modules_5_modules_conv1_parameters_weight_
        )
        batch_norm_42 = (
            l_self_modules_stages_modules_1_modules_5_modules_conv1_parameters_weight_
        ) = None
        out_64 = torch.conv2d(
            out_63,
            weight_42,
            l_self_modules_stages_modules_1_modules_5_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_63 = (
            weight_42
        ) = (
            l_self_modules_stages_modules_1_modules_5_modules_conv1_parameters_bias_
        ) = None
        gelu_40 = torch._C._nn.gelu(out_64)
        out_64 = None
        mul__49 = gelu_40.mul_(1.7015043497085571)
        gelu_40 = None
        reshape_43 = l_self_modules_stages_modules_1_modules_5_modules_conv2_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_80 = (
            l_self_modules_stages_modules_1_modules_5_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_1_modules_5_modules_conv2_parameters_gain_ = None
        view_43 = mul_80.view(-1)
        mul_80 = None
        batch_norm_43 = torch.nn.functional.batch_norm(
            reshape_43,
            None,
            None,
            weight=view_43,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_43 = view_43 = None
        weight_43 = batch_norm_43.reshape_as(
            l_self_modules_stages_modules_1_modules_5_modules_conv2_parameters_weight_
        )
        batch_norm_43 = (
            l_self_modules_stages_modules_1_modules_5_modules_conv2_parameters_weight_
        ) = None
        out_65 = torch.conv2d(
            mul__49,
            weight_43,
            l_self_modules_stages_modules_1_modules_5_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        mul__49 = (
            weight_43
        ) = (
            l_self_modules_stages_modules_1_modules_5_modules_conv2_parameters_bias_
        ) = None
        gelu_41 = torch._C._nn.gelu(out_65)
        out_65 = None
        mul__50 = gelu_41.mul_(1.7015043497085571)
        gelu_41 = None
        reshape_44 = l_self_modules_stages_modules_1_modules_5_modules_conv2b_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_81 = (
            l_self_modules_stages_modules_1_modules_5_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_1_modules_5_modules_conv2b_parameters_gain_ = None
        view_44 = mul_81.view(-1)
        mul_81 = None
        batch_norm_44 = torch.nn.functional.batch_norm(
            reshape_44,
            None,
            None,
            weight=view_44,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_44 = view_44 = None
        weight_44 = batch_norm_44.reshape_as(
            l_self_modules_stages_modules_1_modules_5_modules_conv2b_parameters_weight_
        )
        batch_norm_44 = (
            l_self_modules_stages_modules_1_modules_5_modules_conv2b_parameters_weight_
        ) = None
        out_66 = torch.conv2d(
            mul__50,
            weight_44,
            l_self_modules_stages_modules_1_modules_5_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        mul__50 = (
            weight_44
        ) = (
            l_self_modules_stages_modules_1_modules_5_modules_conv2b_parameters_bias_
        ) = None
        gelu_42 = torch._C._nn.gelu(out_66)
        out_66 = None
        mul__51 = gelu_42.mul_(1.7015043497085571)
        gelu_42 = None
        reshape_45 = l_self_modules_stages_modules_1_modules_5_modules_conv3_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_82 = (
            l_self_modules_stages_modules_1_modules_5_modules_conv3_parameters_gain_
            * 0.0625
        )
        l_self_modules_stages_modules_1_modules_5_modules_conv3_parameters_gain_ = None
        view_45 = mul_82.view(-1)
        mul_82 = None
        batch_norm_45 = torch.nn.functional.batch_norm(
            reshape_45,
            None,
            None,
            weight=view_45,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_45 = view_45 = None
        weight_45 = batch_norm_45.reshape_as(
            l_self_modules_stages_modules_1_modules_5_modules_conv3_parameters_weight_
        )
        batch_norm_45 = (
            l_self_modules_stages_modules_1_modules_5_modules_conv3_parameters_weight_
        ) = None
        out_67 = torch.conv2d(
            mul__51,
            weight_45,
            l_self_modules_stages_modules_1_modules_5_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__51 = (
            weight_45
        ) = (
            l_self_modules_stages_modules_1_modules_5_modules_conv3_parameters_bias_
        ) = None
        x_se_36 = out_67.mean((2, 3), keepdim=True)
        x_se_37 = torch.conv2d(
            x_se_36,
            l_self_modules_stages_modules_1_modules_5_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_5_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_36 = l_self_modules_stages_modules_1_modules_5_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_5_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_38 = torch.nn.functional.relu(x_se_37, inplace=True)
        x_se_37 = None
        x_se_39 = torch.conv2d(
            x_se_38,
            l_self_modules_stages_modules_1_modules_5_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_5_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_38 = l_self_modules_stages_modules_1_modules_5_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_5_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_9 = x_se_39.sigmoid()
        x_se_39 = None
        mul_83 = out_67 * sigmoid_9
        out_67 = sigmoid_9 = None
        out_68 = 2.0 * mul_83
        mul_83 = None
        mul__52 = out_68.mul_(
            l_self_modules_stages_modules_1_modules_5_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_1_modules_5_parameters_skipinit_gain_ = (
            mul__52
        ) = None
        mul_85 = out_68 * 0.2
        out_68 = None
        out_69 = mul_85 + out_62
        mul_85 = out_62 = None
        gelu_43 = torch._C._nn.gelu(out_69)
        mul__53 = gelu_43.mul_(1.7015043497085571)
        gelu_43 = None
        out_70 = mul__53 * 0.8980265101338745
        mul__53 = None
        reshape_46 = l_self_modules_stages_modules_1_modules_6_modules_conv1_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_87 = (
            l_self_modules_stages_modules_1_modules_6_modules_conv1_parameters_gain_
            * 0.04419417382415922
        )
        l_self_modules_stages_modules_1_modules_6_modules_conv1_parameters_gain_ = None
        view_46 = mul_87.view(-1)
        mul_87 = None
        batch_norm_46 = torch.nn.functional.batch_norm(
            reshape_46,
            None,
            None,
            weight=view_46,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_46 = view_46 = None
        weight_46 = batch_norm_46.reshape_as(
            l_self_modules_stages_modules_1_modules_6_modules_conv1_parameters_weight_
        )
        batch_norm_46 = (
            l_self_modules_stages_modules_1_modules_6_modules_conv1_parameters_weight_
        ) = None
        out_71 = torch.conv2d(
            out_70,
            weight_46,
            l_self_modules_stages_modules_1_modules_6_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_70 = (
            weight_46
        ) = (
            l_self_modules_stages_modules_1_modules_6_modules_conv1_parameters_bias_
        ) = None
        gelu_44 = torch._C._nn.gelu(out_71)
        out_71 = None
        mul__54 = gelu_44.mul_(1.7015043497085571)
        gelu_44 = None
        reshape_47 = l_self_modules_stages_modules_1_modules_6_modules_conv2_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_88 = (
            l_self_modules_stages_modules_1_modules_6_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_1_modules_6_modules_conv2_parameters_gain_ = None
        view_47 = mul_88.view(-1)
        mul_88 = None
        batch_norm_47 = torch.nn.functional.batch_norm(
            reshape_47,
            None,
            None,
            weight=view_47,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_47 = view_47 = None
        weight_47 = batch_norm_47.reshape_as(
            l_self_modules_stages_modules_1_modules_6_modules_conv2_parameters_weight_
        )
        batch_norm_47 = (
            l_self_modules_stages_modules_1_modules_6_modules_conv2_parameters_weight_
        ) = None
        out_72 = torch.conv2d(
            mul__54,
            weight_47,
            l_self_modules_stages_modules_1_modules_6_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        mul__54 = (
            weight_47
        ) = (
            l_self_modules_stages_modules_1_modules_6_modules_conv2_parameters_bias_
        ) = None
        gelu_45 = torch._C._nn.gelu(out_72)
        out_72 = None
        mul__55 = gelu_45.mul_(1.7015043497085571)
        gelu_45 = None
        reshape_48 = l_self_modules_stages_modules_1_modules_6_modules_conv2b_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_89 = (
            l_self_modules_stages_modules_1_modules_6_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_1_modules_6_modules_conv2b_parameters_gain_ = None
        view_48 = mul_89.view(-1)
        mul_89 = None
        batch_norm_48 = torch.nn.functional.batch_norm(
            reshape_48,
            None,
            None,
            weight=view_48,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_48 = view_48 = None
        weight_48 = batch_norm_48.reshape_as(
            l_self_modules_stages_modules_1_modules_6_modules_conv2b_parameters_weight_
        )
        batch_norm_48 = (
            l_self_modules_stages_modules_1_modules_6_modules_conv2b_parameters_weight_
        ) = None
        out_73 = torch.conv2d(
            mul__55,
            weight_48,
            l_self_modules_stages_modules_1_modules_6_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        mul__55 = (
            weight_48
        ) = (
            l_self_modules_stages_modules_1_modules_6_modules_conv2b_parameters_bias_
        ) = None
        gelu_46 = torch._C._nn.gelu(out_73)
        out_73 = None
        mul__56 = gelu_46.mul_(1.7015043497085571)
        gelu_46 = None
        reshape_49 = l_self_modules_stages_modules_1_modules_6_modules_conv3_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_90 = (
            l_self_modules_stages_modules_1_modules_6_modules_conv3_parameters_gain_
            * 0.0625
        )
        l_self_modules_stages_modules_1_modules_6_modules_conv3_parameters_gain_ = None
        view_49 = mul_90.view(-1)
        mul_90 = None
        batch_norm_49 = torch.nn.functional.batch_norm(
            reshape_49,
            None,
            None,
            weight=view_49,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_49 = view_49 = None
        weight_49 = batch_norm_49.reshape_as(
            l_self_modules_stages_modules_1_modules_6_modules_conv3_parameters_weight_
        )
        batch_norm_49 = (
            l_self_modules_stages_modules_1_modules_6_modules_conv3_parameters_weight_
        ) = None
        out_74 = torch.conv2d(
            mul__56,
            weight_49,
            l_self_modules_stages_modules_1_modules_6_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__56 = (
            weight_49
        ) = (
            l_self_modules_stages_modules_1_modules_6_modules_conv3_parameters_bias_
        ) = None
        x_se_40 = out_74.mean((2, 3), keepdim=True)
        x_se_41 = torch.conv2d(
            x_se_40,
            l_self_modules_stages_modules_1_modules_6_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_6_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_40 = l_self_modules_stages_modules_1_modules_6_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_6_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_42 = torch.nn.functional.relu(x_se_41, inplace=True)
        x_se_41 = None
        x_se_43 = torch.conv2d(
            x_se_42,
            l_self_modules_stages_modules_1_modules_6_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_6_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_42 = l_self_modules_stages_modules_1_modules_6_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_6_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_10 = x_se_43.sigmoid()
        x_se_43 = None
        mul_91 = out_74 * sigmoid_10
        out_74 = sigmoid_10 = None
        out_75 = 2.0 * mul_91
        mul_91 = None
        mul__57 = out_75.mul_(
            l_self_modules_stages_modules_1_modules_6_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_1_modules_6_parameters_skipinit_gain_ = (
            mul__57
        ) = None
        mul_93 = out_75 * 0.2
        out_75 = None
        out_76 = mul_93 + out_69
        mul_93 = out_69 = None
        gelu_47 = torch._C._nn.gelu(out_76)
        mul__58 = gelu_47.mul_(1.7015043497085571)
        gelu_47 = None
        out_77 = mul__58 * 0.8838834764831842
        mul__58 = None
        reshape_50 = l_self_modules_stages_modules_1_modules_7_modules_conv1_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_95 = (
            l_self_modules_stages_modules_1_modules_7_modules_conv1_parameters_gain_
            * 0.04419417382415922
        )
        l_self_modules_stages_modules_1_modules_7_modules_conv1_parameters_gain_ = None
        view_50 = mul_95.view(-1)
        mul_95 = None
        batch_norm_50 = torch.nn.functional.batch_norm(
            reshape_50,
            None,
            None,
            weight=view_50,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_50 = view_50 = None
        weight_50 = batch_norm_50.reshape_as(
            l_self_modules_stages_modules_1_modules_7_modules_conv1_parameters_weight_
        )
        batch_norm_50 = (
            l_self_modules_stages_modules_1_modules_7_modules_conv1_parameters_weight_
        ) = None
        out_78 = torch.conv2d(
            out_77,
            weight_50,
            l_self_modules_stages_modules_1_modules_7_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_77 = (
            weight_50
        ) = (
            l_self_modules_stages_modules_1_modules_7_modules_conv1_parameters_bias_
        ) = None
        gelu_48 = torch._C._nn.gelu(out_78)
        out_78 = None
        mul__59 = gelu_48.mul_(1.7015043497085571)
        gelu_48 = None
        reshape_51 = l_self_modules_stages_modules_1_modules_7_modules_conv2_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_96 = (
            l_self_modules_stages_modules_1_modules_7_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_1_modules_7_modules_conv2_parameters_gain_ = None
        view_51 = mul_96.view(-1)
        mul_96 = None
        batch_norm_51 = torch.nn.functional.batch_norm(
            reshape_51,
            None,
            None,
            weight=view_51,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_51 = view_51 = None
        weight_51 = batch_norm_51.reshape_as(
            l_self_modules_stages_modules_1_modules_7_modules_conv2_parameters_weight_
        )
        batch_norm_51 = (
            l_self_modules_stages_modules_1_modules_7_modules_conv2_parameters_weight_
        ) = None
        out_79 = torch.conv2d(
            mul__59,
            weight_51,
            l_self_modules_stages_modules_1_modules_7_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        mul__59 = (
            weight_51
        ) = (
            l_self_modules_stages_modules_1_modules_7_modules_conv2_parameters_bias_
        ) = None
        gelu_49 = torch._C._nn.gelu(out_79)
        out_79 = None
        mul__60 = gelu_49.mul_(1.7015043497085571)
        gelu_49 = None
        reshape_52 = l_self_modules_stages_modules_1_modules_7_modules_conv2b_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_97 = (
            l_self_modules_stages_modules_1_modules_7_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_1_modules_7_modules_conv2b_parameters_gain_ = None
        view_52 = mul_97.view(-1)
        mul_97 = None
        batch_norm_52 = torch.nn.functional.batch_norm(
            reshape_52,
            None,
            None,
            weight=view_52,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_52 = view_52 = None
        weight_52 = batch_norm_52.reshape_as(
            l_self_modules_stages_modules_1_modules_7_modules_conv2b_parameters_weight_
        )
        batch_norm_52 = (
            l_self_modules_stages_modules_1_modules_7_modules_conv2b_parameters_weight_
        ) = None
        out_80 = torch.conv2d(
            mul__60,
            weight_52,
            l_self_modules_stages_modules_1_modules_7_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        mul__60 = (
            weight_52
        ) = (
            l_self_modules_stages_modules_1_modules_7_modules_conv2b_parameters_bias_
        ) = None
        gelu_50 = torch._C._nn.gelu(out_80)
        out_80 = None
        mul__61 = gelu_50.mul_(1.7015043497085571)
        gelu_50 = None
        reshape_53 = l_self_modules_stages_modules_1_modules_7_modules_conv3_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_98 = (
            l_self_modules_stages_modules_1_modules_7_modules_conv3_parameters_gain_
            * 0.0625
        )
        l_self_modules_stages_modules_1_modules_7_modules_conv3_parameters_gain_ = None
        view_53 = mul_98.view(-1)
        mul_98 = None
        batch_norm_53 = torch.nn.functional.batch_norm(
            reshape_53,
            None,
            None,
            weight=view_53,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_53 = view_53 = None
        weight_53 = batch_norm_53.reshape_as(
            l_self_modules_stages_modules_1_modules_7_modules_conv3_parameters_weight_
        )
        batch_norm_53 = (
            l_self_modules_stages_modules_1_modules_7_modules_conv3_parameters_weight_
        ) = None
        out_81 = torch.conv2d(
            mul__61,
            weight_53,
            l_self_modules_stages_modules_1_modules_7_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__61 = (
            weight_53
        ) = (
            l_self_modules_stages_modules_1_modules_7_modules_conv3_parameters_bias_
        ) = None
        x_se_44 = out_81.mean((2, 3), keepdim=True)
        x_se_45 = torch.conv2d(
            x_se_44,
            l_self_modules_stages_modules_1_modules_7_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_7_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_44 = l_self_modules_stages_modules_1_modules_7_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_7_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_46 = torch.nn.functional.relu(x_se_45, inplace=True)
        x_se_45 = None
        x_se_47 = torch.conv2d(
            x_se_46,
            l_self_modules_stages_modules_1_modules_7_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_7_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_46 = l_self_modules_stages_modules_1_modules_7_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_7_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_11 = x_se_47.sigmoid()
        x_se_47 = None
        mul_99 = out_81 * sigmoid_11
        out_81 = sigmoid_11 = None
        out_82 = 2.0 * mul_99
        mul_99 = None
        mul__62 = out_82.mul_(
            l_self_modules_stages_modules_1_modules_7_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_1_modules_7_parameters_skipinit_gain_ = (
            mul__62
        ) = None
        mul_101 = out_82 * 0.2
        out_82 = None
        out_83 = mul_101 + out_76
        mul_101 = out_76 = None
        gelu_51 = torch._C._nn.gelu(out_83)
        out_83 = None
        mul__63 = gelu_51.mul_(1.7015043497085571)
        gelu_51 = None
        out_84 = mul__63 * 0.8703882797784891
        mul__63 = None
        avg_pool2d_1 = torch._C._nn.avg_pool2d(out_84, 2, 2, 0, True, False, None)
        reshape_54 = l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_103 = (
            l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_gain_
            * 0.04419417382415922
        )
        l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_gain_ = (
            None
        )
        view_54 = mul_103.view(-1)
        mul_103 = None
        batch_norm_54 = torch.nn.functional.batch_norm(
            reshape_54,
            None,
            None,
            weight=view_54,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_54 = view_54 = None
        weight_54 = batch_norm_54.reshape_as(
            l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_
        )
        batch_norm_54 = l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        shortcut_2 = torch.conv2d(
            avg_pool2d_1,
            weight_54,
            l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        avg_pool2d_1 = (
            weight_54
        ) = l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_bias_ = (None)
        reshape_55 = l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_104 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_gain_
            * 0.04419417382415922
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_gain_ = None
        view_55 = mul_104.view(-1)
        mul_104 = None
        batch_norm_55 = torch.nn.functional.batch_norm(
            reshape_55,
            None,
            None,
            weight=view_55,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_55 = view_55 = None
        weight_55 = batch_norm_55.reshape_as(
            l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_weight_
        )
        batch_norm_55 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_weight_
        ) = None
        out_85 = torch.conv2d(
            out_84,
            weight_55,
            l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_84 = (
            weight_55
        ) = (
            l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_bias_
        ) = None
        gelu_52 = torch._C._nn.gelu(out_85)
        out_85 = None
        mul__64 = gelu_52.mul_(1.7015043497085571)
        gelu_52 = None
        x_3 = torch._C._nn.pad(mul__64, (0, 1, 0, 1), "constant", 0)
        mul__64 = None
        reshape_56 = l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_105 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_gain_ = None
        view_56 = mul_105.view(-1)
        mul_105 = None
        batch_norm_56 = torch.nn.functional.batch_norm(
            reshape_56,
            None,
            None,
            weight=view_56,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_56 = view_56 = None
        weight_56 = batch_norm_56.reshape_as(
            l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_weight_
        )
        batch_norm_56 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_weight_
        ) = None
        out_86 = torch.conv2d(
            x_3,
            weight_56,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            6,
        )
        x_3 = (
            weight_56
        ) = (
            l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_bias_
        ) = None
        gelu_53 = torch._C._nn.gelu(out_86)
        out_86 = None
        mul__65 = gelu_53.mul_(1.7015043497085571)
        gelu_53 = None
        reshape_57 = l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_106 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_gain_ = None
        view_57 = mul_106.view(-1)
        mul_106 = None
        batch_norm_57 = torch.nn.functional.batch_norm(
            reshape_57,
            None,
            None,
            weight=view_57,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_57 = view_57 = None
        weight_57 = batch_norm_57.reshape_as(
            l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_weight_
        )
        batch_norm_57 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_weight_
        ) = None
        out_87 = torch.conv2d(
            mul__65,
            weight_57,
            l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__65 = (
            weight_57
        ) = (
            l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_bias_
        ) = None
        gelu_54 = torch._C._nn.gelu(out_87)
        out_87 = None
        mul__66 = gelu_54.mul_(1.7015043497085571)
        gelu_54 = None
        reshape_58 = l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_107 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_gain_ = None
        view_58 = mul_107.view(-1)
        mul_107 = None
        batch_norm_58 = torch.nn.functional.batch_norm(
            reshape_58,
            None,
            None,
            weight=view_58,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_58 = view_58 = None
        weight_58 = batch_norm_58.reshape_as(
            l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_weight_
        )
        batch_norm_58 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_weight_
        ) = None
        out_88 = torch.conv2d(
            mul__66,
            weight_58,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__66 = (
            weight_58
        ) = (
            l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_bias_
        ) = None
        x_se_48 = out_88.mean((2, 3), keepdim=True)
        x_se_49 = torch.conv2d(
            x_se_48,
            l_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_48 = l_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_50 = torch.nn.functional.relu(x_se_49, inplace=True)
        x_se_49 = None
        x_se_51 = torch.conv2d(
            x_se_50,
            l_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_50 = l_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_12 = x_se_51.sigmoid()
        x_se_51 = None
        mul_108 = out_88 * sigmoid_12
        out_88 = sigmoid_12 = None
        out_89 = 2.0 * mul_108
        mul_108 = None
        mul__67 = out_89.mul_(
            l_self_modules_stages_modules_2_modules_0_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_0_parameters_skipinit_gain_ = (
            mul__67
        ) = None
        mul_110 = out_89 * 0.2
        out_89 = None
        out_90 = mul_110 + shortcut_2
        mul_110 = shortcut_2 = None
        gelu_55 = torch._C._nn.gelu(out_90)
        mul__68 = gelu_55.mul_(1.7015043497085571)
        gelu_55 = None
        out_91 = mul__68 * 0.9805806756909201
        mul__68 = None
        reshape_59 = l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_112 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_gain_ = None
        view_59 = mul_112.view(-1)
        mul_112 = None
        batch_norm_59 = torch.nn.functional.batch_norm(
            reshape_59,
            None,
            None,
            weight=view_59,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_59 = view_59 = None
        weight_59 = batch_norm_59.reshape_as(
            l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_weight_
        )
        batch_norm_59 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_weight_
        ) = None
        out_92 = torch.conv2d(
            out_91,
            weight_59,
            l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_91 = (
            weight_59
        ) = (
            l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_bias_
        ) = None
        gelu_56 = torch._C._nn.gelu(out_92)
        out_92 = None
        mul__69 = gelu_56.mul_(1.7015043497085571)
        gelu_56 = None
        reshape_60 = l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_113 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_gain_ = None
        view_60 = mul_113.view(-1)
        mul_113 = None
        batch_norm_60 = torch.nn.functional.batch_norm(
            reshape_60,
            None,
            None,
            weight=view_60,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_60 = view_60 = None
        weight_60 = batch_norm_60.reshape_as(
            l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_weight_
        )
        batch_norm_60 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_weight_
        ) = None
        out_93 = torch.conv2d(
            mul__69,
            weight_60,
            l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__69 = (
            weight_60
        ) = (
            l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_bias_
        ) = None
        gelu_57 = torch._C._nn.gelu(out_93)
        out_93 = None
        mul__70 = gelu_57.mul_(1.7015043497085571)
        gelu_57 = None
        reshape_61 = l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_114 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_gain_ = None
        view_61 = mul_114.view(-1)
        mul_114 = None
        batch_norm_61 = torch.nn.functional.batch_norm(
            reshape_61,
            None,
            None,
            weight=view_61,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_61 = view_61 = None
        weight_61 = batch_norm_61.reshape_as(
            l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_weight_
        )
        batch_norm_61 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_weight_
        ) = None
        out_94 = torch.conv2d(
            mul__70,
            weight_61,
            l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__70 = (
            weight_61
        ) = (
            l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_bias_
        ) = None
        gelu_58 = torch._C._nn.gelu(out_94)
        out_94 = None
        mul__71 = gelu_58.mul_(1.7015043497085571)
        gelu_58 = None
        reshape_62 = l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_115 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_gain_ = None
        view_62 = mul_115.view(-1)
        mul_115 = None
        batch_norm_62 = torch.nn.functional.batch_norm(
            reshape_62,
            None,
            None,
            weight=view_62,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_62 = view_62 = None
        weight_62 = batch_norm_62.reshape_as(
            l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_weight_
        )
        batch_norm_62 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_weight_
        ) = None
        out_95 = torch.conv2d(
            mul__71,
            weight_62,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__71 = (
            weight_62
        ) = (
            l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_bias_
        ) = None
        x_se_52 = out_95.mean((2, 3), keepdim=True)
        x_se_53 = torch.conv2d(
            x_se_52,
            l_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_52 = l_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_54 = torch.nn.functional.relu(x_se_53, inplace=True)
        x_se_53 = None
        x_se_55 = torch.conv2d(
            x_se_54,
            l_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_54 = l_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_13 = x_se_55.sigmoid()
        x_se_55 = None
        mul_116 = out_95 * sigmoid_13
        out_95 = sigmoid_13 = None
        out_96 = 2.0 * mul_116
        mul_116 = None
        mul__72 = out_96.mul_(
            l_self_modules_stages_modules_2_modules_1_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_1_parameters_skipinit_gain_ = (
            mul__72
        ) = None
        mul_118 = out_96 * 0.2
        out_96 = None
        out_97 = mul_118 + out_90
        mul_118 = out_90 = None
        gelu_59 = torch._C._nn.gelu(out_97)
        mul__73 = gelu_59.mul_(1.7015043497085571)
        gelu_59 = None
        out_98 = mul__73 * 0.9622504486493761
        mul__73 = None
        reshape_63 = l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_120 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_gain_ = None
        view_63 = mul_120.view(-1)
        mul_120 = None
        batch_norm_63 = torch.nn.functional.batch_norm(
            reshape_63,
            None,
            None,
            weight=view_63,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_63 = view_63 = None
        weight_63 = batch_norm_63.reshape_as(
            l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_weight_
        )
        batch_norm_63 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_weight_
        ) = None
        out_99 = torch.conv2d(
            out_98,
            weight_63,
            l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_98 = (
            weight_63
        ) = (
            l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_bias_
        ) = None
        gelu_60 = torch._C._nn.gelu(out_99)
        out_99 = None
        mul__74 = gelu_60.mul_(1.7015043497085571)
        gelu_60 = None
        reshape_64 = l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_121 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_gain_ = None
        view_64 = mul_121.view(-1)
        mul_121 = None
        batch_norm_64 = torch.nn.functional.batch_norm(
            reshape_64,
            None,
            None,
            weight=view_64,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_64 = view_64 = None
        weight_64 = batch_norm_64.reshape_as(
            l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_weight_
        )
        batch_norm_64 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_weight_
        ) = None
        out_100 = torch.conv2d(
            mul__74,
            weight_64,
            l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__74 = (
            weight_64
        ) = (
            l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_bias_
        ) = None
        gelu_61 = torch._C._nn.gelu(out_100)
        out_100 = None
        mul__75 = gelu_61.mul_(1.7015043497085571)
        gelu_61 = None
        reshape_65 = l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_122 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_gain_ = None
        view_65 = mul_122.view(-1)
        mul_122 = None
        batch_norm_65 = torch.nn.functional.batch_norm(
            reshape_65,
            None,
            None,
            weight=view_65,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_65 = view_65 = None
        weight_65 = batch_norm_65.reshape_as(
            l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_weight_
        )
        batch_norm_65 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_weight_
        ) = None
        out_101 = torch.conv2d(
            mul__75,
            weight_65,
            l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__75 = (
            weight_65
        ) = (
            l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_bias_
        ) = None
        gelu_62 = torch._C._nn.gelu(out_101)
        out_101 = None
        mul__76 = gelu_62.mul_(1.7015043497085571)
        gelu_62 = None
        reshape_66 = l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_123 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_gain_ = None
        view_66 = mul_123.view(-1)
        mul_123 = None
        batch_norm_66 = torch.nn.functional.batch_norm(
            reshape_66,
            None,
            None,
            weight=view_66,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_66 = view_66 = None
        weight_66 = batch_norm_66.reshape_as(
            l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_weight_
        )
        batch_norm_66 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_weight_
        ) = None
        out_102 = torch.conv2d(
            mul__76,
            weight_66,
            l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__76 = (
            weight_66
        ) = (
            l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_bias_
        ) = None
        x_se_56 = out_102.mean((2, 3), keepdim=True)
        x_se_57 = torch.conv2d(
            x_se_56,
            l_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_56 = l_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_58 = torch.nn.functional.relu(x_se_57, inplace=True)
        x_se_57 = None
        x_se_59 = torch.conv2d(
            x_se_58,
            l_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_58 = l_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_14 = x_se_59.sigmoid()
        x_se_59 = None
        mul_124 = out_102 * sigmoid_14
        out_102 = sigmoid_14 = None
        out_103 = 2.0 * mul_124
        mul_124 = None
        mul__77 = out_103.mul_(
            l_self_modules_stages_modules_2_modules_2_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_2_parameters_skipinit_gain_ = (
            mul__77
        ) = None
        mul_126 = out_103 * 0.2
        out_103 = None
        out_104 = mul_126 + out_97
        mul_126 = out_97 = None
        gelu_63 = torch._C._nn.gelu(out_104)
        mul__78 = gelu_63.mul_(1.7015043497085571)
        gelu_63 = None
        out_105 = mul__78 * 0.9449111825230679
        mul__78 = None
        reshape_67 = l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_128 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_gain_ = None
        view_67 = mul_128.view(-1)
        mul_128 = None
        batch_norm_67 = torch.nn.functional.batch_norm(
            reshape_67,
            None,
            None,
            weight=view_67,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_67 = view_67 = None
        weight_67 = batch_norm_67.reshape_as(
            l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_weight_
        )
        batch_norm_67 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_weight_
        ) = None
        out_106 = torch.conv2d(
            out_105,
            weight_67,
            l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_105 = (
            weight_67
        ) = (
            l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_bias_
        ) = None
        gelu_64 = torch._C._nn.gelu(out_106)
        out_106 = None
        mul__79 = gelu_64.mul_(1.7015043497085571)
        gelu_64 = None
        reshape_68 = l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_129 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_gain_ = None
        view_68 = mul_129.view(-1)
        mul_129 = None
        batch_norm_68 = torch.nn.functional.batch_norm(
            reshape_68,
            None,
            None,
            weight=view_68,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_68 = view_68 = None
        weight_68 = batch_norm_68.reshape_as(
            l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_weight_
        )
        batch_norm_68 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_weight_
        ) = None
        out_107 = torch.conv2d(
            mul__79,
            weight_68,
            l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__79 = (
            weight_68
        ) = (
            l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_bias_
        ) = None
        gelu_65 = torch._C._nn.gelu(out_107)
        out_107 = None
        mul__80 = gelu_65.mul_(1.7015043497085571)
        gelu_65 = None
        reshape_69 = l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_130 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_gain_ = None
        view_69 = mul_130.view(-1)
        mul_130 = None
        batch_norm_69 = torch.nn.functional.batch_norm(
            reshape_69,
            None,
            None,
            weight=view_69,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_69 = view_69 = None
        weight_69 = batch_norm_69.reshape_as(
            l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_weight_
        )
        batch_norm_69 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_weight_
        ) = None
        out_108 = torch.conv2d(
            mul__80,
            weight_69,
            l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__80 = (
            weight_69
        ) = (
            l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_bias_
        ) = None
        gelu_66 = torch._C._nn.gelu(out_108)
        out_108 = None
        mul__81 = gelu_66.mul_(1.7015043497085571)
        gelu_66 = None
        reshape_70 = l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_131 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_gain_ = None
        view_70 = mul_131.view(-1)
        mul_131 = None
        batch_norm_70 = torch.nn.functional.batch_norm(
            reshape_70,
            None,
            None,
            weight=view_70,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_70 = view_70 = None
        weight_70 = batch_norm_70.reshape_as(
            l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_weight_
        )
        batch_norm_70 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_weight_
        ) = None
        out_109 = torch.conv2d(
            mul__81,
            weight_70,
            l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__81 = (
            weight_70
        ) = (
            l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_bias_
        ) = None
        x_se_60 = out_109.mean((2, 3), keepdim=True)
        x_se_61 = torch.conv2d(
            x_se_60,
            l_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_60 = l_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_62 = torch.nn.functional.relu(x_se_61, inplace=True)
        x_se_61 = None
        x_se_63 = torch.conv2d(
            x_se_62,
            l_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_62 = l_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_15 = x_se_63.sigmoid()
        x_se_63 = None
        mul_132 = out_109 * sigmoid_15
        out_109 = sigmoid_15 = None
        out_110 = 2.0 * mul_132
        mul_132 = None
        mul__82 = out_110.mul_(
            l_self_modules_stages_modules_2_modules_3_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_3_parameters_skipinit_gain_ = (
            mul__82
        ) = None
        mul_134 = out_110 * 0.2
        out_110 = None
        out_111 = mul_134 + out_104
        mul_134 = out_104 = None
        gelu_67 = torch._C._nn.gelu(out_111)
        mul__83 = gelu_67.mul_(1.7015043497085571)
        gelu_67 = None
        out_112 = mul__83 * 0.9284766908852592
        mul__83 = None
        reshape_71 = l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_136 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_gain_ = None
        view_71 = mul_136.view(-1)
        mul_136 = None
        batch_norm_71 = torch.nn.functional.batch_norm(
            reshape_71,
            None,
            None,
            weight=view_71,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_71 = view_71 = None
        weight_71 = batch_norm_71.reshape_as(
            l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_weight_
        )
        batch_norm_71 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_weight_
        ) = None
        out_113 = torch.conv2d(
            out_112,
            weight_71,
            l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_112 = (
            weight_71
        ) = (
            l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_bias_
        ) = None
        gelu_68 = torch._C._nn.gelu(out_113)
        out_113 = None
        mul__84 = gelu_68.mul_(1.7015043497085571)
        gelu_68 = None
        reshape_72 = l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_137 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_gain_ = None
        view_72 = mul_137.view(-1)
        mul_137 = None
        batch_norm_72 = torch.nn.functional.batch_norm(
            reshape_72,
            None,
            None,
            weight=view_72,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_72 = view_72 = None
        weight_72 = batch_norm_72.reshape_as(
            l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_weight_
        )
        batch_norm_72 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_weight_
        ) = None
        out_114 = torch.conv2d(
            mul__84,
            weight_72,
            l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__84 = (
            weight_72
        ) = (
            l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_bias_
        ) = None
        gelu_69 = torch._C._nn.gelu(out_114)
        out_114 = None
        mul__85 = gelu_69.mul_(1.7015043497085571)
        gelu_69 = None
        reshape_73 = l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_138 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_gain_ = None
        view_73 = mul_138.view(-1)
        mul_138 = None
        batch_norm_73 = torch.nn.functional.batch_norm(
            reshape_73,
            None,
            None,
            weight=view_73,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_73 = view_73 = None
        weight_73 = batch_norm_73.reshape_as(
            l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_weight_
        )
        batch_norm_73 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_weight_
        ) = None
        out_115 = torch.conv2d(
            mul__85,
            weight_73,
            l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__85 = (
            weight_73
        ) = (
            l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_bias_
        ) = None
        gelu_70 = torch._C._nn.gelu(out_115)
        out_115 = None
        mul__86 = gelu_70.mul_(1.7015043497085571)
        gelu_70 = None
        reshape_74 = l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_139 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_gain_ = None
        view_74 = mul_139.view(-1)
        mul_139 = None
        batch_norm_74 = torch.nn.functional.batch_norm(
            reshape_74,
            None,
            None,
            weight=view_74,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_74 = view_74 = None
        weight_74 = batch_norm_74.reshape_as(
            l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_weight_
        )
        batch_norm_74 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_weight_
        ) = None
        out_116 = torch.conv2d(
            mul__86,
            weight_74,
            l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__86 = (
            weight_74
        ) = (
            l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_bias_
        ) = None
        x_se_64 = out_116.mean((2, 3), keepdim=True)
        x_se_65 = torch.conv2d(
            x_se_64,
            l_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_64 = l_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_66 = torch.nn.functional.relu(x_se_65, inplace=True)
        x_se_65 = None
        x_se_67 = torch.conv2d(
            x_se_66,
            l_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_66 = l_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_16 = x_se_67.sigmoid()
        x_se_67 = None
        mul_140 = out_116 * sigmoid_16
        out_116 = sigmoid_16 = None
        out_117 = 2.0 * mul_140
        mul_140 = None
        mul__87 = out_117.mul_(
            l_self_modules_stages_modules_2_modules_4_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_4_parameters_skipinit_gain_ = (
            mul__87
        ) = None
        mul_142 = out_117 * 0.2
        out_117 = None
        out_118 = mul_142 + out_111
        mul_142 = out_111 = None
        gelu_71 = torch._C._nn.gelu(out_118)
        mul__88 = gelu_71.mul_(1.7015043497085571)
        gelu_71 = None
        out_119 = mul__88 * 0.9128709291752768
        mul__88 = None
        reshape_75 = l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_144 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_gain_ = None
        view_75 = mul_144.view(-1)
        mul_144 = None
        batch_norm_75 = torch.nn.functional.batch_norm(
            reshape_75,
            None,
            None,
            weight=view_75,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_75 = view_75 = None
        weight_75 = batch_norm_75.reshape_as(
            l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_weight_
        )
        batch_norm_75 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_weight_
        ) = None
        out_120 = torch.conv2d(
            out_119,
            weight_75,
            l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_119 = (
            weight_75
        ) = (
            l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_bias_
        ) = None
        gelu_72 = torch._C._nn.gelu(out_120)
        out_120 = None
        mul__89 = gelu_72.mul_(1.7015043497085571)
        gelu_72 = None
        reshape_76 = l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_145 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_gain_ = None
        view_76 = mul_145.view(-1)
        mul_145 = None
        batch_norm_76 = torch.nn.functional.batch_norm(
            reshape_76,
            None,
            None,
            weight=view_76,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_76 = view_76 = None
        weight_76 = batch_norm_76.reshape_as(
            l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_weight_
        )
        batch_norm_76 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_weight_
        ) = None
        out_121 = torch.conv2d(
            mul__89,
            weight_76,
            l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__89 = (
            weight_76
        ) = (
            l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_bias_
        ) = None
        gelu_73 = torch._C._nn.gelu(out_121)
        out_121 = None
        mul__90 = gelu_73.mul_(1.7015043497085571)
        gelu_73 = None
        reshape_77 = l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_146 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_gain_ = None
        view_77 = mul_146.view(-1)
        mul_146 = None
        batch_norm_77 = torch.nn.functional.batch_norm(
            reshape_77,
            None,
            None,
            weight=view_77,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_77 = view_77 = None
        weight_77 = batch_norm_77.reshape_as(
            l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_weight_
        )
        batch_norm_77 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_weight_
        ) = None
        out_122 = torch.conv2d(
            mul__90,
            weight_77,
            l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__90 = (
            weight_77
        ) = (
            l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_bias_
        ) = None
        gelu_74 = torch._C._nn.gelu(out_122)
        out_122 = None
        mul__91 = gelu_74.mul_(1.7015043497085571)
        gelu_74 = None
        reshape_78 = l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_147 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_gain_ = None
        view_78 = mul_147.view(-1)
        mul_147 = None
        batch_norm_78 = torch.nn.functional.batch_norm(
            reshape_78,
            None,
            None,
            weight=view_78,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_78 = view_78 = None
        weight_78 = batch_norm_78.reshape_as(
            l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_weight_
        )
        batch_norm_78 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_weight_
        ) = None
        out_123 = torch.conv2d(
            mul__91,
            weight_78,
            l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__91 = (
            weight_78
        ) = (
            l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_bias_
        ) = None
        x_se_68 = out_123.mean((2, 3), keepdim=True)
        x_se_69 = torch.conv2d(
            x_se_68,
            l_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_68 = l_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_70 = torch.nn.functional.relu(x_se_69, inplace=True)
        x_se_69 = None
        x_se_71 = torch.conv2d(
            x_se_70,
            l_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_70 = l_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_17 = x_se_71.sigmoid()
        x_se_71 = None
        mul_148 = out_123 * sigmoid_17
        out_123 = sigmoid_17 = None
        out_124 = 2.0 * mul_148
        mul_148 = None
        mul__92 = out_124.mul_(
            l_self_modules_stages_modules_2_modules_5_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_5_parameters_skipinit_gain_ = (
            mul__92
        ) = None
        mul_150 = out_124 * 0.2
        out_124 = None
        out_125 = mul_150 + out_118
        mul_150 = out_118 = None
        gelu_75 = torch._C._nn.gelu(out_125)
        mul__93 = gelu_75.mul_(1.7015043497085571)
        gelu_75 = None
        out_126 = mul__93 * 0.8980265101338745
        mul__93 = None
        reshape_79 = l_self_modules_stages_modules_2_modules_6_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_152 = (
            l_self_modules_stages_modules_2_modules_6_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv1_parameters_gain_ = None
        view_79 = mul_152.view(-1)
        mul_152 = None
        batch_norm_79 = torch.nn.functional.batch_norm(
            reshape_79,
            None,
            None,
            weight=view_79,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_79 = view_79 = None
        weight_79 = batch_norm_79.reshape_as(
            l_self_modules_stages_modules_2_modules_6_modules_conv1_parameters_weight_
        )
        batch_norm_79 = (
            l_self_modules_stages_modules_2_modules_6_modules_conv1_parameters_weight_
        ) = None
        out_127 = torch.conv2d(
            out_126,
            weight_79,
            l_self_modules_stages_modules_2_modules_6_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_126 = (
            weight_79
        ) = (
            l_self_modules_stages_modules_2_modules_6_modules_conv1_parameters_bias_
        ) = None
        gelu_76 = torch._C._nn.gelu(out_127)
        out_127 = None
        mul__94 = gelu_76.mul_(1.7015043497085571)
        gelu_76 = None
        reshape_80 = l_self_modules_stages_modules_2_modules_6_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_153 = (
            l_self_modules_stages_modules_2_modules_6_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv2_parameters_gain_ = None
        view_80 = mul_153.view(-1)
        mul_153 = None
        batch_norm_80 = torch.nn.functional.batch_norm(
            reshape_80,
            None,
            None,
            weight=view_80,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_80 = view_80 = None
        weight_80 = batch_norm_80.reshape_as(
            l_self_modules_stages_modules_2_modules_6_modules_conv2_parameters_weight_
        )
        batch_norm_80 = (
            l_self_modules_stages_modules_2_modules_6_modules_conv2_parameters_weight_
        ) = None
        out_128 = torch.conv2d(
            mul__94,
            weight_80,
            l_self_modules_stages_modules_2_modules_6_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__94 = (
            weight_80
        ) = (
            l_self_modules_stages_modules_2_modules_6_modules_conv2_parameters_bias_
        ) = None
        gelu_77 = torch._C._nn.gelu(out_128)
        out_128 = None
        mul__95 = gelu_77.mul_(1.7015043497085571)
        gelu_77 = None
        reshape_81 = l_self_modules_stages_modules_2_modules_6_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_154 = (
            l_self_modules_stages_modules_2_modules_6_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv2b_parameters_gain_ = None
        view_81 = mul_154.view(-1)
        mul_154 = None
        batch_norm_81 = torch.nn.functional.batch_norm(
            reshape_81,
            None,
            None,
            weight=view_81,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_81 = view_81 = None
        weight_81 = batch_norm_81.reshape_as(
            l_self_modules_stages_modules_2_modules_6_modules_conv2b_parameters_weight_
        )
        batch_norm_81 = (
            l_self_modules_stages_modules_2_modules_6_modules_conv2b_parameters_weight_
        ) = None
        out_129 = torch.conv2d(
            mul__95,
            weight_81,
            l_self_modules_stages_modules_2_modules_6_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__95 = (
            weight_81
        ) = (
            l_self_modules_stages_modules_2_modules_6_modules_conv2b_parameters_bias_
        ) = None
        gelu_78 = torch._C._nn.gelu(out_129)
        out_129 = None
        mul__96 = gelu_78.mul_(1.7015043497085571)
        gelu_78 = None
        reshape_82 = l_self_modules_stages_modules_2_modules_6_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_155 = (
            l_self_modules_stages_modules_2_modules_6_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv3_parameters_gain_ = None
        view_82 = mul_155.view(-1)
        mul_155 = None
        batch_norm_82 = torch.nn.functional.batch_norm(
            reshape_82,
            None,
            None,
            weight=view_82,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_82 = view_82 = None
        weight_82 = batch_norm_82.reshape_as(
            l_self_modules_stages_modules_2_modules_6_modules_conv3_parameters_weight_
        )
        batch_norm_82 = (
            l_self_modules_stages_modules_2_modules_6_modules_conv3_parameters_weight_
        ) = None
        out_130 = torch.conv2d(
            mul__96,
            weight_82,
            l_self_modules_stages_modules_2_modules_6_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__96 = (
            weight_82
        ) = (
            l_self_modules_stages_modules_2_modules_6_modules_conv3_parameters_bias_
        ) = None
        x_se_72 = out_130.mean((2, 3), keepdim=True)
        x_se_73 = torch.conv2d(
            x_se_72,
            l_self_modules_stages_modules_2_modules_6_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_72 = l_self_modules_stages_modules_2_modules_6_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_6_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_74 = torch.nn.functional.relu(x_se_73, inplace=True)
        x_se_73 = None
        x_se_75 = torch.conv2d(
            x_se_74,
            l_self_modules_stages_modules_2_modules_6_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_74 = l_self_modules_stages_modules_2_modules_6_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_6_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_18 = x_se_75.sigmoid()
        x_se_75 = None
        mul_156 = out_130 * sigmoid_18
        out_130 = sigmoid_18 = None
        out_131 = 2.0 * mul_156
        mul_156 = None
        mul__97 = out_131.mul_(
            l_self_modules_stages_modules_2_modules_6_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_6_parameters_skipinit_gain_ = (
            mul__97
        ) = None
        mul_158 = out_131 * 0.2
        out_131 = None
        out_132 = mul_158 + out_125
        mul_158 = out_125 = None
        gelu_79 = torch._C._nn.gelu(out_132)
        mul__98 = gelu_79.mul_(1.7015043497085571)
        gelu_79 = None
        out_133 = mul__98 * 0.8838834764831842
        mul__98 = None
        reshape_83 = l_self_modules_stages_modules_2_modules_7_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_160 = (
            l_self_modules_stages_modules_2_modules_7_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_2_modules_7_modules_conv1_parameters_gain_ = None
        view_83 = mul_160.view(-1)
        mul_160 = None
        batch_norm_83 = torch.nn.functional.batch_norm(
            reshape_83,
            None,
            None,
            weight=view_83,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_83 = view_83 = None
        weight_83 = batch_norm_83.reshape_as(
            l_self_modules_stages_modules_2_modules_7_modules_conv1_parameters_weight_
        )
        batch_norm_83 = (
            l_self_modules_stages_modules_2_modules_7_modules_conv1_parameters_weight_
        ) = None
        out_134 = torch.conv2d(
            out_133,
            weight_83,
            l_self_modules_stages_modules_2_modules_7_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_133 = (
            weight_83
        ) = (
            l_self_modules_stages_modules_2_modules_7_modules_conv1_parameters_bias_
        ) = None
        gelu_80 = torch._C._nn.gelu(out_134)
        out_134 = None
        mul__99 = gelu_80.mul_(1.7015043497085571)
        gelu_80 = None
        reshape_84 = l_self_modules_stages_modules_2_modules_7_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_161 = (
            l_self_modules_stages_modules_2_modules_7_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_7_modules_conv2_parameters_gain_ = None
        view_84 = mul_161.view(-1)
        mul_161 = None
        batch_norm_84 = torch.nn.functional.batch_norm(
            reshape_84,
            None,
            None,
            weight=view_84,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_84 = view_84 = None
        weight_84 = batch_norm_84.reshape_as(
            l_self_modules_stages_modules_2_modules_7_modules_conv2_parameters_weight_
        )
        batch_norm_84 = (
            l_self_modules_stages_modules_2_modules_7_modules_conv2_parameters_weight_
        ) = None
        out_135 = torch.conv2d(
            mul__99,
            weight_84,
            l_self_modules_stages_modules_2_modules_7_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__99 = (
            weight_84
        ) = (
            l_self_modules_stages_modules_2_modules_7_modules_conv2_parameters_bias_
        ) = None
        gelu_81 = torch._C._nn.gelu(out_135)
        out_135 = None
        mul__100 = gelu_81.mul_(1.7015043497085571)
        gelu_81 = None
        reshape_85 = l_self_modules_stages_modules_2_modules_7_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_162 = (
            l_self_modules_stages_modules_2_modules_7_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_7_modules_conv2b_parameters_gain_ = None
        view_85 = mul_162.view(-1)
        mul_162 = None
        batch_norm_85 = torch.nn.functional.batch_norm(
            reshape_85,
            None,
            None,
            weight=view_85,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_85 = view_85 = None
        weight_85 = batch_norm_85.reshape_as(
            l_self_modules_stages_modules_2_modules_7_modules_conv2b_parameters_weight_
        )
        batch_norm_85 = (
            l_self_modules_stages_modules_2_modules_7_modules_conv2b_parameters_weight_
        ) = None
        out_136 = torch.conv2d(
            mul__100,
            weight_85,
            l_self_modules_stages_modules_2_modules_7_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__100 = (
            weight_85
        ) = (
            l_self_modules_stages_modules_2_modules_7_modules_conv2b_parameters_bias_
        ) = None
        gelu_82 = torch._C._nn.gelu(out_136)
        out_136 = None
        mul__101 = gelu_82.mul_(1.7015043497085571)
        gelu_82 = None
        reshape_86 = l_self_modules_stages_modules_2_modules_7_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_163 = (
            l_self_modules_stages_modules_2_modules_7_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_7_modules_conv3_parameters_gain_ = None
        view_86 = mul_163.view(-1)
        mul_163 = None
        batch_norm_86 = torch.nn.functional.batch_norm(
            reshape_86,
            None,
            None,
            weight=view_86,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_86 = view_86 = None
        weight_86 = batch_norm_86.reshape_as(
            l_self_modules_stages_modules_2_modules_7_modules_conv3_parameters_weight_
        )
        batch_norm_86 = (
            l_self_modules_stages_modules_2_modules_7_modules_conv3_parameters_weight_
        ) = None
        out_137 = torch.conv2d(
            mul__101,
            weight_86,
            l_self_modules_stages_modules_2_modules_7_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__101 = (
            weight_86
        ) = (
            l_self_modules_stages_modules_2_modules_7_modules_conv3_parameters_bias_
        ) = None
        x_se_76 = out_137.mean((2, 3), keepdim=True)
        x_se_77 = torch.conv2d(
            x_se_76,
            l_self_modules_stages_modules_2_modules_7_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_7_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_76 = l_self_modules_stages_modules_2_modules_7_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_7_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_78 = torch.nn.functional.relu(x_se_77, inplace=True)
        x_se_77 = None
        x_se_79 = torch.conv2d(
            x_se_78,
            l_self_modules_stages_modules_2_modules_7_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_7_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_78 = l_self_modules_stages_modules_2_modules_7_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_7_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_19 = x_se_79.sigmoid()
        x_se_79 = None
        mul_164 = out_137 * sigmoid_19
        out_137 = sigmoid_19 = None
        out_138 = 2.0 * mul_164
        mul_164 = None
        mul__102 = out_138.mul_(
            l_self_modules_stages_modules_2_modules_7_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_7_parameters_skipinit_gain_ = (
            mul__102
        ) = None
        mul_166 = out_138 * 0.2
        out_138 = None
        out_139 = mul_166 + out_132
        mul_166 = out_132 = None
        gelu_83 = torch._C._nn.gelu(out_139)
        mul__103 = gelu_83.mul_(1.7015043497085571)
        gelu_83 = None
        out_140 = mul__103 * 0.8703882797784891
        mul__103 = None
        reshape_87 = l_self_modules_stages_modules_2_modules_8_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_168 = (
            l_self_modules_stages_modules_2_modules_8_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv1_parameters_gain_ = None
        view_87 = mul_168.view(-1)
        mul_168 = None
        batch_norm_87 = torch.nn.functional.batch_norm(
            reshape_87,
            None,
            None,
            weight=view_87,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_87 = view_87 = None
        weight_87 = batch_norm_87.reshape_as(
            l_self_modules_stages_modules_2_modules_8_modules_conv1_parameters_weight_
        )
        batch_norm_87 = (
            l_self_modules_stages_modules_2_modules_8_modules_conv1_parameters_weight_
        ) = None
        out_141 = torch.conv2d(
            out_140,
            weight_87,
            l_self_modules_stages_modules_2_modules_8_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_140 = (
            weight_87
        ) = (
            l_self_modules_stages_modules_2_modules_8_modules_conv1_parameters_bias_
        ) = None
        gelu_84 = torch._C._nn.gelu(out_141)
        out_141 = None
        mul__104 = gelu_84.mul_(1.7015043497085571)
        gelu_84 = None
        reshape_88 = l_self_modules_stages_modules_2_modules_8_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_169 = (
            l_self_modules_stages_modules_2_modules_8_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv2_parameters_gain_ = None
        view_88 = mul_169.view(-1)
        mul_169 = None
        batch_norm_88 = torch.nn.functional.batch_norm(
            reshape_88,
            None,
            None,
            weight=view_88,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_88 = view_88 = None
        weight_88 = batch_norm_88.reshape_as(
            l_self_modules_stages_modules_2_modules_8_modules_conv2_parameters_weight_
        )
        batch_norm_88 = (
            l_self_modules_stages_modules_2_modules_8_modules_conv2_parameters_weight_
        ) = None
        out_142 = torch.conv2d(
            mul__104,
            weight_88,
            l_self_modules_stages_modules_2_modules_8_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__104 = (
            weight_88
        ) = (
            l_self_modules_stages_modules_2_modules_8_modules_conv2_parameters_bias_
        ) = None
        gelu_85 = torch._C._nn.gelu(out_142)
        out_142 = None
        mul__105 = gelu_85.mul_(1.7015043497085571)
        gelu_85 = None
        reshape_89 = l_self_modules_stages_modules_2_modules_8_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_170 = (
            l_self_modules_stages_modules_2_modules_8_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv2b_parameters_gain_ = None
        view_89 = mul_170.view(-1)
        mul_170 = None
        batch_norm_89 = torch.nn.functional.batch_norm(
            reshape_89,
            None,
            None,
            weight=view_89,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_89 = view_89 = None
        weight_89 = batch_norm_89.reshape_as(
            l_self_modules_stages_modules_2_modules_8_modules_conv2b_parameters_weight_
        )
        batch_norm_89 = (
            l_self_modules_stages_modules_2_modules_8_modules_conv2b_parameters_weight_
        ) = None
        out_143 = torch.conv2d(
            mul__105,
            weight_89,
            l_self_modules_stages_modules_2_modules_8_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__105 = (
            weight_89
        ) = (
            l_self_modules_stages_modules_2_modules_8_modules_conv2b_parameters_bias_
        ) = None
        gelu_86 = torch._C._nn.gelu(out_143)
        out_143 = None
        mul__106 = gelu_86.mul_(1.7015043497085571)
        gelu_86 = None
        reshape_90 = l_self_modules_stages_modules_2_modules_8_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_171 = (
            l_self_modules_stages_modules_2_modules_8_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv3_parameters_gain_ = None
        view_90 = mul_171.view(-1)
        mul_171 = None
        batch_norm_90 = torch.nn.functional.batch_norm(
            reshape_90,
            None,
            None,
            weight=view_90,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_90 = view_90 = None
        weight_90 = batch_norm_90.reshape_as(
            l_self_modules_stages_modules_2_modules_8_modules_conv3_parameters_weight_
        )
        batch_norm_90 = (
            l_self_modules_stages_modules_2_modules_8_modules_conv3_parameters_weight_
        ) = None
        out_144 = torch.conv2d(
            mul__106,
            weight_90,
            l_self_modules_stages_modules_2_modules_8_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__106 = (
            weight_90
        ) = (
            l_self_modules_stages_modules_2_modules_8_modules_conv3_parameters_bias_
        ) = None
        x_se_80 = out_144.mean((2, 3), keepdim=True)
        x_se_81 = torch.conv2d(
            x_se_80,
            l_self_modules_stages_modules_2_modules_8_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_8_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_80 = l_self_modules_stages_modules_2_modules_8_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_8_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_82 = torch.nn.functional.relu(x_se_81, inplace=True)
        x_se_81 = None
        x_se_83 = torch.conv2d(
            x_se_82,
            l_self_modules_stages_modules_2_modules_8_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_8_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_82 = l_self_modules_stages_modules_2_modules_8_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_8_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_20 = x_se_83.sigmoid()
        x_se_83 = None
        mul_172 = out_144 * sigmoid_20
        out_144 = sigmoid_20 = None
        out_145 = 2.0 * mul_172
        mul_172 = None
        mul__107 = out_145.mul_(
            l_self_modules_stages_modules_2_modules_8_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_8_parameters_skipinit_gain_ = (
            mul__107
        ) = None
        mul_174 = out_145 * 0.2
        out_145 = None
        out_146 = mul_174 + out_139
        mul_174 = out_139 = None
        gelu_87 = torch._C._nn.gelu(out_146)
        mul__108 = gelu_87.mul_(1.7015043497085571)
        gelu_87 = None
        out_147 = mul__108 * 0.8574929257125441
        mul__108 = None
        reshape_91 = l_self_modules_stages_modules_2_modules_9_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_176 = (
            l_self_modules_stages_modules_2_modules_9_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_2_modules_9_modules_conv1_parameters_gain_ = None
        view_91 = mul_176.view(-1)
        mul_176 = None
        batch_norm_91 = torch.nn.functional.batch_norm(
            reshape_91,
            None,
            None,
            weight=view_91,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_91 = view_91 = None
        weight_91 = batch_norm_91.reshape_as(
            l_self_modules_stages_modules_2_modules_9_modules_conv1_parameters_weight_
        )
        batch_norm_91 = (
            l_self_modules_stages_modules_2_modules_9_modules_conv1_parameters_weight_
        ) = None
        out_148 = torch.conv2d(
            out_147,
            weight_91,
            l_self_modules_stages_modules_2_modules_9_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_147 = (
            weight_91
        ) = (
            l_self_modules_stages_modules_2_modules_9_modules_conv1_parameters_bias_
        ) = None
        gelu_88 = torch._C._nn.gelu(out_148)
        out_148 = None
        mul__109 = gelu_88.mul_(1.7015043497085571)
        gelu_88 = None
        reshape_92 = l_self_modules_stages_modules_2_modules_9_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_177 = (
            l_self_modules_stages_modules_2_modules_9_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_9_modules_conv2_parameters_gain_ = None
        view_92 = mul_177.view(-1)
        mul_177 = None
        batch_norm_92 = torch.nn.functional.batch_norm(
            reshape_92,
            None,
            None,
            weight=view_92,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_92 = view_92 = None
        weight_92 = batch_norm_92.reshape_as(
            l_self_modules_stages_modules_2_modules_9_modules_conv2_parameters_weight_
        )
        batch_norm_92 = (
            l_self_modules_stages_modules_2_modules_9_modules_conv2_parameters_weight_
        ) = None
        out_149 = torch.conv2d(
            mul__109,
            weight_92,
            l_self_modules_stages_modules_2_modules_9_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__109 = (
            weight_92
        ) = (
            l_self_modules_stages_modules_2_modules_9_modules_conv2_parameters_bias_
        ) = None
        gelu_89 = torch._C._nn.gelu(out_149)
        out_149 = None
        mul__110 = gelu_89.mul_(1.7015043497085571)
        gelu_89 = None
        reshape_93 = l_self_modules_stages_modules_2_modules_9_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_178 = (
            l_self_modules_stages_modules_2_modules_9_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_9_modules_conv2b_parameters_gain_ = None
        view_93 = mul_178.view(-1)
        mul_178 = None
        batch_norm_93 = torch.nn.functional.batch_norm(
            reshape_93,
            None,
            None,
            weight=view_93,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_93 = view_93 = None
        weight_93 = batch_norm_93.reshape_as(
            l_self_modules_stages_modules_2_modules_9_modules_conv2b_parameters_weight_
        )
        batch_norm_93 = (
            l_self_modules_stages_modules_2_modules_9_modules_conv2b_parameters_weight_
        ) = None
        out_150 = torch.conv2d(
            mul__110,
            weight_93,
            l_self_modules_stages_modules_2_modules_9_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__110 = (
            weight_93
        ) = (
            l_self_modules_stages_modules_2_modules_9_modules_conv2b_parameters_bias_
        ) = None
        gelu_90 = torch._C._nn.gelu(out_150)
        out_150 = None
        mul__111 = gelu_90.mul_(1.7015043497085571)
        gelu_90 = None
        reshape_94 = l_self_modules_stages_modules_2_modules_9_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_179 = (
            l_self_modules_stages_modules_2_modules_9_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_9_modules_conv3_parameters_gain_ = None
        view_94 = mul_179.view(-1)
        mul_179 = None
        batch_norm_94 = torch.nn.functional.batch_norm(
            reshape_94,
            None,
            None,
            weight=view_94,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_94 = view_94 = None
        weight_94 = batch_norm_94.reshape_as(
            l_self_modules_stages_modules_2_modules_9_modules_conv3_parameters_weight_
        )
        batch_norm_94 = (
            l_self_modules_stages_modules_2_modules_9_modules_conv3_parameters_weight_
        ) = None
        out_151 = torch.conv2d(
            mul__111,
            weight_94,
            l_self_modules_stages_modules_2_modules_9_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__111 = (
            weight_94
        ) = (
            l_self_modules_stages_modules_2_modules_9_modules_conv3_parameters_bias_
        ) = None
        x_se_84 = out_151.mean((2, 3), keepdim=True)
        x_se_85 = torch.conv2d(
            x_se_84,
            l_self_modules_stages_modules_2_modules_9_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_9_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_84 = l_self_modules_stages_modules_2_modules_9_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_9_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_86 = torch.nn.functional.relu(x_se_85, inplace=True)
        x_se_85 = None
        x_se_87 = torch.conv2d(
            x_se_86,
            l_self_modules_stages_modules_2_modules_9_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_9_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_86 = l_self_modules_stages_modules_2_modules_9_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_9_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_21 = x_se_87.sigmoid()
        x_se_87 = None
        mul_180 = out_151 * sigmoid_21
        out_151 = sigmoid_21 = None
        out_152 = 2.0 * mul_180
        mul_180 = None
        mul__112 = out_152.mul_(
            l_self_modules_stages_modules_2_modules_9_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_9_parameters_skipinit_gain_ = (
            mul__112
        ) = None
        mul_182 = out_152 * 0.2
        out_152 = None
        out_153 = mul_182 + out_146
        mul_182 = out_146 = None
        gelu_91 = torch._C._nn.gelu(out_153)
        mul__113 = gelu_91.mul_(1.7015043497085571)
        gelu_91 = None
        out_154 = mul__113 * 0.8451542547285165
        mul__113 = None
        reshape_95 = l_self_modules_stages_modules_2_modules_10_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_184 = (
            l_self_modules_stages_modules_2_modules_10_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv1_parameters_gain_ = None
        view_95 = mul_184.view(-1)
        mul_184 = None
        batch_norm_95 = torch.nn.functional.batch_norm(
            reshape_95,
            None,
            None,
            weight=view_95,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_95 = view_95 = None
        weight_95 = batch_norm_95.reshape_as(
            l_self_modules_stages_modules_2_modules_10_modules_conv1_parameters_weight_
        )
        batch_norm_95 = (
            l_self_modules_stages_modules_2_modules_10_modules_conv1_parameters_weight_
        ) = None
        out_155 = torch.conv2d(
            out_154,
            weight_95,
            l_self_modules_stages_modules_2_modules_10_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_154 = (
            weight_95
        ) = (
            l_self_modules_stages_modules_2_modules_10_modules_conv1_parameters_bias_
        ) = None
        gelu_92 = torch._C._nn.gelu(out_155)
        out_155 = None
        mul__114 = gelu_92.mul_(1.7015043497085571)
        gelu_92 = None
        reshape_96 = l_self_modules_stages_modules_2_modules_10_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_185 = (
            l_self_modules_stages_modules_2_modules_10_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv2_parameters_gain_ = None
        view_96 = mul_185.view(-1)
        mul_185 = None
        batch_norm_96 = torch.nn.functional.batch_norm(
            reshape_96,
            None,
            None,
            weight=view_96,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_96 = view_96 = None
        weight_96 = batch_norm_96.reshape_as(
            l_self_modules_stages_modules_2_modules_10_modules_conv2_parameters_weight_
        )
        batch_norm_96 = (
            l_self_modules_stages_modules_2_modules_10_modules_conv2_parameters_weight_
        ) = None
        out_156 = torch.conv2d(
            mul__114,
            weight_96,
            l_self_modules_stages_modules_2_modules_10_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__114 = (
            weight_96
        ) = (
            l_self_modules_stages_modules_2_modules_10_modules_conv2_parameters_bias_
        ) = None
        gelu_93 = torch._C._nn.gelu(out_156)
        out_156 = None
        mul__115 = gelu_93.mul_(1.7015043497085571)
        gelu_93 = None
        reshape_97 = l_self_modules_stages_modules_2_modules_10_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_186 = (
            l_self_modules_stages_modules_2_modules_10_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv2b_parameters_gain_ = (
            None
        )
        view_97 = mul_186.view(-1)
        mul_186 = None
        batch_norm_97 = torch.nn.functional.batch_norm(
            reshape_97,
            None,
            None,
            weight=view_97,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_97 = view_97 = None
        weight_97 = batch_norm_97.reshape_as(
            l_self_modules_stages_modules_2_modules_10_modules_conv2b_parameters_weight_
        )
        batch_norm_97 = (
            l_self_modules_stages_modules_2_modules_10_modules_conv2b_parameters_weight_
        ) = None
        out_157 = torch.conv2d(
            mul__115,
            weight_97,
            l_self_modules_stages_modules_2_modules_10_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__115 = (
            weight_97
        ) = (
            l_self_modules_stages_modules_2_modules_10_modules_conv2b_parameters_bias_
        ) = None
        gelu_94 = torch._C._nn.gelu(out_157)
        out_157 = None
        mul__116 = gelu_94.mul_(1.7015043497085571)
        gelu_94 = None
        reshape_98 = l_self_modules_stages_modules_2_modules_10_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_187 = (
            l_self_modules_stages_modules_2_modules_10_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv3_parameters_gain_ = None
        view_98 = mul_187.view(-1)
        mul_187 = None
        batch_norm_98 = torch.nn.functional.batch_norm(
            reshape_98,
            None,
            None,
            weight=view_98,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_98 = view_98 = None
        weight_98 = batch_norm_98.reshape_as(
            l_self_modules_stages_modules_2_modules_10_modules_conv3_parameters_weight_
        )
        batch_norm_98 = (
            l_self_modules_stages_modules_2_modules_10_modules_conv3_parameters_weight_
        ) = None
        out_158 = torch.conv2d(
            mul__116,
            weight_98,
            l_self_modules_stages_modules_2_modules_10_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__116 = (
            weight_98
        ) = (
            l_self_modules_stages_modules_2_modules_10_modules_conv3_parameters_bias_
        ) = None
        x_se_88 = out_158.mean((2, 3), keepdim=True)
        x_se_89 = torch.conv2d(
            x_se_88,
            l_self_modules_stages_modules_2_modules_10_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_10_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_88 = l_self_modules_stages_modules_2_modules_10_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_10_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_90 = torch.nn.functional.relu(x_se_89, inplace=True)
        x_se_89 = None
        x_se_91 = torch.conv2d(
            x_se_90,
            l_self_modules_stages_modules_2_modules_10_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_10_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_90 = l_self_modules_stages_modules_2_modules_10_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_10_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_22 = x_se_91.sigmoid()
        x_se_91 = None
        mul_188 = out_158 * sigmoid_22
        out_158 = sigmoid_22 = None
        out_159 = 2.0 * mul_188
        mul_188 = None
        mul__117 = out_159.mul_(
            l_self_modules_stages_modules_2_modules_10_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_10_parameters_skipinit_gain_ = (
            mul__117
        ) = None
        mul_190 = out_159 * 0.2
        out_159 = None
        out_160 = mul_190 + out_153
        mul_190 = out_153 = None
        gelu_95 = torch._C._nn.gelu(out_160)
        mul__118 = gelu_95.mul_(1.7015043497085571)
        gelu_95 = None
        out_161 = mul__118 * 0.8333333333333333
        mul__118 = None
        reshape_99 = l_self_modules_stages_modules_2_modules_11_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_192 = (
            l_self_modules_stages_modules_2_modules_11_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_2_modules_11_modules_conv1_parameters_gain_ = None
        view_99 = mul_192.view(-1)
        mul_192 = None
        batch_norm_99 = torch.nn.functional.batch_norm(
            reshape_99,
            None,
            None,
            weight=view_99,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_99 = view_99 = None
        weight_99 = batch_norm_99.reshape_as(
            l_self_modules_stages_modules_2_modules_11_modules_conv1_parameters_weight_
        )
        batch_norm_99 = (
            l_self_modules_stages_modules_2_modules_11_modules_conv1_parameters_weight_
        ) = None
        out_162 = torch.conv2d(
            out_161,
            weight_99,
            l_self_modules_stages_modules_2_modules_11_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_161 = (
            weight_99
        ) = (
            l_self_modules_stages_modules_2_modules_11_modules_conv1_parameters_bias_
        ) = None
        gelu_96 = torch._C._nn.gelu(out_162)
        out_162 = None
        mul__119 = gelu_96.mul_(1.7015043497085571)
        gelu_96 = None
        reshape_100 = l_self_modules_stages_modules_2_modules_11_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_193 = (
            l_self_modules_stages_modules_2_modules_11_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_11_modules_conv2_parameters_gain_ = None
        view_100 = mul_193.view(-1)
        mul_193 = None
        batch_norm_100 = torch.nn.functional.batch_norm(
            reshape_100,
            None,
            None,
            weight=view_100,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_100 = view_100 = None
        weight_100 = batch_norm_100.reshape_as(
            l_self_modules_stages_modules_2_modules_11_modules_conv2_parameters_weight_
        )
        batch_norm_100 = (
            l_self_modules_stages_modules_2_modules_11_modules_conv2_parameters_weight_
        ) = None
        out_163 = torch.conv2d(
            mul__119,
            weight_100,
            l_self_modules_stages_modules_2_modules_11_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__119 = (
            weight_100
        ) = (
            l_self_modules_stages_modules_2_modules_11_modules_conv2_parameters_bias_
        ) = None
        gelu_97 = torch._C._nn.gelu(out_163)
        out_163 = None
        mul__120 = gelu_97.mul_(1.7015043497085571)
        gelu_97 = None
        reshape_101 = l_self_modules_stages_modules_2_modules_11_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_194 = (
            l_self_modules_stages_modules_2_modules_11_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_11_modules_conv2b_parameters_gain_ = (
            None
        )
        view_101 = mul_194.view(-1)
        mul_194 = None
        batch_norm_101 = torch.nn.functional.batch_norm(
            reshape_101,
            None,
            None,
            weight=view_101,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_101 = view_101 = None
        weight_101 = batch_norm_101.reshape_as(
            l_self_modules_stages_modules_2_modules_11_modules_conv2b_parameters_weight_
        )
        batch_norm_101 = (
            l_self_modules_stages_modules_2_modules_11_modules_conv2b_parameters_weight_
        ) = None
        out_164 = torch.conv2d(
            mul__120,
            weight_101,
            l_self_modules_stages_modules_2_modules_11_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__120 = (
            weight_101
        ) = (
            l_self_modules_stages_modules_2_modules_11_modules_conv2b_parameters_bias_
        ) = None
        gelu_98 = torch._C._nn.gelu(out_164)
        out_164 = None
        mul__121 = gelu_98.mul_(1.7015043497085571)
        gelu_98 = None
        reshape_102 = l_self_modules_stages_modules_2_modules_11_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_195 = (
            l_self_modules_stages_modules_2_modules_11_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_11_modules_conv3_parameters_gain_ = None
        view_102 = mul_195.view(-1)
        mul_195 = None
        batch_norm_102 = torch.nn.functional.batch_norm(
            reshape_102,
            None,
            None,
            weight=view_102,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_102 = view_102 = None
        weight_102 = batch_norm_102.reshape_as(
            l_self_modules_stages_modules_2_modules_11_modules_conv3_parameters_weight_
        )
        batch_norm_102 = (
            l_self_modules_stages_modules_2_modules_11_modules_conv3_parameters_weight_
        ) = None
        out_165 = torch.conv2d(
            mul__121,
            weight_102,
            l_self_modules_stages_modules_2_modules_11_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__121 = (
            weight_102
        ) = (
            l_self_modules_stages_modules_2_modules_11_modules_conv3_parameters_bias_
        ) = None
        x_se_92 = out_165.mean((2, 3), keepdim=True)
        x_se_93 = torch.conv2d(
            x_se_92,
            l_self_modules_stages_modules_2_modules_11_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_11_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_92 = l_self_modules_stages_modules_2_modules_11_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_11_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_94 = torch.nn.functional.relu(x_se_93, inplace=True)
        x_se_93 = None
        x_se_95 = torch.conv2d(
            x_se_94,
            l_self_modules_stages_modules_2_modules_11_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_11_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_94 = l_self_modules_stages_modules_2_modules_11_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_11_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_23 = x_se_95.sigmoid()
        x_se_95 = None
        mul_196 = out_165 * sigmoid_23
        out_165 = sigmoid_23 = None
        out_166 = 2.0 * mul_196
        mul_196 = None
        mul__122 = out_166.mul_(
            l_self_modules_stages_modules_2_modules_11_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_11_parameters_skipinit_gain_ = (
            mul__122
        ) = None
        mul_198 = out_166 * 0.2
        out_166 = None
        out_167 = mul_198 + out_160
        mul_198 = out_160 = None
        gelu_99 = torch._C._nn.gelu(out_167)
        mul__123 = gelu_99.mul_(1.7015043497085571)
        gelu_99 = None
        out_168 = mul__123 * 0.8219949365267863
        mul__123 = None
        reshape_103 = l_self_modules_stages_modules_2_modules_12_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_200 = (
            l_self_modules_stages_modules_2_modules_12_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv1_parameters_gain_ = None
        view_103 = mul_200.view(-1)
        mul_200 = None
        batch_norm_103 = torch.nn.functional.batch_norm(
            reshape_103,
            None,
            None,
            weight=view_103,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_103 = view_103 = None
        weight_103 = batch_norm_103.reshape_as(
            l_self_modules_stages_modules_2_modules_12_modules_conv1_parameters_weight_
        )
        batch_norm_103 = (
            l_self_modules_stages_modules_2_modules_12_modules_conv1_parameters_weight_
        ) = None
        out_169 = torch.conv2d(
            out_168,
            weight_103,
            l_self_modules_stages_modules_2_modules_12_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_168 = (
            weight_103
        ) = (
            l_self_modules_stages_modules_2_modules_12_modules_conv1_parameters_bias_
        ) = None
        gelu_100 = torch._C._nn.gelu(out_169)
        out_169 = None
        mul__124 = gelu_100.mul_(1.7015043497085571)
        gelu_100 = None
        reshape_104 = l_self_modules_stages_modules_2_modules_12_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_201 = (
            l_self_modules_stages_modules_2_modules_12_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv2_parameters_gain_ = None
        view_104 = mul_201.view(-1)
        mul_201 = None
        batch_norm_104 = torch.nn.functional.batch_norm(
            reshape_104,
            None,
            None,
            weight=view_104,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_104 = view_104 = None
        weight_104 = batch_norm_104.reshape_as(
            l_self_modules_stages_modules_2_modules_12_modules_conv2_parameters_weight_
        )
        batch_norm_104 = (
            l_self_modules_stages_modules_2_modules_12_modules_conv2_parameters_weight_
        ) = None
        out_170 = torch.conv2d(
            mul__124,
            weight_104,
            l_self_modules_stages_modules_2_modules_12_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__124 = (
            weight_104
        ) = (
            l_self_modules_stages_modules_2_modules_12_modules_conv2_parameters_bias_
        ) = None
        gelu_101 = torch._C._nn.gelu(out_170)
        out_170 = None
        mul__125 = gelu_101.mul_(1.7015043497085571)
        gelu_101 = None
        reshape_105 = l_self_modules_stages_modules_2_modules_12_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_202 = (
            l_self_modules_stages_modules_2_modules_12_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv2b_parameters_gain_ = (
            None
        )
        view_105 = mul_202.view(-1)
        mul_202 = None
        batch_norm_105 = torch.nn.functional.batch_norm(
            reshape_105,
            None,
            None,
            weight=view_105,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_105 = view_105 = None
        weight_105 = batch_norm_105.reshape_as(
            l_self_modules_stages_modules_2_modules_12_modules_conv2b_parameters_weight_
        )
        batch_norm_105 = (
            l_self_modules_stages_modules_2_modules_12_modules_conv2b_parameters_weight_
        ) = None
        out_171 = torch.conv2d(
            mul__125,
            weight_105,
            l_self_modules_stages_modules_2_modules_12_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__125 = (
            weight_105
        ) = (
            l_self_modules_stages_modules_2_modules_12_modules_conv2b_parameters_bias_
        ) = None
        gelu_102 = torch._C._nn.gelu(out_171)
        out_171 = None
        mul__126 = gelu_102.mul_(1.7015043497085571)
        gelu_102 = None
        reshape_106 = l_self_modules_stages_modules_2_modules_12_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_203 = (
            l_self_modules_stages_modules_2_modules_12_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv3_parameters_gain_ = None
        view_106 = mul_203.view(-1)
        mul_203 = None
        batch_norm_106 = torch.nn.functional.batch_norm(
            reshape_106,
            None,
            None,
            weight=view_106,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_106 = view_106 = None
        weight_106 = batch_norm_106.reshape_as(
            l_self_modules_stages_modules_2_modules_12_modules_conv3_parameters_weight_
        )
        batch_norm_106 = (
            l_self_modules_stages_modules_2_modules_12_modules_conv3_parameters_weight_
        ) = None
        out_172 = torch.conv2d(
            mul__126,
            weight_106,
            l_self_modules_stages_modules_2_modules_12_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__126 = (
            weight_106
        ) = (
            l_self_modules_stages_modules_2_modules_12_modules_conv3_parameters_bias_
        ) = None
        x_se_96 = out_172.mean((2, 3), keepdim=True)
        x_se_97 = torch.conv2d(
            x_se_96,
            l_self_modules_stages_modules_2_modules_12_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_12_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_96 = l_self_modules_stages_modules_2_modules_12_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_12_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_98 = torch.nn.functional.relu(x_se_97, inplace=True)
        x_se_97 = None
        x_se_99 = torch.conv2d(
            x_se_98,
            l_self_modules_stages_modules_2_modules_12_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_12_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_98 = l_self_modules_stages_modules_2_modules_12_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_12_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_24 = x_se_99.sigmoid()
        x_se_99 = None
        mul_204 = out_172 * sigmoid_24
        out_172 = sigmoid_24 = None
        out_173 = 2.0 * mul_204
        mul_204 = None
        mul__127 = out_173.mul_(
            l_self_modules_stages_modules_2_modules_12_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_12_parameters_skipinit_gain_ = (
            mul__127
        ) = None
        mul_206 = out_173 * 0.2
        out_173 = None
        out_174 = mul_206 + out_167
        mul_206 = out_167 = None
        gelu_103 = torch._C._nn.gelu(out_174)
        mul__128 = gelu_103.mul_(1.7015043497085571)
        gelu_103 = None
        out_175 = mul__128 * 0.8111071056538125
        mul__128 = None
        reshape_107 = l_self_modules_stages_modules_2_modules_13_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_208 = (
            l_self_modules_stages_modules_2_modules_13_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_2_modules_13_modules_conv1_parameters_gain_ = None
        view_107 = mul_208.view(-1)
        mul_208 = None
        batch_norm_107 = torch.nn.functional.batch_norm(
            reshape_107,
            None,
            None,
            weight=view_107,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_107 = view_107 = None
        weight_107 = batch_norm_107.reshape_as(
            l_self_modules_stages_modules_2_modules_13_modules_conv1_parameters_weight_
        )
        batch_norm_107 = (
            l_self_modules_stages_modules_2_modules_13_modules_conv1_parameters_weight_
        ) = None
        out_176 = torch.conv2d(
            out_175,
            weight_107,
            l_self_modules_stages_modules_2_modules_13_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_175 = (
            weight_107
        ) = (
            l_self_modules_stages_modules_2_modules_13_modules_conv1_parameters_bias_
        ) = None
        gelu_104 = torch._C._nn.gelu(out_176)
        out_176 = None
        mul__129 = gelu_104.mul_(1.7015043497085571)
        gelu_104 = None
        reshape_108 = l_self_modules_stages_modules_2_modules_13_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_209 = (
            l_self_modules_stages_modules_2_modules_13_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_13_modules_conv2_parameters_gain_ = None
        view_108 = mul_209.view(-1)
        mul_209 = None
        batch_norm_108 = torch.nn.functional.batch_norm(
            reshape_108,
            None,
            None,
            weight=view_108,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_108 = view_108 = None
        weight_108 = batch_norm_108.reshape_as(
            l_self_modules_stages_modules_2_modules_13_modules_conv2_parameters_weight_
        )
        batch_norm_108 = (
            l_self_modules_stages_modules_2_modules_13_modules_conv2_parameters_weight_
        ) = None
        out_177 = torch.conv2d(
            mul__129,
            weight_108,
            l_self_modules_stages_modules_2_modules_13_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__129 = (
            weight_108
        ) = (
            l_self_modules_stages_modules_2_modules_13_modules_conv2_parameters_bias_
        ) = None
        gelu_105 = torch._C._nn.gelu(out_177)
        out_177 = None
        mul__130 = gelu_105.mul_(1.7015043497085571)
        gelu_105 = None
        reshape_109 = l_self_modules_stages_modules_2_modules_13_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_210 = (
            l_self_modules_stages_modules_2_modules_13_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_13_modules_conv2b_parameters_gain_ = (
            None
        )
        view_109 = mul_210.view(-1)
        mul_210 = None
        batch_norm_109 = torch.nn.functional.batch_norm(
            reshape_109,
            None,
            None,
            weight=view_109,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_109 = view_109 = None
        weight_109 = batch_norm_109.reshape_as(
            l_self_modules_stages_modules_2_modules_13_modules_conv2b_parameters_weight_
        )
        batch_norm_109 = (
            l_self_modules_stages_modules_2_modules_13_modules_conv2b_parameters_weight_
        ) = None
        out_178 = torch.conv2d(
            mul__130,
            weight_109,
            l_self_modules_stages_modules_2_modules_13_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__130 = (
            weight_109
        ) = (
            l_self_modules_stages_modules_2_modules_13_modules_conv2b_parameters_bias_
        ) = None
        gelu_106 = torch._C._nn.gelu(out_178)
        out_178 = None
        mul__131 = gelu_106.mul_(1.7015043497085571)
        gelu_106 = None
        reshape_110 = l_self_modules_stages_modules_2_modules_13_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_211 = (
            l_self_modules_stages_modules_2_modules_13_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_13_modules_conv3_parameters_gain_ = None
        view_110 = mul_211.view(-1)
        mul_211 = None
        batch_norm_110 = torch.nn.functional.batch_norm(
            reshape_110,
            None,
            None,
            weight=view_110,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_110 = view_110 = None
        weight_110 = batch_norm_110.reshape_as(
            l_self_modules_stages_modules_2_modules_13_modules_conv3_parameters_weight_
        )
        batch_norm_110 = (
            l_self_modules_stages_modules_2_modules_13_modules_conv3_parameters_weight_
        ) = None
        out_179 = torch.conv2d(
            mul__131,
            weight_110,
            l_self_modules_stages_modules_2_modules_13_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__131 = (
            weight_110
        ) = (
            l_self_modules_stages_modules_2_modules_13_modules_conv3_parameters_bias_
        ) = None
        x_se_100 = out_179.mean((2, 3), keepdim=True)
        x_se_101 = torch.conv2d(
            x_se_100,
            l_self_modules_stages_modules_2_modules_13_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_13_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_100 = l_self_modules_stages_modules_2_modules_13_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_13_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_102 = torch.nn.functional.relu(x_se_101, inplace=True)
        x_se_101 = None
        x_se_103 = torch.conv2d(
            x_se_102,
            l_self_modules_stages_modules_2_modules_13_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_13_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_102 = l_self_modules_stages_modules_2_modules_13_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_13_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_25 = x_se_103.sigmoid()
        x_se_103 = None
        mul_212 = out_179 * sigmoid_25
        out_179 = sigmoid_25 = None
        out_180 = 2.0 * mul_212
        mul_212 = None
        mul__132 = out_180.mul_(
            l_self_modules_stages_modules_2_modules_13_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_13_parameters_skipinit_gain_ = (
            mul__132
        ) = None
        mul_214 = out_180 * 0.2
        out_180 = None
        out_181 = mul_214 + out_174
        mul_214 = out_174 = None
        gelu_107 = torch._C._nn.gelu(out_181)
        mul__133 = gelu_107.mul_(1.7015043497085571)
        gelu_107 = None
        out_182 = mul__133 * 0.8006407690254355
        mul__133 = None
        reshape_111 = l_self_modules_stages_modules_2_modules_14_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_216 = (
            l_self_modules_stages_modules_2_modules_14_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv1_parameters_gain_ = None
        view_111 = mul_216.view(-1)
        mul_216 = None
        batch_norm_111 = torch.nn.functional.batch_norm(
            reshape_111,
            None,
            None,
            weight=view_111,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_111 = view_111 = None
        weight_111 = batch_norm_111.reshape_as(
            l_self_modules_stages_modules_2_modules_14_modules_conv1_parameters_weight_
        )
        batch_norm_111 = (
            l_self_modules_stages_modules_2_modules_14_modules_conv1_parameters_weight_
        ) = None
        out_183 = torch.conv2d(
            out_182,
            weight_111,
            l_self_modules_stages_modules_2_modules_14_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_182 = (
            weight_111
        ) = (
            l_self_modules_stages_modules_2_modules_14_modules_conv1_parameters_bias_
        ) = None
        gelu_108 = torch._C._nn.gelu(out_183)
        out_183 = None
        mul__134 = gelu_108.mul_(1.7015043497085571)
        gelu_108 = None
        reshape_112 = l_self_modules_stages_modules_2_modules_14_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_217 = (
            l_self_modules_stages_modules_2_modules_14_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv2_parameters_gain_ = None
        view_112 = mul_217.view(-1)
        mul_217 = None
        batch_norm_112 = torch.nn.functional.batch_norm(
            reshape_112,
            None,
            None,
            weight=view_112,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_112 = view_112 = None
        weight_112 = batch_norm_112.reshape_as(
            l_self_modules_stages_modules_2_modules_14_modules_conv2_parameters_weight_
        )
        batch_norm_112 = (
            l_self_modules_stages_modules_2_modules_14_modules_conv2_parameters_weight_
        ) = None
        out_184 = torch.conv2d(
            mul__134,
            weight_112,
            l_self_modules_stages_modules_2_modules_14_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__134 = (
            weight_112
        ) = (
            l_self_modules_stages_modules_2_modules_14_modules_conv2_parameters_bias_
        ) = None
        gelu_109 = torch._C._nn.gelu(out_184)
        out_184 = None
        mul__135 = gelu_109.mul_(1.7015043497085571)
        gelu_109 = None
        reshape_113 = l_self_modules_stages_modules_2_modules_14_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_218 = (
            l_self_modules_stages_modules_2_modules_14_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv2b_parameters_gain_ = (
            None
        )
        view_113 = mul_218.view(-1)
        mul_218 = None
        batch_norm_113 = torch.nn.functional.batch_norm(
            reshape_113,
            None,
            None,
            weight=view_113,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_113 = view_113 = None
        weight_113 = batch_norm_113.reshape_as(
            l_self_modules_stages_modules_2_modules_14_modules_conv2b_parameters_weight_
        )
        batch_norm_113 = (
            l_self_modules_stages_modules_2_modules_14_modules_conv2b_parameters_weight_
        ) = None
        out_185 = torch.conv2d(
            mul__135,
            weight_113,
            l_self_modules_stages_modules_2_modules_14_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__135 = (
            weight_113
        ) = (
            l_self_modules_stages_modules_2_modules_14_modules_conv2b_parameters_bias_
        ) = None
        gelu_110 = torch._C._nn.gelu(out_185)
        out_185 = None
        mul__136 = gelu_110.mul_(1.7015043497085571)
        gelu_110 = None
        reshape_114 = l_self_modules_stages_modules_2_modules_14_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_219 = (
            l_self_modules_stages_modules_2_modules_14_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv3_parameters_gain_ = None
        view_114 = mul_219.view(-1)
        mul_219 = None
        batch_norm_114 = torch.nn.functional.batch_norm(
            reshape_114,
            None,
            None,
            weight=view_114,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_114 = view_114 = None
        weight_114 = batch_norm_114.reshape_as(
            l_self_modules_stages_modules_2_modules_14_modules_conv3_parameters_weight_
        )
        batch_norm_114 = (
            l_self_modules_stages_modules_2_modules_14_modules_conv3_parameters_weight_
        ) = None
        out_186 = torch.conv2d(
            mul__136,
            weight_114,
            l_self_modules_stages_modules_2_modules_14_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__136 = (
            weight_114
        ) = (
            l_self_modules_stages_modules_2_modules_14_modules_conv3_parameters_bias_
        ) = None
        x_se_104 = out_186.mean((2, 3), keepdim=True)
        x_se_105 = torch.conv2d(
            x_se_104,
            l_self_modules_stages_modules_2_modules_14_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_14_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_104 = l_self_modules_stages_modules_2_modules_14_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_14_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_106 = torch.nn.functional.relu(x_se_105, inplace=True)
        x_se_105 = None
        x_se_107 = torch.conv2d(
            x_se_106,
            l_self_modules_stages_modules_2_modules_14_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_14_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_106 = l_self_modules_stages_modules_2_modules_14_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_14_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_26 = x_se_107.sigmoid()
        x_se_107 = None
        mul_220 = out_186 * sigmoid_26
        out_186 = sigmoid_26 = None
        out_187 = 2.0 * mul_220
        mul_220 = None
        mul__137 = out_187.mul_(
            l_self_modules_stages_modules_2_modules_14_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_14_parameters_skipinit_gain_ = (
            mul__137
        ) = None
        mul_222 = out_187 * 0.2
        out_187 = None
        out_188 = mul_222 + out_181
        mul_222 = out_181 = None
        gelu_111 = torch._C._nn.gelu(out_188)
        mul__138 = gelu_111.mul_(1.7015043497085571)
        gelu_111 = None
        out_189 = mul__138 * 0.7905694150420947
        mul__138 = None
        reshape_115 = l_self_modules_stages_modules_2_modules_15_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_224 = (
            l_self_modules_stages_modules_2_modules_15_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_2_modules_15_modules_conv1_parameters_gain_ = None
        view_115 = mul_224.view(-1)
        mul_224 = None
        batch_norm_115 = torch.nn.functional.batch_norm(
            reshape_115,
            None,
            None,
            weight=view_115,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_115 = view_115 = None
        weight_115 = batch_norm_115.reshape_as(
            l_self_modules_stages_modules_2_modules_15_modules_conv1_parameters_weight_
        )
        batch_norm_115 = (
            l_self_modules_stages_modules_2_modules_15_modules_conv1_parameters_weight_
        ) = None
        out_190 = torch.conv2d(
            out_189,
            weight_115,
            l_self_modules_stages_modules_2_modules_15_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_189 = (
            weight_115
        ) = (
            l_self_modules_stages_modules_2_modules_15_modules_conv1_parameters_bias_
        ) = None
        gelu_112 = torch._C._nn.gelu(out_190)
        out_190 = None
        mul__139 = gelu_112.mul_(1.7015043497085571)
        gelu_112 = None
        reshape_116 = l_self_modules_stages_modules_2_modules_15_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_225 = (
            l_self_modules_stages_modules_2_modules_15_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_15_modules_conv2_parameters_gain_ = None
        view_116 = mul_225.view(-1)
        mul_225 = None
        batch_norm_116 = torch.nn.functional.batch_norm(
            reshape_116,
            None,
            None,
            weight=view_116,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_116 = view_116 = None
        weight_116 = batch_norm_116.reshape_as(
            l_self_modules_stages_modules_2_modules_15_modules_conv2_parameters_weight_
        )
        batch_norm_116 = (
            l_self_modules_stages_modules_2_modules_15_modules_conv2_parameters_weight_
        ) = None
        out_191 = torch.conv2d(
            mul__139,
            weight_116,
            l_self_modules_stages_modules_2_modules_15_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__139 = (
            weight_116
        ) = (
            l_self_modules_stages_modules_2_modules_15_modules_conv2_parameters_bias_
        ) = None
        gelu_113 = torch._C._nn.gelu(out_191)
        out_191 = None
        mul__140 = gelu_113.mul_(1.7015043497085571)
        gelu_113 = None
        reshape_117 = l_self_modules_stages_modules_2_modules_15_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_226 = (
            l_self_modules_stages_modules_2_modules_15_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_15_modules_conv2b_parameters_gain_ = (
            None
        )
        view_117 = mul_226.view(-1)
        mul_226 = None
        batch_norm_117 = torch.nn.functional.batch_norm(
            reshape_117,
            None,
            None,
            weight=view_117,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_117 = view_117 = None
        weight_117 = batch_norm_117.reshape_as(
            l_self_modules_stages_modules_2_modules_15_modules_conv2b_parameters_weight_
        )
        batch_norm_117 = (
            l_self_modules_stages_modules_2_modules_15_modules_conv2b_parameters_weight_
        ) = None
        out_192 = torch.conv2d(
            mul__140,
            weight_117,
            l_self_modules_stages_modules_2_modules_15_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__140 = (
            weight_117
        ) = (
            l_self_modules_stages_modules_2_modules_15_modules_conv2b_parameters_bias_
        ) = None
        gelu_114 = torch._C._nn.gelu(out_192)
        out_192 = None
        mul__141 = gelu_114.mul_(1.7015043497085571)
        gelu_114 = None
        reshape_118 = l_self_modules_stages_modules_2_modules_15_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_227 = (
            l_self_modules_stages_modules_2_modules_15_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_15_modules_conv3_parameters_gain_ = None
        view_118 = mul_227.view(-1)
        mul_227 = None
        batch_norm_118 = torch.nn.functional.batch_norm(
            reshape_118,
            None,
            None,
            weight=view_118,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_118 = view_118 = None
        weight_118 = batch_norm_118.reshape_as(
            l_self_modules_stages_modules_2_modules_15_modules_conv3_parameters_weight_
        )
        batch_norm_118 = (
            l_self_modules_stages_modules_2_modules_15_modules_conv3_parameters_weight_
        ) = None
        out_193 = torch.conv2d(
            mul__141,
            weight_118,
            l_self_modules_stages_modules_2_modules_15_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__141 = (
            weight_118
        ) = (
            l_self_modules_stages_modules_2_modules_15_modules_conv3_parameters_bias_
        ) = None
        x_se_108 = out_193.mean((2, 3), keepdim=True)
        x_se_109 = torch.conv2d(
            x_se_108,
            l_self_modules_stages_modules_2_modules_15_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_15_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_108 = l_self_modules_stages_modules_2_modules_15_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_15_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_110 = torch.nn.functional.relu(x_se_109, inplace=True)
        x_se_109 = None
        x_se_111 = torch.conv2d(
            x_se_110,
            l_self_modules_stages_modules_2_modules_15_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_15_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_110 = l_self_modules_stages_modules_2_modules_15_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_15_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_27 = x_se_111.sigmoid()
        x_se_111 = None
        mul_228 = out_193 * sigmoid_27
        out_193 = sigmoid_27 = None
        out_194 = 2.0 * mul_228
        mul_228 = None
        mul__142 = out_194.mul_(
            l_self_modules_stages_modules_2_modules_15_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_15_parameters_skipinit_gain_ = (
            mul__142
        ) = None
        mul_230 = out_194 * 0.2
        out_194 = None
        out_195 = mul_230 + out_188
        mul_230 = out_188 = None
        gelu_115 = torch._C._nn.gelu(out_195)
        mul__143 = gelu_115.mul_(1.7015043497085571)
        gelu_115 = None
        out_196 = mul__143 * 0.7808688094430302
        mul__143 = None
        reshape_119 = l_self_modules_stages_modules_2_modules_16_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_232 = (
            l_self_modules_stages_modules_2_modules_16_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_2_modules_16_modules_conv1_parameters_gain_ = None
        view_119 = mul_232.view(-1)
        mul_232 = None
        batch_norm_119 = torch.nn.functional.batch_norm(
            reshape_119,
            None,
            None,
            weight=view_119,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_119 = view_119 = None
        weight_119 = batch_norm_119.reshape_as(
            l_self_modules_stages_modules_2_modules_16_modules_conv1_parameters_weight_
        )
        batch_norm_119 = (
            l_self_modules_stages_modules_2_modules_16_modules_conv1_parameters_weight_
        ) = None
        out_197 = torch.conv2d(
            out_196,
            weight_119,
            l_self_modules_stages_modules_2_modules_16_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_196 = (
            weight_119
        ) = (
            l_self_modules_stages_modules_2_modules_16_modules_conv1_parameters_bias_
        ) = None
        gelu_116 = torch._C._nn.gelu(out_197)
        out_197 = None
        mul__144 = gelu_116.mul_(1.7015043497085571)
        gelu_116 = None
        reshape_120 = l_self_modules_stages_modules_2_modules_16_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_233 = (
            l_self_modules_stages_modules_2_modules_16_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_16_modules_conv2_parameters_gain_ = None
        view_120 = mul_233.view(-1)
        mul_233 = None
        batch_norm_120 = torch.nn.functional.batch_norm(
            reshape_120,
            None,
            None,
            weight=view_120,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_120 = view_120 = None
        weight_120 = batch_norm_120.reshape_as(
            l_self_modules_stages_modules_2_modules_16_modules_conv2_parameters_weight_
        )
        batch_norm_120 = (
            l_self_modules_stages_modules_2_modules_16_modules_conv2_parameters_weight_
        ) = None
        out_198 = torch.conv2d(
            mul__144,
            weight_120,
            l_self_modules_stages_modules_2_modules_16_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__144 = (
            weight_120
        ) = (
            l_self_modules_stages_modules_2_modules_16_modules_conv2_parameters_bias_
        ) = None
        gelu_117 = torch._C._nn.gelu(out_198)
        out_198 = None
        mul__145 = gelu_117.mul_(1.7015043497085571)
        gelu_117 = None
        reshape_121 = l_self_modules_stages_modules_2_modules_16_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_234 = (
            l_self_modules_stages_modules_2_modules_16_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_16_modules_conv2b_parameters_gain_ = (
            None
        )
        view_121 = mul_234.view(-1)
        mul_234 = None
        batch_norm_121 = torch.nn.functional.batch_norm(
            reshape_121,
            None,
            None,
            weight=view_121,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_121 = view_121 = None
        weight_121 = batch_norm_121.reshape_as(
            l_self_modules_stages_modules_2_modules_16_modules_conv2b_parameters_weight_
        )
        batch_norm_121 = (
            l_self_modules_stages_modules_2_modules_16_modules_conv2b_parameters_weight_
        ) = None
        out_199 = torch.conv2d(
            mul__145,
            weight_121,
            l_self_modules_stages_modules_2_modules_16_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__145 = (
            weight_121
        ) = (
            l_self_modules_stages_modules_2_modules_16_modules_conv2b_parameters_bias_
        ) = None
        gelu_118 = torch._C._nn.gelu(out_199)
        out_199 = None
        mul__146 = gelu_118.mul_(1.7015043497085571)
        gelu_118 = None
        reshape_122 = l_self_modules_stages_modules_2_modules_16_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_235 = (
            l_self_modules_stages_modules_2_modules_16_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_16_modules_conv3_parameters_gain_ = None
        view_122 = mul_235.view(-1)
        mul_235 = None
        batch_norm_122 = torch.nn.functional.batch_norm(
            reshape_122,
            None,
            None,
            weight=view_122,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_122 = view_122 = None
        weight_122 = batch_norm_122.reshape_as(
            l_self_modules_stages_modules_2_modules_16_modules_conv3_parameters_weight_
        )
        batch_norm_122 = (
            l_self_modules_stages_modules_2_modules_16_modules_conv3_parameters_weight_
        ) = None
        out_200 = torch.conv2d(
            mul__146,
            weight_122,
            l_self_modules_stages_modules_2_modules_16_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__146 = (
            weight_122
        ) = (
            l_self_modules_stages_modules_2_modules_16_modules_conv3_parameters_bias_
        ) = None
        x_se_112 = out_200.mean((2, 3), keepdim=True)
        x_se_113 = torch.conv2d(
            x_se_112,
            l_self_modules_stages_modules_2_modules_16_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_16_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_112 = l_self_modules_stages_modules_2_modules_16_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_16_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_114 = torch.nn.functional.relu(x_se_113, inplace=True)
        x_se_113 = None
        x_se_115 = torch.conv2d(
            x_se_114,
            l_self_modules_stages_modules_2_modules_16_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_16_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_114 = l_self_modules_stages_modules_2_modules_16_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_16_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_28 = x_se_115.sigmoid()
        x_se_115 = None
        mul_236 = out_200 * sigmoid_28
        out_200 = sigmoid_28 = None
        out_201 = 2.0 * mul_236
        mul_236 = None
        mul__147 = out_201.mul_(
            l_self_modules_stages_modules_2_modules_16_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_16_parameters_skipinit_gain_ = (
            mul__147
        ) = None
        mul_238 = out_201 * 0.2
        out_201 = None
        out_202 = mul_238 + out_195
        mul_238 = out_195 = None
        gelu_119 = torch._C._nn.gelu(out_202)
        mul__148 = gelu_119.mul_(1.7015043497085571)
        gelu_119 = None
        out_203 = mul__148 * 0.7715167498104594
        mul__148 = None
        reshape_123 = l_self_modules_stages_modules_2_modules_17_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_240 = (
            l_self_modules_stages_modules_2_modules_17_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_2_modules_17_modules_conv1_parameters_gain_ = None
        view_123 = mul_240.view(-1)
        mul_240 = None
        batch_norm_123 = torch.nn.functional.batch_norm(
            reshape_123,
            None,
            None,
            weight=view_123,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_123 = view_123 = None
        weight_123 = batch_norm_123.reshape_as(
            l_self_modules_stages_modules_2_modules_17_modules_conv1_parameters_weight_
        )
        batch_norm_123 = (
            l_self_modules_stages_modules_2_modules_17_modules_conv1_parameters_weight_
        ) = None
        out_204 = torch.conv2d(
            out_203,
            weight_123,
            l_self_modules_stages_modules_2_modules_17_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_203 = (
            weight_123
        ) = (
            l_self_modules_stages_modules_2_modules_17_modules_conv1_parameters_bias_
        ) = None
        gelu_120 = torch._C._nn.gelu(out_204)
        out_204 = None
        mul__149 = gelu_120.mul_(1.7015043497085571)
        gelu_120 = None
        reshape_124 = l_self_modules_stages_modules_2_modules_17_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_241 = (
            l_self_modules_stages_modules_2_modules_17_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_17_modules_conv2_parameters_gain_ = None
        view_124 = mul_241.view(-1)
        mul_241 = None
        batch_norm_124 = torch.nn.functional.batch_norm(
            reshape_124,
            None,
            None,
            weight=view_124,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_124 = view_124 = None
        weight_124 = batch_norm_124.reshape_as(
            l_self_modules_stages_modules_2_modules_17_modules_conv2_parameters_weight_
        )
        batch_norm_124 = (
            l_self_modules_stages_modules_2_modules_17_modules_conv2_parameters_weight_
        ) = None
        out_205 = torch.conv2d(
            mul__149,
            weight_124,
            l_self_modules_stages_modules_2_modules_17_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__149 = (
            weight_124
        ) = (
            l_self_modules_stages_modules_2_modules_17_modules_conv2_parameters_bias_
        ) = None
        gelu_121 = torch._C._nn.gelu(out_205)
        out_205 = None
        mul__150 = gelu_121.mul_(1.7015043497085571)
        gelu_121 = None
        reshape_125 = l_self_modules_stages_modules_2_modules_17_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_242 = (
            l_self_modules_stages_modules_2_modules_17_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_17_modules_conv2b_parameters_gain_ = (
            None
        )
        view_125 = mul_242.view(-1)
        mul_242 = None
        batch_norm_125 = torch.nn.functional.batch_norm(
            reshape_125,
            None,
            None,
            weight=view_125,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_125 = view_125 = None
        weight_125 = batch_norm_125.reshape_as(
            l_self_modules_stages_modules_2_modules_17_modules_conv2b_parameters_weight_
        )
        batch_norm_125 = (
            l_self_modules_stages_modules_2_modules_17_modules_conv2b_parameters_weight_
        ) = None
        out_206 = torch.conv2d(
            mul__150,
            weight_125,
            l_self_modules_stages_modules_2_modules_17_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__150 = (
            weight_125
        ) = (
            l_self_modules_stages_modules_2_modules_17_modules_conv2b_parameters_bias_
        ) = None
        gelu_122 = torch._C._nn.gelu(out_206)
        out_206 = None
        mul__151 = gelu_122.mul_(1.7015043497085571)
        gelu_122 = None
        reshape_126 = l_self_modules_stages_modules_2_modules_17_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_243 = (
            l_self_modules_stages_modules_2_modules_17_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_17_modules_conv3_parameters_gain_ = None
        view_126 = mul_243.view(-1)
        mul_243 = None
        batch_norm_126 = torch.nn.functional.batch_norm(
            reshape_126,
            None,
            None,
            weight=view_126,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_126 = view_126 = None
        weight_126 = batch_norm_126.reshape_as(
            l_self_modules_stages_modules_2_modules_17_modules_conv3_parameters_weight_
        )
        batch_norm_126 = (
            l_self_modules_stages_modules_2_modules_17_modules_conv3_parameters_weight_
        ) = None
        out_207 = torch.conv2d(
            mul__151,
            weight_126,
            l_self_modules_stages_modules_2_modules_17_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__151 = (
            weight_126
        ) = (
            l_self_modules_stages_modules_2_modules_17_modules_conv3_parameters_bias_
        ) = None
        x_se_116 = out_207.mean((2, 3), keepdim=True)
        x_se_117 = torch.conv2d(
            x_se_116,
            l_self_modules_stages_modules_2_modules_17_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_17_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_116 = l_self_modules_stages_modules_2_modules_17_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_17_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_118 = torch.nn.functional.relu(x_se_117, inplace=True)
        x_se_117 = None
        x_se_119 = torch.conv2d(
            x_se_118,
            l_self_modules_stages_modules_2_modules_17_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_17_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_118 = l_self_modules_stages_modules_2_modules_17_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_17_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_29 = x_se_119.sigmoid()
        x_se_119 = None
        mul_244 = out_207 * sigmoid_29
        out_207 = sigmoid_29 = None
        out_208 = 2.0 * mul_244
        mul_244 = None
        mul__152 = out_208.mul_(
            l_self_modules_stages_modules_2_modules_17_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_17_parameters_skipinit_gain_ = (
            mul__152
        ) = None
        mul_246 = out_208 * 0.2
        out_208 = None
        out_209 = mul_246 + out_202
        mul_246 = out_202 = None
        gelu_123 = torch._C._nn.gelu(out_209)
        mul__153 = gelu_123.mul_(1.7015043497085571)
        gelu_123 = None
        out_210 = mul__153 * 0.7624928516630232
        mul__153 = None
        reshape_127 = l_self_modules_stages_modules_2_modules_18_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_248 = (
            l_self_modules_stages_modules_2_modules_18_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_2_modules_18_modules_conv1_parameters_gain_ = None
        view_127 = mul_248.view(-1)
        mul_248 = None
        batch_norm_127 = torch.nn.functional.batch_norm(
            reshape_127,
            None,
            None,
            weight=view_127,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_127 = view_127 = None
        weight_127 = batch_norm_127.reshape_as(
            l_self_modules_stages_modules_2_modules_18_modules_conv1_parameters_weight_
        )
        batch_norm_127 = (
            l_self_modules_stages_modules_2_modules_18_modules_conv1_parameters_weight_
        ) = None
        out_211 = torch.conv2d(
            out_210,
            weight_127,
            l_self_modules_stages_modules_2_modules_18_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_210 = (
            weight_127
        ) = (
            l_self_modules_stages_modules_2_modules_18_modules_conv1_parameters_bias_
        ) = None
        gelu_124 = torch._C._nn.gelu(out_211)
        out_211 = None
        mul__154 = gelu_124.mul_(1.7015043497085571)
        gelu_124 = None
        reshape_128 = l_self_modules_stages_modules_2_modules_18_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_249 = (
            l_self_modules_stages_modules_2_modules_18_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_18_modules_conv2_parameters_gain_ = None
        view_128 = mul_249.view(-1)
        mul_249 = None
        batch_norm_128 = torch.nn.functional.batch_norm(
            reshape_128,
            None,
            None,
            weight=view_128,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_128 = view_128 = None
        weight_128 = batch_norm_128.reshape_as(
            l_self_modules_stages_modules_2_modules_18_modules_conv2_parameters_weight_
        )
        batch_norm_128 = (
            l_self_modules_stages_modules_2_modules_18_modules_conv2_parameters_weight_
        ) = None
        out_212 = torch.conv2d(
            mul__154,
            weight_128,
            l_self_modules_stages_modules_2_modules_18_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__154 = (
            weight_128
        ) = (
            l_self_modules_stages_modules_2_modules_18_modules_conv2_parameters_bias_
        ) = None
        gelu_125 = torch._C._nn.gelu(out_212)
        out_212 = None
        mul__155 = gelu_125.mul_(1.7015043497085571)
        gelu_125 = None
        reshape_129 = l_self_modules_stages_modules_2_modules_18_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_250 = (
            l_self_modules_stages_modules_2_modules_18_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_18_modules_conv2b_parameters_gain_ = (
            None
        )
        view_129 = mul_250.view(-1)
        mul_250 = None
        batch_norm_129 = torch.nn.functional.batch_norm(
            reshape_129,
            None,
            None,
            weight=view_129,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_129 = view_129 = None
        weight_129 = batch_norm_129.reshape_as(
            l_self_modules_stages_modules_2_modules_18_modules_conv2b_parameters_weight_
        )
        batch_norm_129 = (
            l_self_modules_stages_modules_2_modules_18_modules_conv2b_parameters_weight_
        ) = None
        out_213 = torch.conv2d(
            mul__155,
            weight_129,
            l_self_modules_stages_modules_2_modules_18_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__155 = (
            weight_129
        ) = (
            l_self_modules_stages_modules_2_modules_18_modules_conv2b_parameters_bias_
        ) = None
        gelu_126 = torch._C._nn.gelu(out_213)
        out_213 = None
        mul__156 = gelu_126.mul_(1.7015043497085571)
        gelu_126 = None
        reshape_130 = l_self_modules_stages_modules_2_modules_18_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_251 = (
            l_self_modules_stages_modules_2_modules_18_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_18_modules_conv3_parameters_gain_ = None
        view_130 = mul_251.view(-1)
        mul_251 = None
        batch_norm_130 = torch.nn.functional.batch_norm(
            reshape_130,
            None,
            None,
            weight=view_130,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_130 = view_130 = None
        weight_130 = batch_norm_130.reshape_as(
            l_self_modules_stages_modules_2_modules_18_modules_conv3_parameters_weight_
        )
        batch_norm_130 = (
            l_self_modules_stages_modules_2_modules_18_modules_conv3_parameters_weight_
        ) = None
        out_214 = torch.conv2d(
            mul__156,
            weight_130,
            l_self_modules_stages_modules_2_modules_18_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__156 = (
            weight_130
        ) = (
            l_self_modules_stages_modules_2_modules_18_modules_conv3_parameters_bias_
        ) = None
        x_se_120 = out_214.mean((2, 3), keepdim=True)
        x_se_121 = torch.conv2d(
            x_se_120,
            l_self_modules_stages_modules_2_modules_18_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_18_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_120 = l_self_modules_stages_modules_2_modules_18_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_18_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_122 = torch.nn.functional.relu(x_se_121, inplace=True)
        x_se_121 = None
        x_se_123 = torch.conv2d(
            x_se_122,
            l_self_modules_stages_modules_2_modules_18_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_18_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_122 = l_self_modules_stages_modules_2_modules_18_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_18_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_30 = x_se_123.sigmoid()
        x_se_123 = None
        mul_252 = out_214 * sigmoid_30
        out_214 = sigmoid_30 = None
        out_215 = 2.0 * mul_252
        mul_252 = None
        mul__157 = out_215.mul_(
            l_self_modules_stages_modules_2_modules_18_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_18_parameters_skipinit_gain_ = (
            mul__157
        ) = None
        mul_254 = out_215 * 0.2
        out_215 = None
        out_216 = mul_254 + out_209
        mul_254 = out_209 = None
        gelu_127 = torch._C._nn.gelu(out_216)
        mul__158 = gelu_127.mul_(1.7015043497085571)
        gelu_127 = None
        out_217 = mul__158 * 0.753778361444409
        mul__158 = None
        reshape_131 = l_self_modules_stages_modules_2_modules_19_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_256 = (
            l_self_modules_stages_modules_2_modules_19_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_2_modules_19_modules_conv1_parameters_gain_ = None
        view_131 = mul_256.view(-1)
        mul_256 = None
        batch_norm_131 = torch.nn.functional.batch_norm(
            reshape_131,
            None,
            None,
            weight=view_131,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_131 = view_131 = None
        weight_131 = batch_norm_131.reshape_as(
            l_self_modules_stages_modules_2_modules_19_modules_conv1_parameters_weight_
        )
        batch_norm_131 = (
            l_self_modules_stages_modules_2_modules_19_modules_conv1_parameters_weight_
        ) = None
        out_218 = torch.conv2d(
            out_217,
            weight_131,
            l_self_modules_stages_modules_2_modules_19_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_217 = (
            weight_131
        ) = (
            l_self_modules_stages_modules_2_modules_19_modules_conv1_parameters_bias_
        ) = None
        gelu_128 = torch._C._nn.gelu(out_218)
        out_218 = None
        mul__159 = gelu_128.mul_(1.7015043497085571)
        gelu_128 = None
        reshape_132 = l_self_modules_stages_modules_2_modules_19_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_257 = (
            l_self_modules_stages_modules_2_modules_19_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_19_modules_conv2_parameters_gain_ = None
        view_132 = mul_257.view(-1)
        mul_257 = None
        batch_norm_132 = torch.nn.functional.batch_norm(
            reshape_132,
            None,
            None,
            weight=view_132,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_132 = view_132 = None
        weight_132 = batch_norm_132.reshape_as(
            l_self_modules_stages_modules_2_modules_19_modules_conv2_parameters_weight_
        )
        batch_norm_132 = (
            l_self_modules_stages_modules_2_modules_19_modules_conv2_parameters_weight_
        ) = None
        out_219 = torch.conv2d(
            mul__159,
            weight_132,
            l_self_modules_stages_modules_2_modules_19_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__159 = (
            weight_132
        ) = (
            l_self_modules_stages_modules_2_modules_19_modules_conv2_parameters_bias_
        ) = None
        gelu_129 = torch._C._nn.gelu(out_219)
        out_219 = None
        mul__160 = gelu_129.mul_(1.7015043497085571)
        gelu_129 = None
        reshape_133 = l_self_modules_stages_modules_2_modules_19_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_258 = (
            l_self_modules_stages_modules_2_modules_19_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_19_modules_conv2b_parameters_gain_ = (
            None
        )
        view_133 = mul_258.view(-1)
        mul_258 = None
        batch_norm_133 = torch.nn.functional.batch_norm(
            reshape_133,
            None,
            None,
            weight=view_133,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_133 = view_133 = None
        weight_133 = batch_norm_133.reshape_as(
            l_self_modules_stages_modules_2_modules_19_modules_conv2b_parameters_weight_
        )
        batch_norm_133 = (
            l_self_modules_stages_modules_2_modules_19_modules_conv2b_parameters_weight_
        ) = None
        out_220 = torch.conv2d(
            mul__160,
            weight_133,
            l_self_modules_stages_modules_2_modules_19_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__160 = (
            weight_133
        ) = (
            l_self_modules_stages_modules_2_modules_19_modules_conv2b_parameters_bias_
        ) = None
        gelu_130 = torch._C._nn.gelu(out_220)
        out_220 = None
        mul__161 = gelu_130.mul_(1.7015043497085571)
        gelu_130 = None
        reshape_134 = l_self_modules_stages_modules_2_modules_19_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_259 = (
            l_self_modules_stages_modules_2_modules_19_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_19_modules_conv3_parameters_gain_ = None
        view_134 = mul_259.view(-1)
        mul_259 = None
        batch_norm_134 = torch.nn.functional.batch_norm(
            reshape_134,
            None,
            None,
            weight=view_134,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_134 = view_134 = None
        weight_134 = batch_norm_134.reshape_as(
            l_self_modules_stages_modules_2_modules_19_modules_conv3_parameters_weight_
        )
        batch_norm_134 = (
            l_self_modules_stages_modules_2_modules_19_modules_conv3_parameters_weight_
        ) = None
        out_221 = torch.conv2d(
            mul__161,
            weight_134,
            l_self_modules_stages_modules_2_modules_19_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__161 = (
            weight_134
        ) = (
            l_self_modules_stages_modules_2_modules_19_modules_conv3_parameters_bias_
        ) = None
        x_se_124 = out_221.mean((2, 3), keepdim=True)
        x_se_125 = torch.conv2d(
            x_se_124,
            l_self_modules_stages_modules_2_modules_19_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_19_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_124 = l_self_modules_stages_modules_2_modules_19_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_19_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_126 = torch.nn.functional.relu(x_se_125, inplace=True)
        x_se_125 = None
        x_se_127 = torch.conv2d(
            x_se_126,
            l_self_modules_stages_modules_2_modules_19_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_19_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_126 = l_self_modules_stages_modules_2_modules_19_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_19_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_31 = x_se_127.sigmoid()
        x_se_127 = None
        mul_260 = out_221 * sigmoid_31
        out_221 = sigmoid_31 = None
        out_222 = 2.0 * mul_260
        mul_260 = None
        mul__162 = out_222.mul_(
            l_self_modules_stages_modules_2_modules_19_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_19_parameters_skipinit_gain_ = (
            mul__162
        ) = None
        mul_262 = out_222 * 0.2
        out_222 = None
        out_223 = mul_262 + out_216
        mul_262 = out_216 = None
        gelu_131 = torch._C._nn.gelu(out_223)
        mul__163 = gelu_131.mul_(1.7015043497085571)
        gelu_131 = None
        out_224 = mul__163 * 0.7453559924999298
        mul__163 = None
        reshape_135 = l_self_modules_stages_modules_2_modules_20_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_264 = (
            l_self_modules_stages_modules_2_modules_20_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_2_modules_20_modules_conv1_parameters_gain_ = None
        view_135 = mul_264.view(-1)
        mul_264 = None
        batch_norm_135 = torch.nn.functional.batch_norm(
            reshape_135,
            None,
            None,
            weight=view_135,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_135 = view_135 = None
        weight_135 = batch_norm_135.reshape_as(
            l_self_modules_stages_modules_2_modules_20_modules_conv1_parameters_weight_
        )
        batch_norm_135 = (
            l_self_modules_stages_modules_2_modules_20_modules_conv1_parameters_weight_
        ) = None
        out_225 = torch.conv2d(
            out_224,
            weight_135,
            l_self_modules_stages_modules_2_modules_20_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_224 = (
            weight_135
        ) = (
            l_self_modules_stages_modules_2_modules_20_modules_conv1_parameters_bias_
        ) = None
        gelu_132 = torch._C._nn.gelu(out_225)
        out_225 = None
        mul__164 = gelu_132.mul_(1.7015043497085571)
        gelu_132 = None
        reshape_136 = l_self_modules_stages_modules_2_modules_20_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_265 = (
            l_self_modules_stages_modules_2_modules_20_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_20_modules_conv2_parameters_gain_ = None
        view_136 = mul_265.view(-1)
        mul_265 = None
        batch_norm_136 = torch.nn.functional.batch_norm(
            reshape_136,
            None,
            None,
            weight=view_136,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_136 = view_136 = None
        weight_136 = batch_norm_136.reshape_as(
            l_self_modules_stages_modules_2_modules_20_modules_conv2_parameters_weight_
        )
        batch_norm_136 = (
            l_self_modules_stages_modules_2_modules_20_modules_conv2_parameters_weight_
        ) = None
        out_226 = torch.conv2d(
            mul__164,
            weight_136,
            l_self_modules_stages_modules_2_modules_20_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__164 = (
            weight_136
        ) = (
            l_self_modules_stages_modules_2_modules_20_modules_conv2_parameters_bias_
        ) = None
        gelu_133 = torch._C._nn.gelu(out_226)
        out_226 = None
        mul__165 = gelu_133.mul_(1.7015043497085571)
        gelu_133 = None
        reshape_137 = l_self_modules_stages_modules_2_modules_20_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_266 = (
            l_self_modules_stages_modules_2_modules_20_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_20_modules_conv2b_parameters_gain_ = (
            None
        )
        view_137 = mul_266.view(-1)
        mul_266 = None
        batch_norm_137 = torch.nn.functional.batch_norm(
            reshape_137,
            None,
            None,
            weight=view_137,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_137 = view_137 = None
        weight_137 = batch_norm_137.reshape_as(
            l_self_modules_stages_modules_2_modules_20_modules_conv2b_parameters_weight_
        )
        batch_norm_137 = (
            l_self_modules_stages_modules_2_modules_20_modules_conv2b_parameters_weight_
        ) = None
        out_227 = torch.conv2d(
            mul__165,
            weight_137,
            l_self_modules_stages_modules_2_modules_20_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__165 = (
            weight_137
        ) = (
            l_self_modules_stages_modules_2_modules_20_modules_conv2b_parameters_bias_
        ) = None
        gelu_134 = torch._C._nn.gelu(out_227)
        out_227 = None
        mul__166 = gelu_134.mul_(1.7015043497085571)
        gelu_134 = None
        reshape_138 = l_self_modules_stages_modules_2_modules_20_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_267 = (
            l_self_modules_stages_modules_2_modules_20_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_20_modules_conv3_parameters_gain_ = None
        view_138 = mul_267.view(-1)
        mul_267 = None
        batch_norm_138 = torch.nn.functional.batch_norm(
            reshape_138,
            None,
            None,
            weight=view_138,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_138 = view_138 = None
        weight_138 = batch_norm_138.reshape_as(
            l_self_modules_stages_modules_2_modules_20_modules_conv3_parameters_weight_
        )
        batch_norm_138 = (
            l_self_modules_stages_modules_2_modules_20_modules_conv3_parameters_weight_
        ) = None
        out_228 = torch.conv2d(
            mul__166,
            weight_138,
            l_self_modules_stages_modules_2_modules_20_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__166 = (
            weight_138
        ) = (
            l_self_modules_stages_modules_2_modules_20_modules_conv3_parameters_bias_
        ) = None
        x_se_128 = out_228.mean((2, 3), keepdim=True)
        x_se_129 = torch.conv2d(
            x_se_128,
            l_self_modules_stages_modules_2_modules_20_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_20_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_128 = l_self_modules_stages_modules_2_modules_20_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_20_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_130 = torch.nn.functional.relu(x_se_129, inplace=True)
        x_se_129 = None
        x_se_131 = torch.conv2d(
            x_se_130,
            l_self_modules_stages_modules_2_modules_20_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_20_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_130 = l_self_modules_stages_modules_2_modules_20_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_20_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_32 = x_se_131.sigmoid()
        x_se_131 = None
        mul_268 = out_228 * sigmoid_32
        out_228 = sigmoid_32 = None
        out_229 = 2.0 * mul_268
        mul_268 = None
        mul__167 = out_229.mul_(
            l_self_modules_stages_modules_2_modules_20_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_20_parameters_skipinit_gain_ = (
            mul__167
        ) = None
        mul_270 = out_229 * 0.2
        out_229 = None
        out_230 = mul_270 + out_223
        mul_270 = out_223 = None
        gelu_135 = torch._C._nn.gelu(out_230)
        mul__168 = gelu_135.mul_(1.7015043497085571)
        gelu_135 = None
        out_231 = mul__168 * 0.7372097807744855
        mul__168 = None
        reshape_139 = l_self_modules_stages_modules_2_modules_21_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_272 = (
            l_self_modules_stages_modules_2_modules_21_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_2_modules_21_modules_conv1_parameters_gain_ = None
        view_139 = mul_272.view(-1)
        mul_272 = None
        batch_norm_139 = torch.nn.functional.batch_norm(
            reshape_139,
            None,
            None,
            weight=view_139,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_139 = view_139 = None
        weight_139 = batch_norm_139.reshape_as(
            l_self_modules_stages_modules_2_modules_21_modules_conv1_parameters_weight_
        )
        batch_norm_139 = (
            l_self_modules_stages_modules_2_modules_21_modules_conv1_parameters_weight_
        ) = None
        out_232 = torch.conv2d(
            out_231,
            weight_139,
            l_self_modules_stages_modules_2_modules_21_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_231 = (
            weight_139
        ) = (
            l_self_modules_stages_modules_2_modules_21_modules_conv1_parameters_bias_
        ) = None
        gelu_136 = torch._C._nn.gelu(out_232)
        out_232 = None
        mul__169 = gelu_136.mul_(1.7015043497085571)
        gelu_136 = None
        reshape_140 = l_self_modules_stages_modules_2_modules_21_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_273 = (
            l_self_modules_stages_modules_2_modules_21_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_21_modules_conv2_parameters_gain_ = None
        view_140 = mul_273.view(-1)
        mul_273 = None
        batch_norm_140 = torch.nn.functional.batch_norm(
            reshape_140,
            None,
            None,
            weight=view_140,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_140 = view_140 = None
        weight_140 = batch_norm_140.reshape_as(
            l_self_modules_stages_modules_2_modules_21_modules_conv2_parameters_weight_
        )
        batch_norm_140 = (
            l_self_modules_stages_modules_2_modules_21_modules_conv2_parameters_weight_
        ) = None
        out_233 = torch.conv2d(
            mul__169,
            weight_140,
            l_self_modules_stages_modules_2_modules_21_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__169 = (
            weight_140
        ) = (
            l_self_modules_stages_modules_2_modules_21_modules_conv2_parameters_bias_
        ) = None
        gelu_137 = torch._C._nn.gelu(out_233)
        out_233 = None
        mul__170 = gelu_137.mul_(1.7015043497085571)
        gelu_137 = None
        reshape_141 = l_self_modules_stages_modules_2_modules_21_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_274 = (
            l_self_modules_stages_modules_2_modules_21_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_21_modules_conv2b_parameters_gain_ = (
            None
        )
        view_141 = mul_274.view(-1)
        mul_274 = None
        batch_norm_141 = torch.nn.functional.batch_norm(
            reshape_141,
            None,
            None,
            weight=view_141,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_141 = view_141 = None
        weight_141 = batch_norm_141.reshape_as(
            l_self_modules_stages_modules_2_modules_21_modules_conv2b_parameters_weight_
        )
        batch_norm_141 = (
            l_self_modules_stages_modules_2_modules_21_modules_conv2b_parameters_weight_
        ) = None
        out_234 = torch.conv2d(
            mul__170,
            weight_141,
            l_self_modules_stages_modules_2_modules_21_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__170 = (
            weight_141
        ) = (
            l_self_modules_stages_modules_2_modules_21_modules_conv2b_parameters_bias_
        ) = None
        gelu_138 = torch._C._nn.gelu(out_234)
        out_234 = None
        mul__171 = gelu_138.mul_(1.7015043497085571)
        gelu_138 = None
        reshape_142 = l_self_modules_stages_modules_2_modules_21_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_275 = (
            l_self_modules_stages_modules_2_modules_21_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_21_modules_conv3_parameters_gain_ = None
        view_142 = mul_275.view(-1)
        mul_275 = None
        batch_norm_142 = torch.nn.functional.batch_norm(
            reshape_142,
            None,
            None,
            weight=view_142,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_142 = view_142 = None
        weight_142 = batch_norm_142.reshape_as(
            l_self_modules_stages_modules_2_modules_21_modules_conv3_parameters_weight_
        )
        batch_norm_142 = (
            l_self_modules_stages_modules_2_modules_21_modules_conv3_parameters_weight_
        ) = None
        out_235 = torch.conv2d(
            mul__171,
            weight_142,
            l_self_modules_stages_modules_2_modules_21_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__171 = (
            weight_142
        ) = (
            l_self_modules_stages_modules_2_modules_21_modules_conv3_parameters_bias_
        ) = None
        x_se_132 = out_235.mean((2, 3), keepdim=True)
        x_se_133 = torch.conv2d(
            x_se_132,
            l_self_modules_stages_modules_2_modules_21_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_21_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_132 = l_self_modules_stages_modules_2_modules_21_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_21_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_134 = torch.nn.functional.relu(x_se_133, inplace=True)
        x_se_133 = None
        x_se_135 = torch.conv2d(
            x_se_134,
            l_self_modules_stages_modules_2_modules_21_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_21_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_134 = l_self_modules_stages_modules_2_modules_21_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_21_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_33 = x_se_135.sigmoid()
        x_se_135 = None
        mul_276 = out_235 * sigmoid_33
        out_235 = sigmoid_33 = None
        out_236 = 2.0 * mul_276
        mul_276 = None
        mul__172 = out_236.mul_(
            l_self_modules_stages_modules_2_modules_21_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_21_parameters_skipinit_gain_ = (
            mul__172
        ) = None
        mul_278 = out_236 * 0.2
        out_236 = None
        out_237 = mul_278 + out_230
        mul_278 = out_230 = None
        gelu_139 = torch._C._nn.gelu(out_237)
        mul__173 = gelu_139.mul_(1.7015043497085571)
        gelu_139 = None
        out_238 = mul__173 * 0.7293249574894727
        mul__173 = None
        reshape_143 = l_self_modules_stages_modules_2_modules_22_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_280 = (
            l_self_modules_stages_modules_2_modules_22_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_2_modules_22_modules_conv1_parameters_gain_ = None
        view_143 = mul_280.view(-1)
        mul_280 = None
        batch_norm_143 = torch.nn.functional.batch_norm(
            reshape_143,
            None,
            None,
            weight=view_143,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_143 = view_143 = None
        weight_143 = batch_norm_143.reshape_as(
            l_self_modules_stages_modules_2_modules_22_modules_conv1_parameters_weight_
        )
        batch_norm_143 = (
            l_self_modules_stages_modules_2_modules_22_modules_conv1_parameters_weight_
        ) = None
        out_239 = torch.conv2d(
            out_238,
            weight_143,
            l_self_modules_stages_modules_2_modules_22_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_238 = (
            weight_143
        ) = (
            l_self_modules_stages_modules_2_modules_22_modules_conv1_parameters_bias_
        ) = None
        gelu_140 = torch._C._nn.gelu(out_239)
        out_239 = None
        mul__174 = gelu_140.mul_(1.7015043497085571)
        gelu_140 = None
        reshape_144 = l_self_modules_stages_modules_2_modules_22_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_281 = (
            l_self_modules_stages_modules_2_modules_22_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_22_modules_conv2_parameters_gain_ = None
        view_144 = mul_281.view(-1)
        mul_281 = None
        batch_norm_144 = torch.nn.functional.batch_norm(
            reshape_144,
            None,
            None,
            weight=view_144,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_144 = view_144 = None
        weight_144 = batch_norm_144.reshape_as(
            l_self_modules_stages_modules_2_modules_22_modules_conv2_parameters_weight_
        )
        batch_norm_144 = (
            l_self_modules_stages_modules_2_modules_22_modules_conv2_parameters_weight_
        ) = None
        out_240 = torch.conv2d(
            mul__174,
            weight_144,
            l_self_modules_stages_modules_2_modules_22_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__174 = (
            weight_144
        ) = (
            l_self_modules_stages_modules_2_modules_22_modules_conv2_parameters_bias_
        ) = None
        gelu_141 = torch._C._nn.gelu(out_240)
        out_240 = None
        mul__175 = gelu_141.mul_(1.7015043497085571)
        gelu_141 = None
        reshape_145 = l_self_modules_stages_modules_2_modules_22_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_282 = (
            l_self_modules_stages_modules_2_modules_22_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_22_modules_conv2b_parameters_gain_ = (
            None
        )
        view_145 = mul_282.view(-1)
        mul_282 = None
        batch_norm_145 = torch.nn.functional.batch_norm(
            reshape_145,
            None,
            None,
            weight=view_145,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_145 = view_145 = None
        weight_145 = batch_norm_145.reshape_as(
            l_self_modules_stages_modules_2_modules_22_modules_conv2b_parameters_weight_
        )
        batch_norm_145 = (
            l_self_modules_stages_modules_2_modules_22_modules_conv2b_parameters_weight_
        ) = None
        out_241 = torch.conv2d(
            mul__175,
            weight_145,
            l_self_modules_stages_modules_2_modules_22_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__175 = (
            weight_145
        ) = (
            l_self_modules_stages_modules_2_modules_22_modules_conv2b_parameters_bias_
        ) = None
        gelu_142 = torch._C._nn.gelu(out_241)
        out_241 = None
        mul__176 = gelu_142.mul_(1.7015043497085571)
        gelu_142 = None
        reshape_146 = l_self_modules_stages_modules_2_modules_22_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_283 = (
            l_self_modules_stages_modules_2_modules_22_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_22_modules_conv3_parameters_gain_ = None
        view_146 = mul_283.view(-1)
        mul_283 = None
        batch_norm_146 = torch.nn.functional.batch_norm(
            reshape_146,
            None,
            None,
            weight=view_146,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_146 = view_146 = None
        weight_146 = batch_norm_146.reshape_as(
            l_self_modules_stages_modules_2_modules_22_modules_conv3_parameters_weight_
        )
        batch_norm_146 = (
            l_self_modules_stages_modules_2_modules_22_modules_conv3_parameters_weight_
        ) = None
        out_242 = torch.conv2d(
            mul__176,
            weight_146,
            l_self_modules_stages_modules_2_modules_22_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__176 = (
            weight_146
        ) = (
            l_self_modules_stages_modules_2_modules_22_modules_conv3_parameters_bias_
        ) = None
        x_se_136 = out_242.mean((2, 3), keepdim=True)
        x_se_137 = torch.conv2d(
            x_se_136,
            l_self_modules_stages_modules_2_modules_22_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_22_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_136 = l_self_modules_stages_modules_2_modules_22_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_22_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_138 = torch.nn.functional.relu(x_se_137, inplace=True)
        x_se_137 = None
        x_se_139 = torch.conv2d(
            x_se_138,
            l_self_modules_stages_modules_2_modules_22_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_22_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_138 = l_self_modules_stages_modules_2_modules_22_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_22_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_34 = x_se_139.sigmoid()
        x_se_139 = None
        mul_284 = out_242 * sigmoid_34
        out_242 = sigmoid_34 = None
        out_243 = 2.0 * mul_284
        mul_284 = None
        mul__177 = out_243.mul_(
            l_self_modules_stages_modules_2_modules_22_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_22_parameters_skipinit_gain_ = (
            mul__177
        ) = None
        mul_286 = out_243 * 0.2
        out_243 = None
        out_244 = mul_286 + out_237
        mul_286 = out_237 = None
        gelu_143 = torch._C._nn.gelu(out_244)
        mul__178 = gelu_143.mul_(1.7015043497085571)
        gelu_143 = None
        out_245 = mul__178 * 0.721687836487032
        mul__178 = None
        reshape_147 = l_self_modules_stages_modules_2_modules_23_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_288 = (
            l_self_modules_stages_modules_2_modules_23_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_2_modules_23_modules_conv1_parameters_gain_ = None
        view_147 = mul_288.view(-1)
        mul_288 = None
        batch_norm_147 = torch.nn.functional.batch_norm(
            reshape_147,
            None,
            None,
            weight=view_147,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_147 = view_147 = None
        weight_147 = batch_norm_147.reshape_as(
            l_self_modules_stages_modules_2_modules_23_modules_conv1_parameters_weight_
        )
        batch_norm_147 = (
            l_self_modules_stages_modules_2_modules_23_modules_conv1_parameters_weight_
        ) = None
        out_246 = torch.conv2d(
            out_245,
            weight_147,
            l_self_modules_stages_modules_2_modules_23_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_245 = (
            weight_147
        ) = (
            l_self_modules_stages_modules_2_modules_23_modules_conv1_parameters_bias_
        ) = None
        gelu_144 = torch._C._nn.gelu(out_246)
        out_246 = None
        mul__179 = gelu_144.mul_(1.7015043497085571)
        gelu_144 = None
        reshape_148 = l_self_modules_stages_modules_2_modules_23_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_289 = (
            l_self_modules_stages_modules_2_modules_23_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_23_modules_conv2_parameters_gain_ = None
        view_148 = mul_289.view(-1)
        mul_289 = None
        batch_norm_148 = torch.nn.functional.batch_norm(
            reshape_148,
            None,
            None,
            weight=view_148,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_148 = view_148 = None
        weight_148 = batch_norm_148.reshape_as(
            l_self_modules_stages_modules_2_modules_23_modules_conv2_parameters_weight_
        )
        batch_norm_148 = (
            l_self_modules_stages_modules_2_modules_23_modules_conv2_parameters_weight_
        ) = None
        out_247 = torch.conv2d(
            mul__179,
            weight_148,
            l_self_modules_stages_modules_2_modules_23_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__179 = (
            weight_148
        ) = (
            l_self_modules_stages_modules_2_modules_23_modules_conv2_parameters_bias_
        ) = None
        gelu_145 = torch._C._nn.gelu(out_247)
        out_247 = None
        mul__180 = gelu_145.mul_(1.7015043497085571)
        gelu_145 = None
        reshape_149 = l_self_modules_stages_modules_2_modules_23_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_290 = (
            l_self_modules_stages_modules_2_modules_23_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_23_modules_conv2b_parameters_gain_ = (
            None
        )
        view_149 = mul_290.view(-1)
        mul_290 = None
        batch_norm_149 = torch.nn.functional.batch_norm(
            reshape_149,
            None,
            None,
            weight=view_149,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_149 = view_149 = None
        weight_149 = batch_norm_149.reshape_as(
            l_self_modules_stages_modules_2_modules_23_modules_conv2b_parameters_weight_
        )
        batch_norm_149 = (
            l_self_modules_stages_modules_2_modules_23_modules_conv2b_parameters_weight_
        ) = None
        out_248 = torch.conv2d(
            mul__180,
            weight_149,
            l_self_modules_stages_modules_2_modules_23_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__180 = (
            weight_149
        ) = (
            l_self_modules_stages_modules_2_modules_23_modules_conv2b_parameters_bias_
        ) = None
        gelu_146 = torch._C._nn.gelu(out_248)
        out_248 = None
        mul__181 = gelu_146.mul_(1.7015043497085571)
        gelu_146 = None
        reshape_150 = l_self_modules_stages_modules_2_modules_23_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_291 = (
            l_self_modules_stages_modules_2_modules_23_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_23_modules_conv3_parameters_gain_ = None
        view_150 = mul_291.view(-1)
        mul_291 = None
        batch_norm_150 = torch.nn.functional.batch_norm(
            reshape_150,
            None,
            None,
            weight=view_150,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_150 = view_150 = None
        weight_150 = batch_norm_150.reshape_as(
            l_self_modules_stages_modules_2_modules_23_modules_conv3_parameters_weight_
        )
        batch_norm_150 = (
            l_self_modules_stages_modules_2_modules_23_modules_conv3_parameters_weight_
        ) = None
        out_249 = torch.conv2d(
            mul__181,
            weight_150,
            l_self_modules_stages_modules_2_modules_23_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__181 = (
            weight_150
        ) = (
            l_self_modules_stages_modules_2_modules_23_modules_conv3_parameters_bias_
        ) = None
        x_se_140 = out_249.mean((2, 3), keepdim=True)
        x_se_141 = torch.conv2d(
            x_se_140,
            l_self_modules_stages_modules_2_modules_23_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_23_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_140 = l_self_modules_stages_modules_2_modules_23_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_23_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_142 = torch.nn.functional.relu(x_se_141, inplace=True)
        x_se_141 = None
        x_se_143 = torch.conv2d(
            x_se_142,
            l_self_modules_stages_modules_2_modules_23_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_23_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_142 = l_self_modules_stages_modules_2_modules_23_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_23_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_35 = x_se_143.sigmoid()
        x_se_143 = None
        mul_292 = out_249 * sigmoid_35
        out_249 = sigmoid_35 = None
        out_250 = 2.0 * mul_292
        mul_292 = None
        mul__182 = out_250.mul_(
            l_self_modules_stages_modules_2_modules_23_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_23_parameters_skipinit_gain_ = (
            mul__182
        ) = None
        mul_294 = out_250 * 0.2
        out_250 = None
        out_251 = mul_294 + out_244
        mul_294 = out_244 = None
        gelu_147 = torch._C._nn.gelu(out_251)
        out_251 = None
        mul__183 = gelu_147.mul_(1.7015043497085571)
        gelu_147 = None
        out_252 = mul__183 * 0.7142857142857141
        mul__183 = None
        avg_pool2d_2 = torch._C._nn.avg_pool2d(out_252, 2, 2, 0, True, False, None)
        reshape_151 = l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_296 = (
            l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_gain_ = (
            None
        )
        view_151 = mul_296.view(-1)
        mul_296 = None
        batch_norm_151 = torch.nn.functional.batch_norm(
            reshape_151,
            None,
            None,
            weight=view_151,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_151 = view_151 = None
        weight_151 = batch_norm_151.reshape_as(
            l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_weight_
        )
        batch_norm_151 = l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        shortcut_3 = torch.conv2d(
            avg_pool2d_2,
            weight_151,
            l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        avg_pool2d_2 = (
            weight_151
        ) = l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_bias_ = (None)
        reshape_152 = l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_297 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_gain_ = None
        view_152 = mul_297.view(-1)
        mul_297 = None
        batch_norm_152 = torch.nn.functional.batch_norm(
            reshape_152,
            None,
            None,
            weight=view_152,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_152 = view_152 = None
        weight_152 = batch_norm_152.reshape_as(
            l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_weight_
        )
        batch_norm_152 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_weight_
        ) = None
        out_253 = torch.conv2d(
            out_252,
            weight_152,
            l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_252 = (
            weight_152
        ) = (
            l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_bias_
        ) = None
        gelu_148 = torch._C._nn.gelu(out_253)
        out_253 = None
        mul__184 = gelu_148.mul_(1.7015043497085571)
        gelu_148 = None
        x_4 = torch._C._nn.pad(mul__184, (0, 1, 0, 1), "constant", 0)
        mul__184 = None
        reshape_153 = l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_298 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_gain_ = None
        view_153 = mul_298.view(-1)
        mul_298 = None
        batch_norm_153 = torch.nn.functional.batch_norm(
            reshape_153,
            None,
            None,
            weight=view_153,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_153 = view_153 = None
        weight_153 = batch_norm_153.reshape_as(
            l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_weight_
        )
        batch_norm_153 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_weight_
        ) = None
        out_254 = torch.conv2d(
            x_4,
            weight_153,
            l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            6,
        )
        x_4 = (
            weight_153
        ) = (
            l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_bias_
        ) = None
        gelu_149 = torch._C._nn.gelu(out_254)
        out_254 = None
        mul__185 = gelu_149.mul_(1.7015043497085571)
        gelu_149 = None
        reshape_154 = l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_299 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_gain_ = None
        view_154 = mul_299.view(-1)
        mul_299 = None
        batch_norm_154 = torch.nn.functional.batch_norm(
            reshape_154,
            None,
            None,
            weight=view_154,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_154 = view_154 = None
        weight_154 = batch_norm_154.reshape_as(
            l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_weight_
        )
        batch_norm_154 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_weight_
        ) = None
        out_255 = torch.conv2d(
            mul__185,
            weight_154,
            l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__185 = (
            weight_154
        ) = (
            l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_bias_
        ) = None
        gelu_150 = torch._C._nn.gelu(out_255)
        out_255 = None
        mul__186 = gelu_150.mul_(1.7015043497085571)
        gelu_150 = None
        reshape_155 = l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_300 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_gain_ = None
        view_155 = mul_300.view(-1)
        mul_300 = None
        batch_norm_155 = torch.nn.functional.batch_norm(
            reshape_155,
            None,
            None,
            weight=view_155,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_155 = view_155 = None
        weight_155 = batch_norm_155.reshape_as(
            l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_weight_
        )
        batch_norm_155 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_weight_
        ) = None
        out_256 = torch.conv2d(
            mul__186,
            weight_155,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__186 = (
            weight_155
        ) = (
            l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_bias_
        ) = None
        x_se_144 = out_256.mean((2, 3), keepdim=True)
        x_se_145 = torch.conv2d(
            x_se_144,
            l_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_144 = l_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_146 = torch.nn.functional.relu(x_se_145, inplace=True)
        x_se_145 = None
        x_se_147 = torch.conv2d(
            x_se_146,
            l_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_146 = l_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_36 = x_se_147.sigmoid()
        x_se_147 = None
        mul_301 = out_256 * sigmoid_36
        out_256 = sigmoid_36 = None
        out_257 = 2.0 * mul_301
        mul_301 = None
        mul__187 = out_257.mul_(
            l_self_modules_stages_modules_3_modules_0_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_3_modules_0_parameters_skipinit_gain_ = (
            mul__187
        ) = None
        mul_303 = out_257 * 0.2
        out_257 = None
        out_258 = mul_303 + shortcut_3
        mul_303 = shortcut_3 = None
        gelu_151 = torch._C._nn.gelu(out_258)
        mul__188 = gelu_151.mul_(1.7015043497085571)
        gelu_151 = None
        out_259 = mul__188 * 0.9805806756909201
        mul__188 = None
        reshape_156 = l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_305 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_gain_ = None
        view_156 = mul_305.view(-1)
        mul_305 = None
        batch_norm_156 = torch.nn.functional.batch_norm(
            reshape_156,
            None,
            None,
            weight=view_156,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_156 = view_156 = None
        weight_156 = batch_norm_156.reshape_as(
            l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_weight_
        )
        batch_norm_156 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_weight_
        ) = None
        out_260 = torch.conv2d(
            out_259,
            weight_156,
            l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_259 = (
            weight_156
        ) = (
            l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_bias_
        ) = None
        gelu_152 = torch._C._nn.gelu(out_260)
        out_260 = None
        mul__189 = gelu_152.mul_(1.7015043497085571)
        gelu_152 = None
        reshape_157 = l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_306 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_gain_ = None
        view_157 = mul_306.view(-1)
        mul_306 = None
        batch_norm_157 = torch.nn.functional.batch_norm(
            reshape_157,
            None,
            None,
            weight=view_157,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_157 = view_157 = None
        weight_157 = batch_norm_157.reshape_as(
            l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_weight_
        )
        batch_norm_157 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_weight_
        ) = None
        out_261 = torch.conv2d(
            mul__189,
            weight_157,
            l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__189 = (
            weight_157
        ) = (
            l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_bias_
        ) = None
        gelu_153 = torch._C._nn.gelu(out_261)
        out_261 = None
        mul__190 = gelu_153.mul_(1.7015043497085571)
        gelu_153 = None
        reshape_158 = l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_307 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_gain_ = None
        view_158 = mul_307.view(-1)
        mul_307 = None
        batch_norm_158 = torch.nn.functional.batch_norm(
            reshape_158,
            None,
            None,
            weight=view_158,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_158 = view_158 = None
        weight_158 = batch_norm_158.reshape_as(
            l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_weight_
        )
        batch_norm_158 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_weight_
        ) = None
        out_262 = torch.conv2d(
            mul__190,
            weight_158,
            l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__190 = (
            weight_158
        ) = (
            l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_bias_
        ) = None
        gelu_154 = torch._C._nn.gelu(out_262)
        out_262 = None
        mul__191 = gelu_154.mul_(1.7015043497085571)
        gelu_154 = None
        reshape_159 = l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_308 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_gain_ = None
        view_159 = mul_308.view(-1)
        mul_308 = None
        batch_norm_159 = torch.nn.functional.batch_norm(
            reshape_159,
            None,
            None,
            weight=view_159,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_159 = view_159 = None
        weight_159 = batch_norm_159.reshape_as(
            l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_weight_
        )
        batch_norm_159 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_weight_
        ) = None
        out_263 = torch.conv2d(
            mul__191,
            weight_159,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__191 = (
            weight_159
        ) = (
            l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_bias_
        ) = None
        x_se_148 = out_263.mean((2, 3), keepdim=True)
        x_se_149 = torch.conv2d(
            x_se_148,
            l_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_148 = l_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_150 = torch.nn.functional.relu(x_se_149, inplace=True)
        x_se_149 = None
        x_se_151 = torch.conv2d(
            x_se_150,
            l_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_150 = l_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_37 = x_se_151.sigmoid()
        x_se_151 = None
        mul_309 = out_263 * sigmoid_37
        out_263 = sigmoid_37 = None
        out_264 = 2.0 * mul_309
        mul_309 = None
        mul__192 = out_264.mul_(
            l_self_modules_stages_modules_3_modules_1_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_3_modules_1_parameters_skipinit_gain_ = (
            mul__192
        ) = None
        mul_311 = out_264 * 0.2
        out_264 = None
        out_265 = mul_311 + out_258
        mul_311 = out_258 = None
        gelu_155 = torch._C._nn.gelu(out_265)
        mul__193 = gelu_155.mul_(1.7015043497085571)
        gelu_155 = None
        out_266 = mul__193 * 0.9622504486493761
        mul__193 = None
        reshape_160 = l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_313 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_gain_ = None
        view_160 = mul_313.view(-1)
        mul_313 = None
        batch_norm_160 = torch.nn.functional.batch_norm(
            reshape_160,
            None,
            None,
            weight=view_160,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_160 = view_160 = None
        weight_160 = batch_norm_160.reshape_as(
            l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_weight_
        )
        batch_norm_160 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_weight_
        ) = None
        out_267 = torch.conv2d(
            out_266,
            weight_160,
            l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_266 = (
            weight_160
        ) = (
            l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_bias_
        ) = None
        gelu_156 = torch._C._nn.gelu(out_267)
        out_267 = None
        mul__194 = gelu_156.mul_(1.7015043497085571)
        gelu_156 = None
        reshape_161 = l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_314 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_gain_ = None
        view_161 = mul_314.view(-1)
        mul_314 = None
        batch_norm_161 = torch.nn.functional.batch_norm(
            reshape_161,
            None,
            None,
            weight=view_161,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_161 = view_161 = None
        weight_161 = batch_norm_161.reshape_as(
            l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_weight_
        )
        batch_norm_161 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_weight_
        ) = None
        out_268 = torch.conv2d(
            mul__194,
            weight_161,
            l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__194 = (
            weight_161
        ) = (
            l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_bias_
        ) = None
        gelu_157 = torch._C._nn.gelu(out_268)
        out_268 = None
        mul__195 = gelu_157.mul_(1.7015043497085571)
        gelu_157 = None
        reshape_162 = l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_315 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_gain_ = None
        view_162 = mul_315.view(-1)
        mul_315 = None
        batch_norm_162 = torch.nn.functional.batch_norm(
            reshape_162,
            None,
            None,
            weight=view_162,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_162 = view_162 = None
        weight_162 = batch_norm_162.reshape_as(
            l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_weight_
        )
        batch_norm_162 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_weight_
        ) = None
        out_269 = torch.conv2d(
            mul__195,
            weight_162,
            l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__195 = (
            weight_162
        ) = (
            l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_bias_
        ) = None
        gelu_158 = torch._C._nn.gelu(out_269)
        out_269 = None
        mul__196 = gelu_158.mul_(1.7015043497085571)
        gelu_158 = None
        reshape_163 = l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_316 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_gain_ = None
        view_163 = mul_316.view(-1)
        mul_316 = None
        batch_norm_163 = torch.nn.functional.batch_norm(
            reshape_163,
            None,
            None,
            weight=view_163,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_163 = view_163 = None
        weight_163 = batch_norm_163.reshape_as(
            l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_weight_
        )
        batch_norm_163 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_weight_
        ) = None
        out_270 = torch.conv2d(
            mul__196,
            weight_163,
            l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__196 = (
            weight_163
        ) = (
            l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_bias_
        ) = None
        x_se_152 = out_270.mean((2, 3), keepdim=True)
        x_se_153 = torch.conv2d(
            x_se_152,
            l_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_152 = l_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_154 = torch.nn.functional.relu(x_se_153, inplace=True)
        x_se_153 = None
        x_se_155 = torch.conv2d(
            x_se_154,
            l_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_154 = l_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_38 = x_se_155.sigmoid()
        x_se_155 = None
        mul_317 = out_270 * sigmoid_38
        out_270 = sigmoid_38 = None
        out_271 = 2.0 * mul_317
        mul_317 = None
        mul__197 = out_271.mul_(
            l_self_modules_stages_modules_3_modules_2_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_3_modules_2_parameters_skipinit_gain_ = (
            mul__197
        ) = None
        mul_319 = out_271 * 0.2
        out_271 = None
        out_272 = mul_319 + out_265
        mul_319 = out_265 = None
        gelu_159 = torch._C._nn.gelu(out_272)
        mul__198 = gelu_159.mul_(1.7015043497085571)
        gelu_159 = None
        out_273 = mul__198 * 0.9449111825230679
        mul__198 = None
        reshape_164 = l_self_modules_stages_modules_3_modules_3_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_321 = (
            l_self_modules_stages_modules_3_modules_3_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_3_modules_3_modules_conv1_parameters_gain_ = None
        view_164 = mul_321.view(-1)
        mul_321 = None
        batch_norm_164 = torch.nn.functional.batch_norm(
            reshape_164,
            None,
            None,
            weight=view_164,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_164 = view_164 = None
        weight_164 = batch_norm_164.reshape_as(
            l_self_modules_stages_modules_3_modules_3_modules_conv1_parameters_weight_
        )
        batch_norm_164 = (
            l_self_modules_stages_modules_3_modules_3_modules_conv1_parameters_weight_
        ) = None
        out_274 = torch.conv2d(
            out_273,
            weight_164,
            l_self_modules_stages_modules_3_modules_3_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_273 = (
            weight_164
        ) = (
            l_self_modules_stages_modules_3_modules_3_modules_conv1_parameters_bias_
        ) = None
        gelu_160 = torch._C._nn.gelu(out_274)
        out_274 = None
        mul__199 = gelu_160.mul_(1.7015043497085571)
        gelu_160 = None
        reshape_165 = l_self_modules_stages_modules_3_modules_3_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_322 = (
            l_self_modules_stages_modules_3_modules_3_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_3_modules_conv2_parameters_gain_ = None
        view_165 = mul_322.view(-1)
        mul_322 = None
        batch_norm_165 = torch.nn.functional.batch_norm(
            reshape_165,
            None,
            None,
            weight=view_165,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_165 = view_165 = None
        weight_165 = batch_norm_165.reshape_as(
            l_self_modules_stages_modules_3_modules_3_modules_conv2_parameters_weight_
        )
        batch_norm_165 = (
            l_self_modules_stages_modules_3_modules_3_modules_conv2_parameters_weight_
        ) = None
        out_275 = torch.conv2d(
            mul__199,
            weight_165,
            l_self_modules_stages_modules_3_modules_3_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__199 = (
            weight_165
        ) = (
            l_self_modules_stages_modules_3_modules_3_modules_conv2_parameters_bias_
        ) = None
        gelu_161 = torch._C._nn.gelu(out_275)
        out_275 = None
        mul__200 = gelu_161.mul_(1.7015043497085571)
        gelu_161 = None
        reshape_166 = l_self_modules_stages_modules_3_modules_3_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_323 = (
            l_self_modules_stages_modules_3_modules_3_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_3_modules_conv2b_parameters_gain_ = None
        view_166 = mul_323.view(-1)
        mul_323 = None
        batch_norm_166 = torch.nn.functional.batch_norm(
            reshape_166,
            None,
            None,
            weight=view_166,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_166 = view_166 = None
        weight_166 = batch_norm_166.reshape_as(
            l_self_modules_stages_modules_3_modules_3_modules_conv2b_parameters_weight_
        )
        batch_norm_166 = (
            l_self_modules_stages_modules_3_modules_3_modules_conv2b_parameters_weight_
        ) = None
        out_276 = torch.conv2d(
            mul__200,
            weight_166,
            l_self_modules_stages_modules_3_modules_3_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__200 = (
            weight_166
        ) = (
            l_self_modules_stages_modules_3_modules_3_modules_conv2b_parameters_bias_
        ) = None
        gelu_162 = torch._C._nn.gelu(out_276)
        out_276 = None
        mul__201 = gelu_162.mul_(1.7015043497085571)
        gelu_162 = None
        reshape_167 = l_self_modules_stages_modules_3_modules_3_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_324 = (
            l_self_modules_stages_modules_3_modules_3_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_3_modules_3_modules_conv3_parameters_gain_ = None
        view_167 = mul_324.view(-1)
        mul_324 = None
        batch_norm_167 = torch.nn.functional.batch_norm(
            reshape_167,
            None,
            None,
            weight=view_167,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_167 = view_167 = None
        weight_167 = batch_norm_167.reshape_as(
            l_self_modules_stages_modules_3_modules_3_modules_conv3_parameters_weight_
        )
        batch_norm_167 = (
            l_self_modules_stages_modules_3_modules_3_modules_conv3_parameters_weight_
        ) = None
        out_277 = torch.conv2d(
            mul__201,
            weight_167,
            l_self_modules_stages_modules_3_modules_3_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__201 = (
            weight_167
        ) = (
            l_self_modules_stages_modules_3_modules_3_modules_conv3_parameters_bias_
        ) = None
        x_se_156 = out_277.mean((2, 3), keepdim=True)
        x_se_157 = torch.conv2d(
            x_se_156,
            l_self_modules_stages_modules_3_modules_3_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_3_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_156 = l_self_modules_stages_modules_3_modules_3_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_3_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_158 = torch.nn.functional.relu(x_se_157, inplace=True)
        x_se_157 = None
        x_se_159 = torch.conv2d(
            x_se_158,
            l_self_modules_stages_modules_3_modules_3_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_3_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_158 = l_self_modules_stages_modules_3_modules_3_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_3_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_39 = x_se_159.sigmoid()
        x_se_159 = None
        mul_325 = out_277 * sigmoid_39
        out_277 = sigmoid_39 = None
        out_278 = 2.0 * mul_325
        mul_325 = None
        mul__202 = out_278.mul_(
            l_self_modules_stages_modules_3_modules_3_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_3_modules_3_parameters_skipinit_gain_ = (
            mul__202
        ) = None
        mul_327 = out_278 * 0.2
        out_278 = None
        out_279 = mul_327 + out_272
        mul_327 = out_272 = None
        gelu_163 = torch._C._nn.gelu(out_279)
        mul__203 = gelu_163.mul_(1.7015043497085571)
        gelu_163 = None
        out_280 = mul__203 * 0.9284766908852592
        mul__203 = None
        reshape_168 = l_self_modules_stages_modules_3_modules_4_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_329 = (
            l_self_modules_stages_modules_3_modules_4_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_3_modules_4_modules_conv1_parameters_gain_ = None
        view_168 = mul_329.view(-1)
        mul_329 = None
        batch_norm_168 = torch.nn.functional.batch_norm(
            reshape_168,
            None,
            None,
            weight=view_168,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_168 = view_168 = None
        weight_168 = batch_norm_168.reshape_as(
            l_self_modules_stages_modules_3_modules_4_modules_conv1_parameters_weight_
        )
        batch_norm_168 = (
            l_self_modules_stages_modules_3_modules_4_modules_conv1_parameters_weight_
        ) = None
        out_281 = torch.conv2d(
            out_280,
            weight_168,
            l_self_modules_stages_modules_3_modules_4_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_280 = (
            weight_168
        ) = (
            l_self_modules_stages_modules_3_modules_4_modules_conv1_parameters_bias_
        ) = None
        gelu_164 = torch._C._nn.gelu(out_281)
        out_281 = None
        mul__204 = gelu_164.mul_(1.7015043497085571)
        gelu_164 = None
        reshape_169 = l_self_modules_stages_modules_3_modules_4_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_330 = (
            l_self_modules_stages_modules_3_modules_4_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_4_modules_conv2_parameters_gain_ = None
        view_169 = mul_330.view(-1)
        mul_330 = None
        batch_norm_169 = torch.nn.functional.batch_norm(
            reshape_169,
            None,
            None,
            weight=view_169,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_169 = view_169 = None
        weight_169 = batch_norm_169.reshape_as(
            l_self_modules_stages_modules_3_modules_4_modules_conv2_parameters_weight_
        )
        batch_norm_169 = (
            l_self_modules_stages_modules_3_modules_4_modules_conv2_parameters_weight_
        ) = None
        out_282 = torch.conv2d(
            mul__204,
            weight_169,
            l_self_modules_stages_modules_3_modules_4_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__204 = (
            weight_169
        ) = (
            l_self_modules_stages_modules_3_modules_4_modules_conv2_parameters_bias_
        ) = None
        gelu_165 = torch._C._nn.gelu(out_282)
        out_282 = None
        mul__205 = gelu_165.mul_(1.7015043497085571)
        gelu_165 = None
        reshape_170 = l_self_modules_stages_modules_3_modules_4_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_331 = (
            l_self_modules_stages_modules_3_modules_4_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_4_modules_conv2b_parameters_gain_ = None
        view_170 = mul_331.view(-1)
        mul_331 = None
        batch_norm_170 = torch.nn.functional.batch_norm(
            reshape_170,
            None,
            None,
            weight=view_170,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_170 = view_170 = None
        weight_170 = batch_norm_170.reshape_as(
            l_self_modules_stages_modules_3_modules_4_modules_conv2b_parameters_weight_
        )
        batch_norm_170 = (
            l_self_modules_stages_modules_3_modules_4_modules_conv2b_parameters_weight_
        ) = None
        out_283 = torch.conv2d(
            mul__205,
            weight_170,
            l_self_modules_stages_modules_3_modules_4_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__205 = (
            weight_170
        ) = (
            l_self_modules_stages_modules_3_modules_4_modules_conv2b_parameters_bias_
        ) = None
        gelu_166 = torch._C._nn.gelu(out_283)
        out_283 = None
        mul__206 = gelu_166.mul_(1.7015043497085571)
        gelu_166 = None
        reshape_171 = l_self_modules_stages_modules_3_modules_4_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_332 = (
            l_self_modules_stages_modules_3_modules_4_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_3_modules_4_modules_conv3_parameters_gain_ = None
        view_171 = mul_332.view(-1)
        mul_332 = None
        batch_norm_171 = torch.nn.functional.batch_norm(
            reshape_171,
            None,
            None,
            weight=view_171,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_171 = view_171 = None
        weight_171 = batch_norm_171.reshape_as(
            l_self_modules_stages_modules_3_modules_4_modules_conv3_parameters_weight_
        )
        batch_norm_171 = (
            l_self_modules_stages_modules_3_modules_4_modules_conv3_parameters_weight_
        ) = None
        out_284 = torch.conv2d(
            mul__206,
            weight_171,
            l_self_modules_stages_modules_3_modules_4_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__206 = (
            weight_171
        ) = (
            l_self_modules_stages_modules_3_modules_4_modules_conv3_parameters_bias_
        ) = None
        x_se_160 = out_284.mean((2, 3), keepdim=True)
        x_se_161 = torch.conv2d(
            x_se_160,
            l_self_modules_stages_modules_3_modules_4_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_4_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_160 = l_self_modules_stages_modules_3_modules_4_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_4_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_162 = torch.nn.functional.relu(x_se_161, inplace=True)
        x_se_161 = None
        x_se_163 = torch.conv2d(
            x_se_162,
            l_self_modules_stages_modules_3_modules_4_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_4_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_162 = l_self_modules_stages_modules_3_modules_4_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_4_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_40 = x_se_163.sigmoid()
        x_se_163 = None
        mul_333 = out_284 * sigmoid_40
        out_284 = sigmoid_40 = None
        out_285 = 2.0 * mul_333
        mul_333 = None
        mul__207 = out_285.mul_(
            l_self_modules_stages_modules_3_modules_4_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_3_modules_4_parameters_skipinit_gain_ = (
            mul__207
        ) = None
        mul_335 = out_285 * 0.2
        out_285 = None
        out_286 = mul_335 + out_279
        mul_335 = out_279 = None
        gelu_167 = torch._C._nn.gelu(out_286)
        mul__208 = gelu_167.mul_(1.7015043497085571)
        gelu_167 = None
        out_287 = mul__208 * 0.9128709291752768
        mul__208 = None
        reshape_172 = l_self_modules_stages_modules_3_modules_5_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_337 = (
            l_self_modules_stages_modules_3_modules_5_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_3_modules_5_modules_conv1_parameters_gain_ = None
        view_172 = mul_337.view(-1)
        mul_337 = None
        batch_norm_172 = torch.nn.functional.batch_norm(
            reshape_172,
            None,
            None,
            weight=view_172,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_172 = view_172 = None
        weight_172 = batch_norm_172.reshape_as(
            l_self_modules_stages_modules_3_modules_5_modules_conv1_parameters_weight_
        )
        batch_norm_172 = (
            l_self_modules_stages_modules_3_modules_5_modules_conv1_parameters_weight_
        ) = None
        out_288 = torch.conv2d(
            out_287,
            weight_172,
            l_self_modules_stages_modules_3_modules_5_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_287 = (
            weight_172
        ) = (
            l_self_modules_stages_modules_3_modules_5_modules_conv1_parameters_bias_
        ) = None
        gelu_168 = torch._C._nn.gelu(out_288)
        out_288 = None
        mul__209 = gelu_168.mul_(1.7015043497085571)
        gelu_168 = None
        reshape_173 = l_self_modules_stages_modules_3_modules_5_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_338 = (
            l_self_modules_stages_modules_3_modules_5_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_5_modules_conv2_parameters_gain_ = None
        view_173 = mul_338.view(-1)
        mul_338 = None
        batch_norm_173 = torch.nn.functional.batch_norm(
            reshape_173,
            None,
            None,
            weight=view_173,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_173 = view_173 = None
        weight_173 = batch_norm_173.reshape_as(
            l_self_modules_stages_modules_3_modules_5_modules_conv2_parameters_weight_
        )
        batch_norm_173 = (
            l_self_modules_stages_modules_3_modules_5_modules_conv2_parameters_weight_
        ) = None
        out_289 = torch.conv2d(
            mul__209,
            weight_173,
            l_self_modules_stages_modules_3_modules_5_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__209 = (
            weight_173
        ) = (
            l_self_modules_stages_modules_3_modules_5_modules_conv2_parameters_bias_
        ) = None
        gelu_169 = torch._C._nn.gelu(out_289)
        out_289 = None
        mul__210 = gelu_169.mul_(1.7015043497085571)
        gelu_169 = None
        reshape_174 = l_self_modules_stages_modules_3_modules_5_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_339 = (
            l_self_modules_stages_modules_3_modules_5_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_5_modules_conv2b_parameters_gain_ = None
        view_174 = mul_339.view(-1)
        mul_339 = None
        batch_norm_174 = torch.nn.functional.batch_norm(
            reshape_174,
            None,
            None,
            weight=view_174,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_174 = view_174 = None
        weight_174 = batch_norm_174.reshape_as(
            l_self_modules_stages_modules_3_modules_5_modules_conv2b_parameters_weight_
        )
        batch_norm_174 = (
            l_self_modules_stages_modules_3_modules_5_modules_conv2b_parameters_weight_
        ) = None
        out_290 = torch.conv2d(
            mul__210,
            weight_174,
            l_self_modules_stages_modules_3_modules_5_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__210 = (
            weight_174
        ) = (
            l_self_modules_stages_modules_3_modules_5_modules_conv2b_parameters_bias_
        ) = None
        gelu_170 = torch._C._nn.gelu(out_290)
        out_290 = None
        mul__211 = gelu_170.mul_(1.7015043497085571)
        gelu_170 = None
        reshape_175 = l_self_modules_stages_modules_3_modules_5_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_340 = (
            l_self_modules_stages_modules_3_modules_5_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_3_modules_5_modules_conv3_parameters_gain_ = None
        view_175 = mul_340.view(-1)
        mul_340 = None
        batch_norm_175 = torch.nn.functional.batch_norm(
            reshape_175,
            None,
            None,
            weight=view_175,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_175 = view_175 = None
        weight_175 = batch_norm_175.reshape_as(
            l_self_modules_stages_modules_3_modules_5_modules_conv3_parameters_weight_
        )
        batch_norm_175 = (
            l_self_modules_stages_modules_3_modules_5_modules_conv3_parameters_weight_
        ) = None
        out_291 = torch.conv2d(
            mul__211,
            weight_175,
            l_self_modules_stages_modules_3_modules_5_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__211 = (
            weight_175
        ) = (
            l_self_modules_stages_modules_3_modules_5_modules_conv3_parameters_bias_
        ) = None
        x_se_164 = out_291.mean((2, 3), keepdim=True)
        x_se_165 = torch.conv2d(
            x_se_164,
            l_self_modules_stages_modules_3_modules_5_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_5_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_164 = l_self_modules_stages_modules_3_modules_5_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_5_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_166 = torch.nn.functional.relu(x_se_165, inplace=True)
        x_se_165 = None
        x_se_167 = torch.conv2d(
            x_se_166,
            l_self_modules_stages_modules_3_modules_5_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_5_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_166 = l_self_modules_stages_modules_3_modules_5_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_5_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_41 = x_se_167.sigmoid()
        x_se_167 = None
        mul_341 = out_291 * sigmoid_41
        out_291 = sigmoid_41 = None
        out_292 = 2.0 * mul_341
        mul_341 = None
        mul__212 = out_292.mul_(
            l_self_modules_stages_modules_3_modules_5_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_3_modules_5_parameters_skipinit_gain_ = (
            mul__212
        ) = None
        mul_343 = out_292 * 0.2
        out_292 = None
        out_293 = mul_343 + out_286
        mul_343 = out_286 = None
        gelu_171 = torch._C._nn.gelu(out_293)
        mul__213 = gelu_171.mul_(1.7015043497085571)
        gelu_171 = None
        out_294 = mul__213 * 0.8980265101338745
        mul__213 = None
        reshape_176 = l_self_modules_stages_modules_3_modules_6_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_345 = (
            l_self_modules_stages_modules_3_modules_6_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_3_modules_6_modules_conv1_parameters_gain_ = None
        view_176 = mul_345.view(-1)
        mul_345 = None
        batch_norm_176 = torch.nn.functional.batch_norm(
            reshape_176,
            None,
            None,
            weight=view_176,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_176 = view_176 = None
        weight_176 = batch_norm_176.reshape_as(
            l_self_modules_stages_modules_3_modules_6_modules_conv1_parameters_weight_
        )
        batch_norm_176 = (
            l_self_modules_stages_modules_3_modules_6_modules_conv1_parameters_weight_
        ) = None
        out_295 = torch.conv2d(
            out_294,
            weight_176,
            l_self_modules_stages_modules_3_modules_6_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_294 = (
            weight_176
        ) = (
            l_self_modules_stages_modules_3_modules_6_modules_conv1_parameters_bias_
        ) = None
        gelu_172 = torch._C._nn.gelu(out_295)
        out_295 = None
        mul__214 = gelu_172.mul_(1.7015043497085571)
        gelu_172 = None
        reshape_177 = l_self_modules_stages_modules_3_modules_6_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_346 = (
            l_self_modules_stages_modules_3_modules_6_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_6_modules_conv2_parameters_gain_ = None
        view_177 = mul_346.view(-1)
        mul_346 = None
        batch_norm_177 = torch.nn.functional.batch_norm(
            reshape_177,
            None,
            None,
            weight=view_177,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_177 = view_177 = None
        weight_177 = batch_norm_177.reshape_as(
            l_self_modules_stages_modules_3_modules_6_modules_conv2_parameters_weight_
        )
        batch_norm_177 = (
            l_self_modules_stages_modules_3_modules_6_modules_conv2_parameters_weight_
        ) = None
        out_296 = torch.conv2d(
            mul__214,
            weight_177,
            l_self_modules_stages_modules_3_modules_6_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__214 = (
            weight_177
        ) = (
            l_self_modules_stages_modules_3_modules_6_modules_conv2_parameters_bias_
        ) = None
        gelu_173 = torch._C._nn.gelu(out_296)
        out_296 = None
        mul__215 = gelu_173.mul_(1.7015043497085571)
        gelu_173 = None
        reshape_178 = l_self_modules_stages_modules_3_modules_6_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_347 = (
            l_self_modules_stages_modules_3_modules_6_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_6_modules_conv2b_parameters_gain_ = None
        view_178 = mul_347.view(-1)
        mul_347 = None
        batch_norm_178 = torch.nn.functional.batch_norm(
            reshape_178,
            None,
            None,
            weight=view_178,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_178 = view_178 = None
        weight_178 = batch_norm_178.reshape_as(
            l_self_modules_stages_modules_3_modules_6_modules_conv2b_parameters_weight_
        )
        batch_norm_178 = (
            l_self_modules_stages_modules_3_modules_6_modules_conv2b_parameters_weight_
        ) = None
        out_297 = torch.conv2d(
            mul__215,
            weight_178,
            l_self_modules_stages_modules_3_modules_6_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__215 = (
            weight_178
        ) = (
            l_self_modules_stages_modules_3_modules_6_modules_conv2b_parameters_bias_
        ) = None
        gelu_174 = torch._C._nn.gelu(out_297)
        out_297 = None
        mul__216 = gelu_174.mul_(1.7015043497085571)
        gelu_174 = None
        reshape_179 = l_self_modules_stages_modules_3_modules_6_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_348 = (
            l_self_modules_stages_modules_3_modules_6_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_3_modules_6_modules_conv3_parameters_gain_ = None
        view_179 = mul_348.view(-1)
        mul_348 = None
        batch_norm_179 = torch.nn.functional.batch_norm(
            reshape_179,
            None,
            None,
            weight=view_179,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_179 = view_179 = None
        weight_179 = batch_norm_179.reshape_as(
            l_self_modules_stages_modules_3_modules_6_modules_conv3_parameters_weight_
        )
        batch_norm_179 = (
            l_self_modules_stages_modules_3_modules_6_modules_conv3_parameters_weight_
        ) = None
        out_298 = torch.conv2d(
            mul__216,
            weight_179,
            l_self_modules_stages_modules_3_modules_6_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__216 = (
            weight_179
        ) = (
            l_self_modules_stages_modules_3_modules_6_modules_conv3_parameters_bias_
        ) = None
        x_se_168 = out_298.mean((2, 3), keepdim=True)
        x_se_169 = torch.conv2d(
            x_se_168,
            l_self_modules_stages_modules_3_modules_6_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_6_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_168 = l_self_modules_stages_modules_3_modules_6_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_6_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_170 = torch.nn.functional.relu(x_se_169, inplace=True)
        x_se_169 = None
        x_se_171 = torch.conv2d(
            x_se_170,
            l_self_modules_stages_modules_3_modules_6_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_6_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_170 = l_self_modules_stages_modules_3_modules_6_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_6_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_42 = x_se_171.sigmoid()
        x_se_171 = None
        mul_349 = out_298 * sigmoid_42
        out_298 = sigmoid_42 = None
        out_299 = 2.0 * mul_349
        mul_349 = None
        mul__217 = out_299.mul_(
            l_self_modules_stages_modules_3_modules_6_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_3_modules_6_parameters_skipinit_gain_ = (
            mul__217
        ) = None
        mul_351 = out_299 * 0.2
        out_299 = None
        out_300 = mul_351 + out_293
        mul_351 = out_293 = None
        gelu_175 = torch._C._nn.gelu(out_300)
        mul__218 = gelu_175.mul_(1.7015043497085571)
        gelu_175 = None
        out_301 = mul__218 * 0.8838834764831842
        mul__218 = None
        reshape_180 = l_self_modules_stages_modules_3_modules_7_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_353 = (
            l_self_modules_stages_modules_3_modules_7_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_3_modules_7_modules_conv1_parameters_gain_ = None
        view_180 = mul_353.view(-1)
        mul_353 = None
        batch_norm_180 = torch.nn.functional.batch_norm(
            reshape_180,
            None,
            None,
            weight=view_180,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_180 = view_180 = None
        weight_180 = batch_norm_180.reshape_as(
            l_self_modules_stages_modules_3_modules_7_modules_conv1_parameters_weight_
        )
        batch_norm_180 = (
            l_self_modules_stages_modules_3_modules_7_modules_conv1_parameters_weight_
        ) = None
        out_302 = torch.conv2d(
            out_301,
            weight_180,
            l_self_modules_stages_modules_3_modules_7_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_301 = (
            weight_180
        ) = (
            l_self_modules_stages_modules_3_modules_7_modules_conv1_parameters_bias_
        ) = None
        gelu_176 = torch._C._nn.gelu(out_302)
        out_302 = None
        mul__219 = gelu_176.mul_(1.7015043497085571)
        gelu_176 = None
        reshape_181 = l_self_modules_stages_modules_3_modules_7_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_354 = (
            l_self_modules_stages_modules_3_modules_7_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_7_modules_conv2_parameters_gain_ = None
        view_181 = mul_354.view(-1)
        mul_354 = None
        batch_norm_181 = torch.nn.functional.batch_norm(
            reshape_181,
            None,
            None,
            weight=view_181,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_181 = view_181 = None
        weight_181 = batch_norm_181.reshape_as(
            l_self_modules_stages_modules_3_modules_7_modules_conv2_parameters_weight_
        )
        batch_norm_181 = (
            l_self_modules_stages_modules_3_modules_7_modules_conv2_parameters_weight_
        ) = None
        out_303 = torch.conv2d(
            mul__219,
            weight_181,
            l_self_modules_stages_modules_3_modules_7_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__219 = (
            weight_181
        ) = (
            l_self_modules_stages_modules_3_modules_7_modules_conv2_parameters_bias_
        ) = None
        gelu_177 = torch._C._nn.gelu(out_303)
        out_303 = None
        mul__220 = gelu_177.mul_(1.7015043497085571)
        gelu_177 = None
        reshape_182 = l_self_modules_stages_modules_3_modules_7_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_355 = (
            l_self_modules_stages_modules_3_modules_7_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_7_modules_conv2b_parameters_gain_ = None
        view_182 = mul_355.view(-1)
        mul_355 = None
        batch_norm_182 = torch.nn.functional.batch_norm(
            reshape_182,
            None,
            None,
            weight=view_182,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_182 = view_182 = None
        weight_182 = batch_norm_182.reshape_as(
            l_self_modules_stages_modules_3_modules_7_modules_conv2b_parameters_weight_
        )
        batch_norm_182 = (
            l_self_modules_stages_modules_3_modules_7_modules_conv2b_parameters_weight_
        ) = None
        out_304 = torch.conv2d(
            mul__220,
            weight_182,
            l_self_modules_stages_modules_3_modules_7_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__220 = (
            weight_182
        ) = (
            l_self_modules_stages_modules_3_modules_7_modules_conv2b_parameters_bias_
        ) = None
        gelu_178 = torch._C._nn.gelu(out_304)
        out_304 = None
        mul__221 = gelu_178.mul_(1.7015043497085571)
        gelu_178 = None
        reshape_183 = l_self_modules_stages_modules_3_modules_7_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_356 = (
            l_self_modules_stages_modules_3_modules_7_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_3_modules_7_modules_conv3_parameters_gain_ = None
        view_183 = mul_356.view(-1)
        mul_356 = None
        batch_norm_183 = torch.nn.functional.batch_norm(
            reshape_183,
            None,
            None,
            weight=view_183,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_183 = view_183 = None
        weight_183 = batch_norm_183.reshape_as(
            l_self_modules_stages_modules_3_modules_7_modules_conv3_parameters_weight_
        )
        batch_norm_183 = (
            l_self_modules_stages_modules_3_modules_7_modules_conv3_parameters_weight_
        ) = None
        out_305 = torch.conv2d(
            mul__221,
            weight_183,
            l_self_modules_stages_modules_3_modules_7_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__221 = (
            weight_183
        ) = (
            l_self_modules_stages_modules_3_modules_7_modules_conv3_parameters_bias_
        ) = None
        x_se_172 = out_305.mean((2, 3), keepdim=True)
        x_se_173 = torch.conv2d(
            x_se_172,
            l_self_modules_stages_modules_3_modules_7_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_7_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_172 = l_self_modules_stages_modules_3_modules_7_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_7_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_174 = torch.nn.functional.relu(x_se_173, inplace=True)
        x_se_173 = None
        x_se_175 = torch.conv2d(
            x_se_174,
            l_self_modules_stages_modules_3_modules_7_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_7_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_174 = l_self_modules_stages_modules_3_modules_7_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_7_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_43 = x_se_175.sigmoid()
        x_se_175 = None
        mul_357 = out_305 * sigmoid_43
        out_305 = sigmoid_43 = None
        out_306 = 2.0 * mul_357
        mul_357 = None
        mul__222 = out_306.mul_(
            l_self_modules_stages_modules_3_modules_7_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_3_modules_7_parameters_skipinit_gain_ = (
            mul__222
        ) = None
        mul_359 = out_306 * 0.2
        out_306 = None
        out_307 = mul_359 + out_300
        mul_359 = out_300 = None
        gelu_179 = torch._C._nn.gelu(out_307)
        mul__223 = gelu_179.mul_(1.7015043497085571)
        gelu_179 = None
        out_308 = mul__223 * 0.8703882797784891
        mul__223 = None
        reshape_184 = l_self_modules_stages_modules_3_modules_8_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_361 = (
            l_self_modules_stages_modules_3_modules_8_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_3_modules_8_modules_conv1_parameters_gain_ = None
        view_184 = mul_361.view(-1)
        mul_361 = None
        batch_norm_184 = torch.nn.functional.batch_norm(
            reshape_184,
            None,
            None,
            weight=view_184,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_184 = view_184 = None
        weight_184 = batch_norm_184.reshape_as(
            l_self_modules_stages_modules_3_modules_8_modules_conv1_parameters_weight_
        )
        batch_norm_184 = (
            l_self_modules_stages_modules_3_modules_8_modules_conv1_parameters_weight_
        ) = None
        out_309 = torch.conv2d(
            out_308,
            weight_184,
            l_self_modules_stages_modules_3_modules_8_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_308 = (
            weight_184
        ) = (
            l_self_modules_stages_modules_3_modules_8_modules_conv1_parameters_bias_
        ) = None
        gelu_180 = torch._C._nn.gelu(out_309)
        out_309 = None
        mul__224 = gelu_180.mul_(1.7015043497085571)
        gelu_180 = None
        reshape_185 = l_self_modules_stages_modules_3_modules_8_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_362 = (
            l_self_modules_stages_modules_3_modules_8_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_8_modules_conv2_parameters_gain_ = None
        view_185 = mul_362.view(-1)
        mul_362 = None
        batch_norm_185 = torch.nn.functional.batch_norm(
            reshape_185,
            None,
            None,
            weight=view_185,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_185 = view_185 = None
        weight_185 = batch_norm_185.reshape_as(
            l_self_modules_stages_modules_3_modules_8_modules_conv2_parameters_weight_
        )
        batch_norm_185 = (
            l_self_modules_stages_modules_3_modules_8_modules_conv2_parameters_weight_
        ) = None
        out_310 = torch.conv2d(
            mul__224,
            weight_185,
            l_self_modules_stages_modules_3_modules_8_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__224 = (
            weight_185
        ) = (
            l_self_modules_stages_modules_3_modules_8_modules_conv2_parameters_bias_
        ) = None
        gelu_181 = torch._C._nn.gelu(out_310)
        out_310 = None
        mul__225 = gelu_181.mul_(1.7015043497085571)
        gelu_181 = None
        reshape_186 = l_self_modules_stages_modules_3_modules_8_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_363 = (
            l_self_modules_stages_modules_3_modules_8_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_8_modules_conv2b_parameters_gain_ = None
        view_186 = mul_363.view(-1)
        mul_363 = None
        batch_norm_186 = torch.nn.functional.batch_norm(
            reshape_186,
            None,
            None,
            weight=view_186,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_186 = view_186 = None
        weight_186 = batch_norm_186.reshape_as(
            l_self_modules_stages_modules_3_modules_8_modules_conv2b_parameters_weight_
        )
        batch_norm_186 = (
            l_self_modules_stages_modules_3_modules_8_modules_conv2b_parameters_weight_
        ) = None
        out_311 = torch.conv2d(
            mul__225,
            weight_186,
            l_self_modules_stages_modules_3_modules_8_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__225 = (
            weight_186
        ) = (
            l_self_modules_stages_modules_3_modules_8_modules_conv2b_parameters_bias_
        ) = None
        gelu_182 = torch._C._nn.gelu(out_311)
        out_311 = None
        mul__226 = gelu_182.mul_(1.7015043497085571)
        gelu_182 = None
        reshape_187 = l_self_modules_stages_modules_3_modules_8_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_364 = (
            l_self_modules_stages_modules_3_modules_8_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_3_modules_8_modules_conv3_parameters_gain_ = None
        view_187 = mul_364.view(-1)
        mul_364 = None
        batch_norm_187 = torch.nn.functional.batch_norm(
            reshape_187,
            None,
            None,
            weight=view_187,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_187 = view_187 = None
        weight_187 = batch_norm_187.reshape_as(
            l_self_modules_stages_modules_3_modules_8_modules_conv3_parameters_weight_
        )
        batch_norm_187 = (
            l_self_modules_stages_modules_3_modules_8_modules_conv3_parameters_weight_
        ) = None
        out_312 = torch.conv2d(
            mul__226,
            weight_187,
            l_self_modules_stages_modules_3_modules_8_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__226 = (
            weight_187
        ) = (
            l_self_modules_stages_modules_3_modules_8_modules_conv3_parameters_bias_
        ) = None
        x_se_176 = out_312.mean((2, 3), keepdim=True)
        x_se_177 = torch.conv2d(
            x_se_176,
            l_self_modules_stages_modules_3_modules_8_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_8_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_176 = l_self_modules_stages_modules_3_modules_8_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_8_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_178 = torch.nn.functional.relu(x_se_177, inplace=True)
        x_se_177 = None
        x_se_179 = torch.conv2d(
            x_se_178,
            l_self_modules_stages_modules_3_modules_8_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_8_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_178 = l_self_modules_stages_modules_3_modules_8_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_8_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_44 = x_se_179.sigmoid()
        x_se_179 = None
        mul_365 = out_312 * sigmoid_44
        out_312 = sigmoid_44 = None
        out_313 = 2.0 * mul_365
        mul_365 = None
        mul__227 = out_313.mul_(
            l_self_modules_stages_modules_3_modules_8_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_3_modules_8_parameters_skipinit_gain_ = (
            mul__227
        ) = None
        mul_367 = out_313 * 0.2
        out_313 = None
        out_314 = mul_367 + out_307
        mul_367 = out_307 = None
        gelu_183 = torch._C._nn.gelu(out_314)
        mul__228 = gelu_183.mul_(1.7015043497085571)
        gelu_183 = None
        out_315 = mul__228 * 0.8574929257125441
        mul__228 = None
        reshape_188 = l_self_modules_stages_modules_3_modules_9_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_369 = (
            l_self_modules_stages_modules_3_modules_9_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_3_modules_9_modules_conv1_parameters_gain_ = None
        view_188 = mul_369.view(-1)
        mul_369 = None
        batch_norm_188 = torch.nn.functional.batch_norm(
            reshape_188,
            None,
            None,
            weight=view_188,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_188 = view_188 = None
        weight_188 = batch_norm_188.reshape_as(
            l_self_modules_stages_modules_3_modules_9_modules_conv1_parameters_weight_
        )
        batch_norm_188 = (
            l_self_modules_stages_modules_3_modules_9_modules_conv1_parameters_weight_
        ) = None
        out_316 = torch.conv2d(
            out_315,
            weight_188,
            l_self_modules_stages_modules_3_modules_9_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_315 = (
            weight_188
        ) = (
            l_self_modules_stages_modules_3_modules_9_modules_conv1_parameters_bias_
        ) = None
        gelu_184 = torch._C._nn.gelu(out_316)
        out_316 = None
        mul__229 = gelu_184.mul_(1.7015043497085571)
        gelu_184 = None
        reshape_189 = l_self_modules_stages_modules_3_modules_9_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_370 = (
            l_self_modules_stages_modules_3_modules_9_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_9_modules_conv2_parameters_gain_ = None
        view_189 = mul_370.view(-1)
        mul_370 = None
        batch_norm_189 = torch.nn.functional.batch_norm(
            reshape_189,
            None,
            None,
            weight=view_189,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_189 = view_189 = None
        weight_189 = batch_norm_189.reshape_as(
            l_self_modules_stages_modules_3_modules_9_modules_conv2_parameters_weight_
        )
        batch_norm_189 = (
            l_self_modules_stages_modules_3_modules_9_modules_conv2_parameters_weight_
        ) = None
        out_317 = torch.conv2d(
            mul__229,
            weight_189,
            l_self_modules_stages_modules_3_modules_9_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__229 = (
            weight_189
        ) = (
            l_self_modules_stages_modules_3_modules_9_modules_conv2_parameters_bias_
        ) = None
        gelu_185 = torch._C._nn.gelu(out_317)
        out_317 = None
        mul__230 = gelu_185.mul_(1.7015043497085571)
        gelu_185 = None
        reshape_190 = l_self_modules_stages_modules_3_modules_9_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_371 = (
            l_self_modules_stages_modules_3_modules_9_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_9_modules_conv2b_parameters_gain_ = None
        view_190 = mul_371.view(-1)
        mul_371 = None
        batch_norm_190 = torch.nn.functional.batch_norm(
            reshape_190,
            None,
            None,
            weight=view_190,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_190 = view_190 = None
        weight_190 = batch_norm_190.reshape_as(
            l_self_modules_stages_modules_3_modules_9_modules_conv2b_parameters_weight_
        )
        batch_norm_190 = (
            l_self_modules_stages_modules_3_modules_9_modules_conv2b_parameters_weight_
        ) = None
        out_318 = torch.conv2d(
            mul__230,
            weight_190,
            l_self_modules_stages_modules_3_modules_9_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__230 = (
            weight_190
        ) = (
            l_self_modules_stages_modules_3_modules_9_modules_conv2b_parameters_bias_
        ) = None
        gelu_186 = torch._C._nn.gelu(out_318)
        out_318 = None
        mul__231 = gelu_186.mul_(1.7015043497085571)
        gelu_186 = None
        reshape_191 = l_self_modules_stages_modules_3_modules_9_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_372 = (
            l_self_modules_stages_modules_3_modules_9_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_3_modules_9_modules_conv3_parameters_gain_ = None
        view_191 = mul_372.view(-1)
        mul_372 = None
        batch_norm_191 = torch.nn.functional.batch_norm(
            reshape_191,
            None,
            None,
            weight=view_191,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_191 = view_191 = None
        weight_191 = batch_norm_191.reshape_as(
            l_self_modules_stages_modules_3_modules_9_modules_conv3_parameters_weight_
        )
        batch_norm_191 = (
            l_self_modules_stages_modules_3_modules_9_modules_conv3_parameters_weight_
        ) = None
        out_319 = torch.conv2d(
            mul__231,
            weight_191,
            l_self_modules_stages_modules_3_modules_9_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__231 = (
            weight_191
        ) = (
            l_self_modules_stages_modules_3_modules_9_modules_conv3_parameters_bias_
        ) = None
        x_se_180 = out_319.mean((2, 3), keepdim=True)
        x_se_181 = torch.conv2d(
            x_se_180,
            l_self_modules_stages_modules_3_modules_9_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_9_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_180 = l_self_modules_stages_modules_3_modules_9_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_9_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_182 = torch.nn.functional.relu(x_se_181, inplace=True)
        x_se_181 = None
        x_se_183 = torch.conv2d(
            x_se_182,
            l_self_modules_stages_modules_3_modules_9_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_9_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_182 = l_self_modules_stages_modules_3_modules_9_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_9_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_45 = x_se_183.sigmoid()
        x_se_183 = None
        mul_373 = out_319 * sigmoid_45
        out_319 = sigmoid_45 = None
        out_320 = 2.0 * mul_373
        mul_373 = None
        mul__232 = out_320.mul_(
            l_self_modules_stages_modules_3_modules_9_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_3_modules_9_parameters_skipinit_gain_ = (
            mul__232
        ) = None
        mul_375 = out_320 * 0.2
        out_320 = None
        out_321 = mul_375 + out_314
        mul_375 = out_314 = None
        gelu_187 = torch._C._nn.gelu(out_321)
        mul__233 = gelu_187.mul_(1.7015043497085571)
        gelu_187 = None
        out_322 = mul__233 * 0.8451542547285165
        mul__233 = None
        reshape_192 = l_self_modules_stages_modules_3_modules_10_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_377 = (
            l_self_modules_stages_modules_3_modules_10_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_3_modules_10_modules_conv1_parameters_gain_ = None
        view_192 = mul_377.view(-1)
        mul_377 = None
        batch_norm_192 = torch.nn.functional.batch_norm(
            reshape_192,
            None,
            None,
            weight=view_192,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_192 = view_192 = None
        weight_192 = batch_norm_192.reshape_as(
            l_self_modules_stages_modules_3_modules_10_modules_conv1_parameters_weight_
        )
        batch_norm_192 = (
            l_self_modules_stages_modules_3_modules_10_modules_conv1_parameters_weight_
        ) = None
        out_323 = torch.conv2d(
            out_322,
            weight_192,
            l_self_modules_stages_modules_3_modules_10_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_322 = (
            weight_192
        ) = (
            l_self_modules_stages_modules_3_modules_10_modules_conv1_parameters_bias_
        ) = None
        gelu_188 = torch._C._nn.gelu(out_323)
        out_323 = None
        mul__234 = gelu_188.mul_(1.7015043497085571)
        gelu_188 = None
        reshape_193 = l_self_modules_stages_modules_3_modules_10_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_378 = (
            l_self_modules_stages_modules_3_modules_10_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_10_modules_conv2_parameters_gain_ = None
        view_193 = mul_378.view(-1)
        mul_378 = None
        batch_norm_193 = torch.nn.functional.batch_norm(
            reshape_193,
            None,
            None,
            weight=view_193,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_193 = view_193 = None
        weight_193 = batch_norm_193.reshape_as(
            l_self_modules_stages_modules_3_modules_10_modules_conv2_parameters_weight_
        )
        batch_norm_193 = (
            l_self_modules_stages_modules_3_modules_10_modules_conv2_parameters_weight_
        ) = None
        out_324 = torch.conv2d(
            mul__234,
            weight_193,
            l_self_modules_stages_modules_3_modules_10_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__234 = (
            weight_193
        ) = (
            l_self_modules_stages_modules_3_modules_10_modules_conv2_parameters_bias_
        ) = None
        gelu_189 = torch._C._nn.gelu(out_324)
        out_324 = None
        mul__235 = gelu_189.mul_(1.7015043497085571)
        gelu_189 = None
        reshape_194 = l_self_modules_stages_modules_3_modules_10_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_379 = (
            l_self_modules_stages_modules_3_modules_10_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_10_modules_conv2b_parameters_gain_ = (
            None
        )
        view_194 = mul_379.view(-1)
        mul_379 = None
        batch_norm_194 = torch.nn.functional.batch_norm(
            reshape_194,
            None,
            None,
            weight=view_194,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_194 = view_194 = None
        weight_194 = batch_norm_194.reshape_as(
            l_self_modules_stages_modules_3_modules_10_modules_conv2b_parameters_weight_
        )
        batch_norm_194 = (
            l_self_modules_stages_modules_3_modules_10_modules_conv2b_parameters_weight_
        ) = None
        out_325 = torch.conv2d(
            mul__235,
            weight_194,
            l_self_modules_stages_modules_3_modules_10_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__235 = (
            weight_194
        ) = (
            l_self_modules_stages_modules_3_modules_10_modules_conv2b_parameters_bias_
        ) = None
        gelu_190 = torch._C._nn.gelu(out_325)
        out_325 = None
        mul__236 = gelu_190.mul_(1.7015043497085571)
        gelu_190 = None
        reshape_195 = l_self_modules_stages_modules_3_modules_10_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_380 = (
            l_self_modules_stages_modules_3_modules_10_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_3_modules_10_modules_conv3_parameters_gain_ = None
        view_195 = mul_380.view(-1)
        mul_380 = None
        batch_norm_195 = torch.nn.functional.batch_norm(
            reshape_195,
            None,
            None,
            weight=view_195,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_195 = view_195 = None
        weight_195 = batch_norm_195.reshape_as(
            l_self_modules_stages_modules_3_modules_10_modules_conv3_parameters_weight_
        )
        batch_norm_195 = (
            l_self_modules_stages_modules_3_modules_10_modules_conv3_parameters_weight_
        ) = None
        out_326 = torch.conv2d(
            mul__236,
            weight_195,
            l_self_modules_stages_modules_3_modules_10_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__236 = (
            weight_195
        ) = (
            l_self_modules_stages_modules_3_modules_10_modules_conv3_parameters_bias_
        ) = None
        x_se_184 = out_326.mean((2, 3), keepdim=True)
        x_se_185 = torch.conv2d(
            x_se_184,
            l_self_modules_stages_modules_3_modules_10_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_10_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_184 = l_self_modules_stages_modules_3_modules_10_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_10_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_186 = torch.nn.functional.relu(x_se_185, inplace=True)
        x_se_185 = None
        x_se_187 = torch.conv2d(
            x_se_186,
            l_self_modules_stages_modules_3_modules_10_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_10_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_186 = l_self_modules_stages_modules_3_modules_10_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_10_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_46 = x_se_187.sigmoid()
        x_se_187 = None
        mul_381 = out_326 * sigmoid_46
        out_326 = sigmoid_46 = None
        out_327 = 2.0 * mul_381
        mul_381 = None
        mul__237 = out_327.mul_(
            l_self_modules_stages_modules_3_modules_10_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_3_modules_10_parameters_skipinit_gain_ = (
            mul__237
        ) = None
        mul_383 = out_327 * 0.2
        out_327 = None
        out_328 = mul_383 + out_321
        mul_383 = out_321 = None
        gelu_191 = torch._C._nn.gelu(out_328)
        mul__238 = gelu_191.mul_(1.7015043497085571)
        gelu_191 = None
        out_329 = mul__238 * 0.8333333333333333
        mul__238 = None
        reshape_196 = l_self_modules_stages_modules_3_modules_11_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_385 = (
            l_self_modules_stages_modules_3_modules_11_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_3_modules_11_modules_conv1_parameters_gain_ = None
        view_196 = mul_385.view(-1)
        mul_385 = None
        batch_norm_196 = torch.nn.functional.batch_norm(
            reshape_196,
            None,
            None,
            weight=view_196,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_196 = view_196 = None
        weight_196 = batch_norm_196.reshape_as(
            l_self_modules_stages_modules_3_modules_11_modules_conv1_parameters_weight_
        )
        batch_norm_196 = (
            l_self_modules_stages_modules_3_modules_11_modules_conv1_parameters_weight_
        ) = None
        out_330 = torch.conv2d(
            out_329,
            weight_196,
            l_self_modules_stages_modules_3_modules_11_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_329 = (
            weight_196
        ) = (
            l_self_modules_stages_modules_3_modules_11_modules_conv1_parameters_bias_
        ) = None
        gelu_192 = torch._C._nn.gelu(out_330)
        out_330 = None
        mul__239 = gelu_192.mul_(1.7015043497085571)
        gelu_192 = None
        reshape_197 = l_self_modules_stages_modules_3_modules_11_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_386 = (
            l_self_modules_stages_modules_3_modules_11_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_11_modules_conv2_parameters_gain_ = None
        view_197 = mul_386.view(-1)
        mul_386 = None
        batch_norm_197 = torch.nn.functional.batch_norm(
            reshape_197,
            None,
            None,
            weight=view_197,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_197 = view_197 = None
        weight_197 = batch_norm_197.reshape_as(
            l_self_modules_stages_modules_3_modules_11_modules_conv2_parameters_weight_
        )
        batch_norm_197 = (
            l_self_modules_stages_modules_3_modules_11_modules_conv2_parameters_weight_
        ) = None
        out_331 = torch.conv2d(
            mul__239,
            weight_197,
            l_self_modules_stages_modules_3_modules_11_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__239 = (
            weight_197
        ) = (
            l_self_modules_stages_modules_3_modules_11_modules_conv2_parameters_bias_
        ) = None
        gelu_193 = torch._C._nn.gelu(out_331)
        out_331 = None
        mul__240 = gelu_193.mul_(1.7015043497085571)
        gelu_193 = None
        reshape_198 = l_self_modules_stages_modules_3_modules_11_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_387 = (
            l_self_modules_stages_modules_3_modules_11_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_11_modules_conv2b_parameters_gain_ = (
            None
        )
        view_198 = mul_387.view(-1)
        mul_387 = None
        batch_norm_198 = torch.nn.functional.batch_norm(
            reshape_198,
            None,
            None,
            weight=view_198,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_198 = view_198 = None
        weight_198 = batch_norm_198.reshape_as(
            l_self_modules_stages_modules_3_modules_11_modules_conv2b_parameters_weight_
        )
        batch_norm_198 = (
            l_self_modules_stages_modules_3_modules_11_modules_conv2b_parameters_weight_
        ) = None
        out_332 = torch.conv2d(
            mul__240,
            weight_198,
            l_self_modules_stages_modules_3_modules_11_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__240 = (
            weight_198
        ) = (
            l_self_modules_stages_modules_3_modules_11_modules_conv2b_parameters_bias_
        ) = None
        gelu_194 = torch._C._nn.gelu(out_332)
        out_332 = None
        mul__241 = gelu_194.mul_(1.7015043497085571)
        gelu_194 = None
        reshape_199 = l_self_modules_stages_modules_3_modules_11_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_388 = (
            l_self_modules_stages_modules_3_modules_11_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_3_modules_11_modules_conv3_parameters_gain_ = None
        view_199 = mul_388.view(-1)
        mul_388 = None
        batch_norm_199 = torch.nn.functional.batch_norm(
            reshape_199,
            None,
            None,
            weight=view_199,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_199 = view_199 = None
        weight_199 = batch_norm_199.reshape_as(
            l_self_modules_stages_modules_3_modules_11_modules_conv3_parameters_weight_
        )
        batch_norm_199 = (
            l_self_modules_stages_modules_3_modules_11_modules_conv3_parameters_weight_
        ) = None
        out_333 = torch.conv2d(
            mul__241,
            weight_199,
            l_self_modules_stages_modules_3_modules_11_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__241 = (
            weight_199
        ) = (
            l_self_modules_stages_modules_3_modules_11_modules_conv3_parameters_bias_
        ) = None
        x_se_188 = out_333.mean((2, 3), keepdim=True)
        x_se_189 = torch.conv2d(
            x_se_188,
            l_self_modules_stages_modules_3_modules_11_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_11_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_188 = l_self_modules_stages_modules_3_modules_11_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_11_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_190 = torch.nn.functional.relu(x_se_189, inplace=True)
        x_se_189 = None
        x_se_191 = torch.conv2d(
            x_se_190,
            l_self_modules_stages_modules_3_modules_11_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_11_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_190 = l_self_modules_stages_modules_3_modules_11_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_11_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_47 = x_se_191.sigmoid()
        x_se_191 = None
        mul_389 = out_333 * sigmoid_47
        out_333 = sigmoid_47 = None
        out_334 = 2.0 * mul_389
        mul_389 = None
        mul__242 = out_334.mul_(
            l_self_modules_stages_modules_3_modules_11_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_3_modules_11_parameters_skipinit_gain_ = (
            mul__242
        ) = None
        mul_391 = out_334 * 0.2
        out_334 = None
        out_335 = mul_391 + out_328
        mul_391 = out_328 = None
        reshape_200 = l_self_modules_final_conv_parameters_weight_.reshape(1, 3072, -1)
        mul_392 = l_self_modules_final_conv_parameters_gain_ * 0.02551551815399144
        l_self_modules_final_conv_parameters_gain_ = None
        view_200 = mul_392.view(-1)
        mul_392 = None
        batch_norm_200 = torch.nn.functional.batch_norm(
            reshape_200,
            None,
            None,
            weight=view_200,
            training=True,
            momentum=0.0,
            eps=1e-05,
        )
        reshape_200 = view_200 = None
        weight_200 = batch_norm_200.reshape_as(
            l_self_modules_final_conv_parameters_weight_
        )
        batch_norm_200 = l_self_modules_final_conv_parameters_weight_ = None
        x_5 = torch.conv2d(
            out_335,
            weight_200,
            l_self_modules_final_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_335 = weight_200 = l_self_modules_final_conv_parameters_bias_ = None
        gelu_195 = torch._C._nn.gelu(x_5)
        x_5 = None
        x_6 = gelu_195.mul_(1.7015043497085571)
        gelu_195 = None
        x_7 = torch.nn.functional.adaptive_avg_pool2d(x_6, 1)
        x_6 = None
        x_8 = x_7.flatten(1, -1)
        x_7 = None
        x_9 = torch.nn.functional.dropout(x_8, 0.0, False, False)
        x_8 = None
        x_10 = torch._C._nn.linear(
            x_9,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_9 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_10,)
