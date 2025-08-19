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
        out_6 = None
        mul__8 = gelu_7.mul_(1.7015043497085571)
        gelu_7 = None
        out_7 = mul__8 * 0.9805806756909201
        mul__8 = None
        avg_pool2d = torch._C._nn.avg_pool2d(out_7, 2, 2, 0, True, False, None)
        reshape_9 = l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_14 = (
            l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_gain_
            * 0.0625
        )
        l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_gain_ = (
            None
        )
        view_9 = mul_14.view(-1)
        mul_14 = None
        batch_norm_9 = torch.nn.functional.batch_norm(
            reshape_9, None, None, weight=view_9, training=True, momentum=0.0, eps=1e-05
        )
        reshape_9 = view_9 = None
        weight_9 = batch_norm_9.reshape_as(
            l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_
        )
        batch_norm_9 = l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        shortcut_1 = torch.conv2d(
            avg_pool2d,
            weight_9,
            l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        avg_pool2d = (
            weight_9
        ) = l_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_bias_ = (None)
        reshape_10 = l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_15 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_gain_
            * 0.0625
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_gain_ = None
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
            l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_weight_
        )
        batch_norm_10 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_weight_
        ) = None
        out_8 = torch.conv2d(
            out_7,
            weight_10,
            l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_7 = (
            weight_10
        ) = (
            l_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_bias_
        ) = None
        gelu_8 = torch._C._nn.gelu(out_8)
        out_8 = None
        mul__9 = gelu_8.mul_(1.7015043497085571)
        gelu_8 = None
        x_2 = torch._C._nn.pad(mul__9, (0, 1, 0, 1), "constant", 0)
        mul__9 = None
        reshape_11 = l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_16 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_gain_ = None
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
            l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_weight_
        )
        batch_norm_11 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_weight_
        ) = None
        out_9 = torch.conv2d(
            x_2,
            weight_11,
            l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            2,
        )
        x_2 = (
            weight_11
        ) = (
            l_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_bias_
        ) = None
        gelu_9 = torch._C._nn.gelu(out_9)
        out_9 = None
        mul__10 = gelu_9.mul_(1.7015043497085571)
        gelu_9 = None
        reshape_12 = l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_17 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_gain_ = None
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
            l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_weight_
        )
        batch_norm_12 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_weight_
        ) = None
        out_10 = torch.conv2d(
            mul__10,
            weight_12,
            l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        mul__10 = (
            weight_12
        ) = (
            l_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_bias_
        ) = None
        gelu_10 = torch._C._nn.gelu(out_10)
        out_10 = None
        mul__11 = gelu_10.mul_(1.7015043497085571)
        gelu_10 = None
        reshape_13 = l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_18 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_gain_
            * 0.0625
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_gain_ = None
        view_13 = mul_18.view(-1)
        mul_18 = None
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
            l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_weight_
        )
        batch_norm_13 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_weight_
        ) = None
        out_11 = torch.conv2d(
            mul__11,
            weight_13,
            l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__11 = (
            weight_13
        ) = (
            l_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_bias_
        ) = None
        x_se_4 = out_11.mean((2, 3), keepdim=True)
        x_se_5 = torch.conv2d(
            x_se_4,
            l_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_4 = l_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_6 = torch.nn.functional.relu(x_se_5, inplace=True)
        x_se_5 = None
        x_se_7 = torch.conv2d(
            x_se_6,
            l_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_6 = l_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_1 = x_se_7.sigmoid()
        x_se_7 = None
        mul_19 = out_11 * sigmoid_1
        out_11 = sigmoid_1 = None
        out_12 = 2.0 * mul_19
        mul_19 = None
        mul__12 = out_12.mul_(
            l_self_modules_stages_modules_1_modules_0_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_1_modules_0_parameters_skipinit_gain_ = (
            mul__12
        ) = None
        mul_21 = out_12 * 0.2
        out_12 = None
        out_13 = mul_21 + shortcut_1
        mul_21 = shortcut_1 = None
        gelu_11 = torch._C._nn.gelu(out_13)
        mul__13 = gelu_11.mul_(1.7015043497085571)
        gelu_11 = None
        out_14 = mul__13 * 0.9805806756909201
        mul__13 = None
        reshape_14 = l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_23 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_gain_
            * 0.04419417382415922
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_gain_ = None
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
            l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_weight_
        )
        batch_norm_14 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_weight_
        ) = None
        out_15 = torch.conv2d(
            out_14,
            weight_14,
            l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_14 = (
            weight_14
        ) = (
            l_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_bias_
        ) = None
        gelu_12 = torch._C._nn.gelu(out_15)
        out_15 = None
        mul__14 = gelu_12.mul_(1.7015043497085571)
        gelu_12 = None
        reshape_15 = l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_24 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_gain_ = None
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
            l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_weight_
        )
        batch_norm_15 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_weight_
        ) = None
        out_16 = torch.conv2d(
            mul__14,
            weight_15,
            l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        mul__14 = (
            weight_15
        ) = (
            l_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_bias_
        ) = None
        gelu_13 = torch._C._nn.gelu(out_16)
        out_16 = None
        mul__15 = gelu_13.mul_(1.7015043497085571)
        gelu_13 = None
        reshape_16 = l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_weight_.reshape(
            1, 256, -1
        )
        mul_25 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_gain_ = None
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
            l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_weight_
        )
        batch_norm_16 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_weight_
        ) = None
        out_17 = torch.conv2d(
            mul__15,
            weight_16,
            l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        mul__15 = (
            weight_16
        ) = (
            l_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_bias_
        ) = None
        gelu_14 = torch._C._nn.gelu(out_17)
        out_17 = None
        mul__16 = gelu_14.mul_(1.7015043497085571)
        gelu_14 = None
        reshape_17 = l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_weight_.reshape(
            1, 512, -1
        )
        mul_26 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_gain_
            * 0.0625
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_gain_ = None
        view_17 = mul_26.view(-1)
        mul_26 = None
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
            l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_weight_
        )
        batch_norm_17 = (
            l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_weight_
        ) = None
        out_18 = torch.conv2d(
            mul__16,
            weight_17,
            l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__16 = (
            weight_17
        ) = (
            l_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_bias_
        ) = None
        x_se_8 = out_18.mean((2, 3), keepdim=True)
        x_se_9 = torch.conv2d(
            x_se_8,
            l_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_8 = l_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_10 = torch.nn.functional.relu(x_se_9, inplace=True)
        x_se_9 = None
        x_se_11 = torch.conv2d(
            x_se_10,
            l_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_10 = l_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_2 = x_se_11.sigmoid()
        x_se_11 = None
        mul_27 = out_18 * sigmoid_2
        out_18 = sigmoid_2 = None
        out_19 = 2.0 * mul_27
        mul_27 = None
        mul__17 = out_19.mul_(
            l_self_modules_stages_modules_1_modules_1_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_1_modules_1_parameters_skipinit_gain_ = (
            mul__17
        ) = None
        mul_29 = out_19 * 0.2
        out_19 = None
        out_20 = mul_29 + out_13
        mul_29 = out_13 = None
        gelu_15 = torch._C._nn.gelu(out_20)
        out_20 = None
        mul__18 = gelu_15.mul_(1.7015043497085571)
        gelu_15 = None
        out_21 = mul__18 * 0.9622504486493761
        mul__18 = None
        avg_pool2d_1 = torch._C._nn.avg_pool2d(out_21, 2, 2, 0, True, False, None)
        reshape_18 = l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_31 = (
            l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_gain_
            * 0.04419417382415922
        )
        l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_gain_ = (
            None
        )
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
            l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_
        )
        batch_norm_18 = l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        shortcut_2 = torch.conv2d(
            avg_pool2d_1,
            weight_18,
            l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        avg_pool2d_1 = (
            weight_18
        ) = l_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_bias_ = (None)
        reshape_19 = l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_32 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_gain_
            * 0.04419417382415922
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_gain_ = None
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
            l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_weight_
        )
        batch_norm_19 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_weight_
        ) = None
        out_22 = torch.conv2d(
            out_21,
            weight_19,
            l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_21 = (
            weight_19
        ) = (
            l_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_bias_
        ) = None
        gelu_16 = torch._C._nn.gelu(out_22)
        out_22 = None
        mul__19 = gelu_16.mul_(1.7015043497085571)
        gelu_16 = None
        x_3 = torch._C._nn.pad(mul__19, (0, 1, 0, 1), "constant", 0)
        mul__19 = None
        reshape_20 = l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_33 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_gain_ = None
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
            l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_weight_
        )
        batch_norm_20 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_weight_
        ) = None
        out_23 = torch.conv2d(
            x_3,
            weight_20,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            6,
        )
        x_3 = (
            weight_20
        ) = (
            l_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_bias_
        ) = None
        gelu_17 = torch._C._nn.gelu(out_23)
        out_23 = None
        mul__20 = gelu_17.mul_(1.7015043497085571)
        gelu_17 = None
        reshape_21 = l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_34 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_gain_ = None
        view_21 = mul_34.view(-1)
        mul_34 = None
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
            l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_weight_
        )
        batch_norm_21 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_weight_
        ) = None
        out_24 = torch.conv2d(
            mul__20,
            weight_21,
            l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__20 = (
            weight_21
        ) = (
            l_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_bias_
        ) = None
        gelu_18 = torch._C._nn.gelu(out_24)
        out_24 = None
        mul__21 = gelu_18.mul_(1.7015043497085571)
        gelu_18 = None
        reshape_22 = l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_35 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_gain_ = None
        view_22 = mul_35.view(-1)
        mul_35 = None
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
            l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_weight_
        )
        batch_norm_22 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_weight_
        ) = None
        out_25 = torch.conv2d(
            mul__21,
            weight_22,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__21 = (
            weight_22
        ) = (
            l_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_bias_
        ) = None
        x_se_12 = out_25.mean((2, 3), keepdim=True)
        x_se_13 = torch.conv2d(
            x_se_12,
            l_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_12 = l_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_14 = torch.nn.functional.relu(x_se_13, inplace=True)
        x_se_13 = None
        x_se_15 = torch.conv2d(
            x_se_14,
            l_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_14 = l_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_3 = x_se_15.sigmoid()
        x_se_15 = None
        mul_36 = out_25 * sigmoid_3
        out_25 = sigmoid_3 = None
        out_26 = 2.0 * mul_36
        mul_36 = None
        mul__22 = out_26.mul_(
            l_self_modules_stages_modules_2_modules_0_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_0_parameters_skipinit_gain_ = (
            mul__22
        ) = None
        mul_38 = out_26 * 0.2
        out_26 = None
        out_27 = mul_38 + shortcut_2
        mul_38 = shortcut_2 = None
        gelu_19 = torch._C._nn.gelu(out_27)
        mul__23 = gelu_19.mul_(1.7015043497085571)
        gelu_19 = None
        out_28 = mul__23 * 0.9805806756909201
        mul__23 = None
        reshape_23 = l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_40 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_gain_ = None
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
            l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_weight_
        )
        batch_norm_23 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_weight_
        ) = None
        out_29 = torch.conv2d(
            out_28,
            weight_23,
            l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_28 = (
            weight_23
        ) = (
            l_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_bias_
        ) = None
        gelu_20 = torch._C._nn.gelu(out_29)
        out_29 = None
        mul__24 = gelu_20.mul_(1.7015043497085571)
        gelu_20 = None
        reshape_24 = l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_41 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_gain_ = None
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
            l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_weight_
        )
        batch_norm_24 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_weight_
        ) = None
        out_30 = torch.conv2d(
            mul__24,
            weight_24,
            l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__24 = (
            weight_24
        ) = (
            l_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_bias_
        ) = None
        gelu_21 = torch._C._nn.gelu(out_30)
        out_30 = None
        mul__25 = gelu_21.mul_(1.7015043497085571)
        gelu_21 = None
        reshape_25 = l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_42 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_gain_ = None
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
            l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_weight_
        )
        batch_norm_25 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_weight_
        ) = None
        out_31 = torch.conv2d(
            mul__25,
            weight_25,
            l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__25 = (
            weight_25
        ) = (
            l_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_bias_
        ) = None
        gelu_22 = torch._C._nn.gelu(out_31)
        out_31 = None
        mul__26 = gelu_22.mul_(1.7015043497085571)
        gelu_22 = None
        reshape_26 = l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_43 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_gain_ = None
        view_26 = mul_43.view(-1)
        mul_43 = None
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
            l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_weight_
        )
        batch_norm_26 = (
            l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_weight_
        ) = None
        out_32 = torch.conv2d(
            mul__26,
            weight_26,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__26 = (
            weight_26
        ) = (
            l_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_bias_
        ) = None
        x_se_16 = out_32.mean((2, 3), keepdim=True)
        x_se_17 = torch.conv2d(
            x_se_16,
            l_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_16 = l_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_18 = torch.nn.functional.relu(x_se_17, inplace=True)
        x_se_17 = None
        x_se_19 = torch.conv2d(
            x_se_18,
            l_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_18 = l_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_4 = x_se_19.sigmoid()
        x_se_19 = None
        mul_44 = out_32 * sigmoid_4
        out_32 = sigmoid_4 = None
        out_33 = 2.0 * mul_44
        mul_44 = None
        mul__27 = out_33.mul_(
            l_self_modules_stages_modules_2_modules_1_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_1_parameters_skipinit_gain_ = (
            mul__27
        ) = None
        mul_46 = out_33 * 0.2
        out_33 = None
        out_34 = mul_46 + out_27
        mul_46 = out_27 = None
        gelu_23 = torch._C._nn.gelu(out_34)
        mul__28 = gelu_23.mul_(1.7015043497085571)
        gelu_23 = None
        out_35 = mul__28 * 0.9622504486493761
        mul__28 = None
        reshape_27 = l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_48 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_gain_ = None
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
            l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_weight_
        )
        batch_norm_27 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_weight_
        ) = None
        out_36 = torch.conv2d(
            out_35,
            weight_27,
            l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_35 = (
            weight_27
        ) = (
            l_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_bias_
        ) = None
        gelu_24 = torch._C._nn.gelu(out_36)
        out_36 = None
        mul__29 = gelu_24.mul_(1.7015043497085571)
        gelu_24 = None
        reshape_28 = l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_49 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_gain_ = None
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
            l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_weight_
        )
        batch_norm_28 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_weight_
        ) = None
        out_37 = torch.conv2d(
            mul__29,
            weight_28,
            l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__29 = (
            weight_28
        ) = (
            l_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_bias_
        ) = None
        gelu_25 = torch._C._nn.gelu(out_37)
        out_37 = None
        mul__30 = gelu_25.mul_(1.7015043497085571)
        gelu_25 = None
        reshape_29 = l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_50 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_gain_ = None
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
            l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_weight_
        )
        batch_norm_29 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_weight_
        ) = None
        out_38 = torch.conv2d(
            mul__30,
            weight_29,
            l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__30 = (
            weight_29
        ) = (
            l_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_bias_
        ) = None
        gelu_26 = torch._C._nn.gelu(out_38)
        out_38 = None
        mul__31 = gelu_26.mul_(1.7015043497085571)
        gelu_26 = None
        reshape_30 = l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_51 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_gain_ = None
        view_30 = mul_51.view(-1)
        mul_51 = None
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
            l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_weight_
        )
        batch_norm_30 = (
            l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_weight_
        ) = None
        out_39 = torch.conv2d(
            mul__31,
            weight_30,
            l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__31 = (
            weight_30
        ) = (
            l_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_bias_
        ) = None
        x_se_20 = out_39.mean((2, 3), keepdim=True)
        x_se_21 = torch.conv2d(
            x_se_20,
            l_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_20 = l_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_22 = torch.nn.functional.relu(x_se_21, inplace=True)
        x_se_21 = None
        x_se_23 = torch.conv2d(
            x_se_22,
            l_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_22 = l_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_5 = x_se_23.sigmoid()
        x_se_23 = None
        mul_52 = out_39 * sigmoid_5
        out_39 = sigmoid_5 = None
        out_40 = 2.0 * mul_52
        mul_52 = None
        mul__32 = out_40.mul_(
            l_self_modules_stages_modules_2_modules_2_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_2_parameters_skipinit_gain_ = (
            mul__32
        ) = None
        mul_54 = out_40 * 0.2
        out_40 = None
        out_41 = mul_54 + out_34
        mul_54 = out_34 = None
        gelu_27 = torch._C._nn.gelu(out_41)
        mul__33 = gelu_27.mul_(1.7015043497085571)
        gelu_27 = None
        out_42 = mul__33 * 0.9449111825230679
        mul__33 = None
        reshape_31 = l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_56 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_gain_ = None
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
            l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_weight_
        )
        batch_norm_31 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_weight_
        ) = None
        out_43 = torch.conv2d(
            out_42,
            weight_31,
            l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_42 = (
            weight_31
        ) = (
            l_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_bias_
        ) = None
        gelu_28 = torch._C._nn.gelu(out_43)
        out_43 = None
        mul__34 = gelu_28.mul_(1.7015043497085571)
        gelu_28 = None
        reshape_32 = l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_57 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_gain_ = None
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
            l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_weight_
        )
        batch_norm_32 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_weight_
        ) = None
        out_44 = torch.conv2d(
            mul__34,
            weight_32,
            l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__34 = (
            weight_32
        ) = (
            l_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_bias_
        ) = None
        gelu_29 = torch._C._nn.gelu(out_44)
        out_44 = None
        mul__35 = gelu_29.mul_(1.7015043497085571)
        gelu_29 = None
        reshape_33 = l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_58 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_gain_ = None
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
            l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_weight_
        )
        batch_norm_33 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_weight_
        ) = None
        out_45 = torch.conv2d(
            mul__35,
            weight_33,
            l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__35 = (
            weight_33
        ) = (
            l_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_bias_
        ) = None
        gelu_30 = torch._C._nn.gelu(out_45)
        out_45 = None
        mul__36 = gelu_30.mul_(1.7015043497085571)
        gelu_30 = None
        reshape_34 = l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_59 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_gain_ = None
        view_34 = mul_59.view(-1)
        mul_59 = None
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
            l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_weight_
        )
        batch_norm_34 = (
            l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_weight_
        ) = None
        out_46 = torch.conv2d(
            mul__36,
            weight_34,
            l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__36 = (
            weight_34
        ) = (
            l_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_bias_
        ) = None
        x_se_24 = out_46.mean((2, 3), keepdim=True)
        x_se_25 = torch.conv2d(
            x_se_24,
            l_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_24 = l_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_26 = torch.nn.functional.relu(x_se_25, inplace=True)
        x_se_25 = None
        x_se_27 = torch.conv2d(
            x_se_26,
            l_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_26 = l_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_6 = x_se_27.sigmoid()
        x_se_27 = None
        mul_60 = out_46 * sigmoid_6
        out_46 = sigmoid_6 = None
        out_47 = 2.0 * mul_60
        mul_60 = None
        mul__37 = out_47.mul_(
            l_self_modules_stages_modules_2_modules_3_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_3_parameters_skipinit_gain_ = (
            mul__37
        ) = None
        mul_62 = out_47 * 0.2
        out_47 = None
        out_48 = mul_62 + out_41
        mul_62 = out_41 = None
        gelu_31 = torch._C._nn.gelu(out_48)
        mul__38 = gelu_31.mul_(1.7015043497085571)
        gelu_31 = None
        out_49 = mul__38 * 0.9284766908852592
        mul__38 = None
        reshape_35 = l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_64 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_gain_ = None
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
            l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_weight_
        )
        batch_norm_35 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_weight_
        ) = None
        out_50 = torch.conv2d(
            out_49,
            weight_35,
            l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_49 = (
            weight_35
        ) = (
            l_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_bias_
        ) = None
        gelu_32 = torch._C._nn.gelu(out_50)
        out_50 = None
        mul__39 = gelu_32.mul_(1.7015043497085571)
        gelu_32 = None
        reshape_36 = l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_65 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_gain_ = None
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
            l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_weight_
        )
        batch_norm_36 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_weight_
        ) = None
        out_51 = torch.conv2d(
            mul__39,
            weight_36,
            l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__39 = (
            weight_36
        ) = (
            l_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_bias_
        ) = None
        gelu_33 = torch._C._nn.gelu(out_51)
        out_51 = None
        mul__40 = gelu_33.mul_(1.7015043497085571)
        gelu_33 = None
        reshape_37 = l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_66 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_gain_ = None
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
            l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_weight_
        )
        batch_norm_37 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_weight_
        ) = None
        out_52 = torch.conv2d(
            mul__40,
            weight_37,
            l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__40 = (
            weight_37
        ) = (
            l_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_bias_
        ) = None
        gelu_34 = torch._C._nn.gelu(out_52)
        out_52 = None
        mul__41 = gelu_34.mul_(1.7015043497085571)
        gelu_34 = None
        reshape_38 = l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_67 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_gain_ = None
        view_38 = mul_67.view(-1)
        mul_67 = None
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
            l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_weight_
        )
        batch_norm_38 = (
            l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_weight_
        ) = None
        out_53 = torch.conv2d(
            mul__41,
            weight_38,
            l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__41 = (
            weight_38
        ) = (
            l_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_bias_
        ) = None
        x_se_28 = out_53.mean((2, 3), keepdim=True)
        x_se_29 = torch.conv2d(
            x_se_28,
            l_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_28 = l_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_30 = torch.nn.functional.relu(x_se_29, inplace=True)
        x_se_29 = None
        x_se_31 = torch.conv2d(
            x_se_30,
            l_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_30 = l_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_7 = x_se_31.sigmoid()
        x_se_31 = None
        mul_68 = out_53 * sigmoid_7
        out_53 = sigmoid_7 = None
        out_54 = 2.0 * mul_68
        mul_68 = None
        mul__42 = out_54.mul_(
            l_self_modules_stages_modules_2_modules_4_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_4_parameters_skipinit_gain_ = (
            mul__42
        ) = None
        mul_70 = out_54 * 0.2
        out_54 = None
        out_55 = mul_70 + out_48
        mul_70 = out_48 = None
        gelu_35 = torch._C._nn.gelu(out_55)
        mul__43 = gelu_35.mul_(1.7015043497085571)
        gelu_35 = None
        out_56 = mul__43 * 0.9128709291752768
        mul__43 = None
        reshape_39 = l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_72 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_gain_ = None
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
            l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_weight_
        )
        batch_norm_39 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_weight_
        ) = None
        out_57 = torch.conv2d(
            out_56,
            weight_39,
            l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_56 = (
            weight_39
        ) = (
            l_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_bias_
        ) = None
        gelu_36 = torch._C._nn.gelu(out_57)
        out_57 = None
        mul__44 = gelu_36.mul_(1.7015043497085571)
        gelu_36 = None
        reshape_40 = l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_73 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_gain_ = None
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
            l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_weight_
        )
        batch_norm_40 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_weight_
        ) = None
        out_58 = torch.conv2d(
            mul__44,
            weight_40,
            l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__44 = (
            weight_40
        ) = (
            l_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_bias_
        ) = None
        gelu_37 = torch._C._nn.gelu(out_58)
        out_58 = None
        mul__45 = gelu_37.mul_(1.7015043497085571)
        gelu_37 = None
        reshape_41 = l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_74 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_gain_ = None
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
            l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_weight_
        )
        batch_norm_41 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_weight_
        ) = None
        out_59 = torch.conv2d(
            mul__45,
            weight_41,
            l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__45 = (
            weight_41
        ) = (
            l_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_bias_
        ) = None
        gelu_38 = torch._C._nn.gelu(out_59)
        out_59 = None
        mul__46 = gelu_38.mul_(1.7015043497085571)
        gelu_38 = None
        reshape_42 = l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_75 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_gain_ = None
        view_42 = mul_75.view(-1)
        mul_75 = None
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
            l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_weight_
        )
        batch_norm_42 = (
            l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_weight_
        ) = None
        out_60 = torch.conv2d(
            mul__46,
            weight_42,
            l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__46 = (
            weight_42
        ) = (
            l_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_bias_
        ) = None
        x_se_32 = out_60.mean((2, 3), keepdim=True)
        x_se_33 = torch.conv2d(
            x_se_32,
            l_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_32 = l_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_34 = torch.nn.functional.relu(x_se_33, inplace=True)
        x_se_33 = None
        x_se_35 = torch.conv2d(
            x_se_34,
            l_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_34 = l_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_8 = x_se_35.sigmoid()
        x_se_35 = None
        mul_76 = out_60 * sigmoid_8
        out_60 = sigmoid_8 = None
        out_61 = 2.0 * mul_76
        mul_76 = None
        mul__47 = out_61.mul_(
            l_self_modules_stages_modules_2_modules_5_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_2_modules_5_parameters_skipinit_gain_ = (
            mul__47
        ) = None
        mul_78 = out_61 * 0.2
        out_61 = None
        out_62 = mul_78 + out_55
        mul_78 = out_55 = None
        gelu_39 = torch._C._nn.gelu(out_62)
        out_62 = None
        mul__48 = gelu_39.mul_(1.7015043497085571)
        gelu_39 = None
        out_63 = mul__48 * 0.8980265101338745
        mul__48 = None
        avg_pool2d_2 = torch._C._nn.avg_pool2d(out_63, 2, 2, 0, True, False, None)
        reshape_43 = l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_80 = (
            l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_gain_ = (
            None
        )
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
            l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_weight_
        )
        batch_norm_43 = l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        shortcut_3 = torch.conv2d(
            avg_pool2d_2,
            weight_43,
            l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        avg_pool2d_2 = (
            weight_43
        ) = l_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_bias_ = (None)
        reshape_44 = l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_81 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_gain_ = None
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
            l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_weight_
        )
        batch_norm_44 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_weight_
        ) = None
        out_64 = torch.conv2d(
            out_63,
            weight_44,
            l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_63 = (
            weight_44
        ) = (
            l_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_bias_
        ) = None
        gelu_40 = torch._C._nn.gelu(out_64)
        out_64 = None
        mul__49 = gelu_40.mul_(1.7015043497085571)
        gelu_40 = None
        x_4 = torch._C._nn.pad(mul__49, (0, 1, 0, 1), "constant", 0)
        mul__49 = None
        reshape_45 = l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_82 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_gain_ = None
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
            l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_weight_
        )
        batch_norm_45 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_weight_
        ) = None
        out_65 = torch.conv2d(
            x_4,
            weight_45,
            l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            6,
        )
        x_4 = (
            weight_45
        ) = (
            l_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_bias_
        ) = None
        gelu_41 = torch._C._nn.gelu(out_65)
        out_65 = None
        mul__50 = gelu_41.mul_(1.7015043497085571)
        gelu_41 = None
        reshape_46 = l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_83 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_gain_ = None
        view_46 = mul_83.view(-1)
        mul_83 = None
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
            l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_weight_
        )
        batch_norm_46 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_weight_
        ) = None
        out_66 = torch.conv2d(
            mul__50,
            weight_46,
            l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__50 = (
            weight_46
        ) = (
            l_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_bias_
        ) = None
        gelu_42 = torch._C._nn.gelu(out_66)
        out_66 = None
        mul__51 = gelu_42.mul_(1.7015043497085571)
        gelu_42 = None
        reshape_47 = l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_84 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_gain_ = None
        view_47 = mul_84.view(-1)
        mul_84 = None
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
            l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_weight_
        )
        batch_norm_47 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_weight_
        ) = None
        out_67 = torch.conv2d(
            mul__51,
            weight_47,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__51 = (
            weight_47
        ) = (
            l_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_bias_
        ) = None
        x_se_36 = out_67.mean((2, 3), keepdim=True)
        x_se_37 = torch.conv2d(
            x_se_36,
            l_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_36 = l_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_38 = torch.nn.functional.relu(x_se_37, inplace=True)
        x_se_37 = None
        x_se_39 = torch.conv2d(
            x_se_38,
            l_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_38 = l_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_9 = x_se_39.sigmoid()
        x_se_39 = None
        mul_85 = out_67 * sigmoid_9
        out_67 = sigmoid_9 = None
        out_68 = 2.0 * mul_85
        mul_85 = None
        mul__52 = out_68.mul_(
            l_self_modules_stages_modules_3_modules_0_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_3_modules_0_parameters_skipinit_gain_ = (
            mul__52
        ) = None
        mul_87 = out_68 * 0.2
        out_68 = None
        out_69 = mul_87 + shortcut_3
        mul_87 = shortcut_3 = None
        gelu_43 = torch._C._nn.gelu(out_69)
        mul__53 = gelu_43.mul_(1.7015043497085571)
        gelu_43 = None
        out_70 = mul__53 * 0.9805806756909201
        mul__53 = None
        reshape_48 = l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_89 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_gain_ = None
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
            l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_weight_
        )
        batch_norm_48 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_weight_
        ) = None
        out_71 = torch.conv2d(
            out_70,
            weight_48,
            l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_70 = (
            weight_48
        ) = (
            l_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_bias_
        ) = None
        gelu_44 = torch._C._nn.gelu(out_71)
        out_71 = None
        mul__54 = gelu_44.mul_(1.7015043497085571)
        gelu_44 = None
        reshape_49 = l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_90 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_gain_ = None
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
            l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_weight_
        )
        batch_norm_49 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_weight_
        ) = None
        out_72 = torch.conv2d(
            mul__54,
            weight_49,
            l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__54 = (
            weight_49
        ) = (
            l_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_bias_
        ) = None
        gelu_45 = torch._C._nn.gelu(out_72)
        out_72 = None
        mul__55 = gelu_45.mul_(1.7015043497085571)
        gelu_45 = None
        reshape_50 = l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_91 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_gain_ = None
        view_50 = mul_91.view(-1)
        mul_91 = None
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
            l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_weight_
        )
        batch_norm_50 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_weight_
        ) = None
        out_73 = torch.conv2d(
            mul__55,
            weight_50,
            l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__55 = (
            weight_50
        ) = (
            l_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_bias_
        ) = None
        gelu_46 = torch._C._nn.gelu(out_73)
        out_73 = None
        mul__56 = gelu_46.mul_(1.7015043497085571)
        gelu_46 = None
        reshape_51 = l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_92 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_gain_ = None
        view_51 = mul_92.view(-1)
        mul_92 = None
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
            l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_weight_
        )
        batch_norm_51 = (
            l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_weight_
        ) = None
        out_74 = torch.conv2d(
            mul__56,
            weight_51,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__56 = (
            weight_51
        ) = (
            l_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_bias_
        ) = None
        x_se_40 = out_74.mean((2, 3), keepdim=True)
        x_se_41 = torch.conv2d(
            x_se_40,
            l_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_40 = l_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_42 = torch.nn.functional.relu(x_se_41, inplace=True)
        x_se_41 = None
        x_se_43 = torch.conv2d(
            x_se_42,
            l_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_42 = l_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_10 = x_se_43.sigmoid()
        x_se_43 = None
        mul_93 = out_74 * sigmoid_10
        out_74 = sigmoid_10 = None
        out_75 = 2.0 * mul_93
        mul_93 = None
        mul__57 = out_75.mul_(
            l_self_modules_stages_modules_3_modules_1_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_3_modules_1_parameters_skipinit_gain_ = (
            mul__57
        ) = None
        mul_95 = out_75 * 0.2
        out_75 = None
        out_76 = mul_95 + out_69
        mul_95 = out_69 = None
        gelu_47 = torch._C._nn.gelu(out_76)
        mul__58 = gelu_47.mul_(1.7015043497085571)
        gelu_47 = None
        out_77 = mul__58 * 0.9622504486493761
        mul__58 = None
        reshape_52 = l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_97 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_gain_
            * 0.02551551815399144
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_gain_ = None
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
            l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_weight_
        )
        batch_norm_52 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_weight_
        ) = None
        out_78 = torch.conv2d(
            out_77,
            weight_52,
            l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_77 = (
            weight_52
        ) = (
            l_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_bias_
        ) = None
        gelu_48 = torch._C._nn.gelu(out_78)
        out_78 = None
        mul__59 = gelu_48.mul_(1.7015043497085571)
        gelu_48 = None
        reshape_53 = l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_98 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_gain_ = None
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
            l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_weight_
        )
        batch_norm_53 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_weight_
        ) = None
        out_79 = torch.conv2d(
            mul__59,
            weight_53,
            l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__59 = (
            weight_53
        ) = (
            l_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_bias_
        ) = None
        gelu_49 = torch._C._nn.gelu(out_79)
        out_79 = None
        mul__60 = gelu_49.mul_(1.7015043497085571)
        gelu_49 = None
        reshape_54 = l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_weight_.reshape(
            1, 768, -1
        )
        mul_99 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_gain_
            * 0.02946278254943948
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_gain_ = None
        view_54 = mul_99.view(-1)
        mul_99 = None
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
            l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_weight_
        )
        batch_norm_54 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_weight_
        ) = None
        out_80 = torch.conv2d(
            mul__60,
            weight_54,
            l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            6,
        )
        mul__60 = (
            weight_54
        ) = (
            l_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_bias_
        ) = None
        gelu_50 = torch._C._nn.gelu(out_80)
        out_80 = None
        mul__61 = gelu_50.mul_(1.7015043497085571)
        gelu_50 = None
        reshape_55 = l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_weight_.reshape(
            1, 1536, -1
        )
        mul_100 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_gain_
            * 0.03608439182435161
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_gain_ = None
        view_55 = mul_100.view(-1)
        mul_100 = None
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
            l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_weight_
        )
        batch_norm_55 = (
            l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_weight_
        ) = None
        out_81 = torch.conv2d(
            mul__61,
            weight_55,
            l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        mul__61 = (
            weight_55
        ) = (
            l_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_bias_
        ) = None
        x_se_44 = out_81.mean((2, 3), keepdim=True)
        x_se_45 = torch.conv2d(
            x_se_44,
            l_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_44 = l_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_fc1_parameters_bias_ = (None)
        x_se_46 = torch.nn.functional.relu(x_se_45, inplace=True)
        x_se_45 = None
        x_se_47 = torch.conv2d(
            x_se_46,
            l_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_46 = l_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_fc2_parameters_bias_ = (None)
        sigmoid_11 = x_se_47.sigmoid()
        x_se_47 = None
        mul_101 = out_81 * sigmoid_11
        out_81 = sigmoid_11 = None
        out_82 = 2.0 * mul_101
        mul_101 = None
        mul__62 = out_82.mul_(
            l_self_modules_stages_modules_3_modules_2_parameters_skipinit_gain_
        )
        l_self_modules_stages_modules_3_modules_2_parameters_skipinit_gain_ = (
            mul__62
        ) = None
        mul_103 = out_82 * 0.2
        out_82 = None
        out_83 = mul_103 + out_76
        mul_103 = out_76 = None
        reshape_56 = l_self_modules_final_conv_parameters_weight_.reshape(1, 3072, -1)
        mul_104 = l_self_modules_final_conv_parameters_gain_ * 0.02551551815399144
        l_self_modules_final_conv_parameters_gain_ = None
        view_56 = mul_104.view(-1)
        mul_104 = None
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
            l_self_modules_final_conv_parameters_weight_
        )
        batch_norm_56 = l_self_modules_final_conv_parameters_weight_ = None
        x_5 = torch.conv2d(
            out_83,
            weight_56,
            l_self_modules_final_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_83 = weight_56 = l_self_modules_final_conv_parameters_bias_ = None
        gelu_51 = torch._C._nn.gelu(x_5)
        x_5 = None
        x_6 = gelu_51.mul_(1.7015043497085571)
        gelu_51 = None
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
