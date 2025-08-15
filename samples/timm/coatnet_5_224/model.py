import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_kxk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_kxk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_1x1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_kxk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_1x1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_kxk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_1x1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_pre_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_pre_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_pre_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_pre_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_kxk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv3_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv3_1x1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_pre_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_pre_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_pre_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_pre_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_conv1_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_conv2_kxk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_conv3_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_conv3_1x1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_pre_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_pre_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_pre_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_pre_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_conv1_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_conv2_kxk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_conv3_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_conv3_1x1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_pre_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_pre_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_pre_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_pre_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_conv1_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_conv2_kxk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_conv3_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_conv3_1x1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_pre_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_pre_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_pre_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_pre_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_conv1_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_conv2_kxk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_conv3_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_conv3_1x1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_pre_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_pre_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_pre_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_pre_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_conv1_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_conv2_kxk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_conv3_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_conv3_1x1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_27_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_27_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_27_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_27_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_x_ = L_x_
        l_self_modules_stem_modules_norm1_buffers_running_mean_ = (
            L_self_modules_stem_modules_norm1_buffers_running_mean_
        )
        l_self_modules_stem_modules_norm1_buffers_running_var_ = (
            L_self_modules_stem_modules_norm1_buffers_running_var_
        )
        l_self_modules_stem_modules_norm1_parameters_weight_ = (
            L_self_modules_stem_modules_norm1_parameters_weight_
        )
        l_self_modules_stem_modules_norm1_parameters_bias_ = (
            L_self_modules_stem_modules_norm1_parameters_bias_
        )
        l_self_modules_stem_modules_conv2_parameters_weight_ = (
            L_self_modules_stem_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_kxk_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_kxk_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_kxk_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_kxk_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_1x1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_1x1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_kxk_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_kxk_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_1x1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_1x1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_kxk_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_kxk_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_1x1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_1x1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_pre_norm_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_pre_norm_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_pre_norm_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_pre_norm_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_pre_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_pre_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_pre_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_pre_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_kxk_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_kxk_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_se_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_se_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_se_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_se_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_se_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_se_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_se_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_se_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv3_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv3_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv3_1x1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv3_1x1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_pre_norm_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_pre_norm_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_pre_norm_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_pre_norm_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_pre_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_pre_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_pre_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_pre_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_conv1_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_conv1_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_conv2_kxk_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_conv2_kxk_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_se_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_se_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_se_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_se_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_se_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_se_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_se_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_se_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_conv3_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_conv3_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_conv3_1x1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_conv3_1x1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_pre_norm_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_pre_norm_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_pre_norm_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_pre_norm_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_pre_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_pre_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_pre_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_pre_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_conv1_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_conv1_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_conv2_kxk_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_conv2_kxk_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_se_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_se_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_se_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_se_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_se_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_se_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_se_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_se_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_conv3_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_conv3_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_conv3_1x1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_8_modules_conv3_1x1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_pre_norm_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_pre_norm_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_pre_norm_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_pre_norm_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_pre_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_pre_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_pre_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_pre_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_conv1_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_conv1_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_conv2_kxk_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_conv2_kxk_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_se_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_se_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_se_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_se_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_se_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_se_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_se_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_se_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_conv3_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_conv3_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_conv3_1x1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_9_modules_conv3_1x1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_pre_norm_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_pre_norm_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_pre_norm_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_pre_norm_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_pre_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_pre_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_pre_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_pre_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_conv1_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_conv1_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_conv2_kxk_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_conv2_kxk_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_se_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_se_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_se_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_se_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_se_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_se_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_se_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_se_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_conv3_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_conv3_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_conv3_1x1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_10_modules_conv3_1x1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_pre_norm_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_pre_norm_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_pre_norm_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_pre_norm_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_pre_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_pre_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_pre_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_pre_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_conv1_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_conv1_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_conv2_kxk_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_conv2_kxk_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_se_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_se_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_se_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_se_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_se_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_se_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_se_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_se_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_conv3_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_conv3_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_conv3_1x1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_11_modules_conv3_1x1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_27_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_27_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_27_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_27_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
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
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_stem_modules_conv1_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_stem_modules_norm1_buffers_running_mean_,
            l_self_modules_stem_modules_norm1_buffers_running_var_,
            l_self_modules_stem_modules_norm1_parameters_weight_,
            l_self_modules_stem_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_self_modules_stem_modules_norm1_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_norm1_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_norm1_parameters_weight_
        ) = l_self_modules_stem_modules_norm1_parameters_bias_ = None
        x_2 = torch._C._nn.gelu(x_1)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_stem_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_stem_modules_conv2_parameters_weight_ = None
        x_4 = torch._C._nn.avg_pool2d(x_3, 2, 2, 0, False, True, None)
        x_5 = torch.conv2d(
            x_4,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_4 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_ = (None)
        x_6 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_3 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_parameters_bias_ = (None)
        x_7 = torch.conv2d(
            x_6,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_6 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_ = (None)
        x_8 = torch.nn.functional.batch_norm(
            x_7,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_7 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        x_9 = torch._C._nn.gelu(x_8)
        x_8 = None
        x_10 = torch.conv2d(
            x_9,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1024,
        )
        x_9 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_ = (None)
        x_11 = torch.nn.functional.batch_norm(
            x_10,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_10 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_12 = torch._C._nn.gelu(x_11)
        x_11 = None
        x_se = x_12.mean((2, 3), keepdim=True)
        x_se_1 = torch.conv2d(
            x_se,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc1_parameters_bias_ = (None)
        x_se_2 = torch.nn.functional.silu(x_se_1, inplace=True)
        x_se_1 = None
        x_se_3 = torch.conv2d(
            x_se_2,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_2 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc2_parameters_bias_ = (None)
        sigmoid = x_se_3.sigmoid()
        x_se_3 = None
        x_13 = x_12 * sigmoid
        x_12 = sigmoid = None
        x_14 = torch.conv2d(
            x_13,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_13 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_bias_ = (None)
        x_15 = x_14 + x_5
        x_14 = x_5 = None
        x_16 = torch.nn.functional.batch_norm(
            x_15,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_bias_ = (None)
        x_17 = torch.conv2d(
            x_16,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_16 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_ = (None)
        x_18 = torch.nn.functional.batch_norm(
            x_17,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_17 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_19 = torch._C._nn.gelu(x_18)
        x_18 = None
        x_20 = torch.conv2d(
            x_19,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1024,
        )
        x_19 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_ = (None)
        x_21 = torch.nn.functional.batch_norm(
            x_20,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_20 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_22 = torch._C._nn.gelu(x_21)
        x_21 = None
        x_se_4 = x_22.mean((2, 3), keepdim=True)
        x_se_5 = torch.conv2d(
            x_se_4,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_4 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc1_parameters_bias_ = (None)
        x_se_6 = torch.nn.functional.silu(x_se_5, inplace=True)
        x_se_5 = None
        x_se_7 = torch.conv2d(
            x_se_6,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_6 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc2_parameters_bias_ = (None)
        sigmoid_1 = x_se_7.sigmoid()
        x_se_7 = None
        x_23 = x_22 * sigmoid_1
        x_22 = sigmoid_1 = None
        x_24 = torch.conv2d(
            x_23,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_23 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_bias_ = (None)
        x_25 = x_24 + x_15
        x_24 = x_15 = None
        x_26 = torch._C._nn.avg_pool2d(x_25, 2, 2, 0, False, True, None)
        x_27 = torch.conv2d(
            x_26,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_26 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_ = (None)
        x_28 = torch.nn.functional.batch_norm(
            x_25,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_25 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_bias_ = (None)
        x_29 = torch.conv2d(
            x_28,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_28 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_ = (None)
        x_30 = torch.nn.functional.batch_norm(
            x_29,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_29 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        x_31 = torch._C._nn.gelu(x_30)
        x_30 = None
        x_32 = torch.conv2d(
            x_31,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            2048,
        )
        x_31 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_ = (None)
        x_33 = torch.nn.functional.batch_norm(
            x_32,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_32 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_34 = torch._C._nn.gelu(x_33)
        x_33 = None
        x_se_8 = x_34.mean((2, 3), keepdim=True)
        x_se_9 = torch.conv2d(
            x_se_8,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_8 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc1_parameters_bias_ = (None)
        x_se_10 = torch.nn.functional.silu(x_se_9, inplace=True)
        x_se_9 = None
        x_se_11 = torch.conv2d(
            x_se_10,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_10 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc2_parameters_bias_ = (None)
        sigmoid_2 = x_se_11.sigmoid()
        x_se_11 = None
        x_35 = x_34 * sigmoid_2
        x_34 = sigmoid_2 = None
        x_36 = torch.conv2d(
            x_35,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_35 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_bias_ = (None)
        x_37 = x_36 + x_27
        x_36 = x_27 = None
        x_38 = torch.nn.functional.batch_norm(
            x_37,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_bias_ = (None)
        x_39 = torch.conv2d(
            x_38,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_38 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_ = (None)
        x_40 = torch.nn.functional.batch_norm(
            x_39,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_39 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_41 = torch._C._nn.gelu(x_40)
        x_40 = None
        x_42 = torch.conv2d(
            x_41,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2048,
        )
        x_41 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_ = (None)
        x_43 = torch.nn.functional.batch_norm(
            x_42,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_42 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_44 = torch._C._nn.gelu(x_43)
        x_43 = None
        x_se_12 = x_44.mean((2, 3), keepdim=True)
        x_se_13 = torch.conv2d(
            x_se_12,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_12 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc1_parameters_bias_ = (None)
        x_se_14 = torch.nn.functional.silu(x_se_13, inplace=True)
        x_se_13 = None
        x_se_15 = torch.conv2d(
            x_se_14,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_14 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc2_parameters_bias_ = (None)
        sigmoid_3 = x_se_15.sigmoid()
        x_se_15 = None
        x_45 = x_44 * sigmoid_3
        x_44 = sigmoid_3 = None
        x_46 = torch.conv2d(
            x_45,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_45 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_bias_ = (None)
        x_47 = x_46 + x_37
        x_46 = x_37 = None
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_bias_ = (None)
        x_49 = torch.conv2d(
            x_48,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_48 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_1x1_parameters_weight_ = (None)
        x_50 = torch.nn.functional.batch_norm(
            x_49,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_49 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (None)
        x_51 = torch._C._nn.gelu(x_50)
        x_50 = None
        x_52 = torch.conv2d(
            x_51,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2048,
        )
        x_51 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_kxk_parameters_weight_ = (None)
        x_53 = torch.nn.functional.batch_norm(
            x_52,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_52 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (None)
        x_54 = torch._C._nn.gelu(x_53)
        x_53 = None
        x_se_16 = x_54.mean((2, 3), keepdim=True)
        x_se_17 = torch.conv2d(
            x_se_16,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_16 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc1_parameters_bias_ = (None)
        x_se_18 = torch.nn.functional.silu(x_se_17, inplace=True)
        x_se_17 = None
        x_se_19 = torch.conv2d(
            x_se_18,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_18 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc2_parameters_bias_ = (None)
        sigmoid_4 = x_se_19.sigmoid()
        x_se_19 = None
        x_55 = x_54 * sigmoid_4
        x_54 = sigmoid_4 = None
        x_56 = torch.conv2d(
            x_55,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_55 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_bias_ = (None)
        x_57 = x_56 + x_47
        x_56 = x_47 = None
        x_58 = torch.nn.functional.batch_norm(
            x_57,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_parameters_bias_ = (None)
        x_59 = torch.conv2d(
            x_58,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_58 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_1x1_parameters_weight_ = (None)
        x_60 = torch.nn.functional.batch_norm(
            x_59,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_59 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_bias_ = (None)
        x_61 = torch._C._nn.gelu(x_60)
        x_60 = None
        x_62 = torch.conv2d(
            x_61,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2048,
        )
        x_61 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_kxk_parameters_weight_ = (None)
        x_63 = torch.nn.functional.batch_norm(
            x_62,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_62 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_bias_ = (None)
        x_64 = torch._C._nn.gelu(x_63)
        x_63 = None
        x_se_20 = x_64.mean((2, 3), keepdim=True)
        x_se_21 = torch.conv2d(
            x_se_20,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_20 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc1_parameters_bias_ = (None)
        x_se_22 = torch.nn.functional.silu(x_se_21, inplace=True)
        x_se_21 = None
        x_se_23 = torch.conv2d(
            x_se_22,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_22 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc2_parameters_bias_ = (None)
        sigmoid_5 = x_se_23.sigmoid()
        x_se_23 = None
        x_65 = x_64 * sigmoid_5
        x_64 = sigmoid_5 = None
        x_66 = torch.conv2d(
            x_65,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_65 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_1x1_parameters_bias_ = (None)
        x_67 = x_66 + x_57
        x_66 = x_57 = None
        x_68 = torch.nn.functional.batch_norm(
            x_67,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_parameters_bias_ = (None)
        x_69 = torch.conv2d(
            x_68,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_68 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_1x1_parameters_weight_ = (None)
        x_70 = torch.nn.functional.batch_norm(
            x_69,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_69 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_bias_ = (None)
        x_71 = torch._C._nn.gelu(x_70)
        x_70 = None
        x_72 = torch.conv2d(
            x_71,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2048,
        )
        x_71 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_kxk_parameters_weight_ = (None)
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_72 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_bias_ = (None)
        x_74 = torch._C._nn.gelu(x_73)
        x_73 = None
        x_se_24 = x_74.mean((2, 3), keepdim=True)
        x_se_25 = torch.conv2d(
            x_se_24,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_24 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc1_parameters_bias_ = (None)
        x_se_26 = torch.nn.functional.silu(x_se_25, inplace=True)
        x_se_25 = None
        x_se_27 = torch.conv2d(
            x_se_26,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_26 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc2_parameters_bias_ = (None)
        sigmoid_6 = x_se_27.sigmoid()
        x_se_27 = None
        x_75 = x_74 * sigmoid_6
        x_74 = sigmoid_6 = None
        x_76 = torch.conv2d(
            x_75,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_75 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_1x1_parameters_bias_ = (None)
        x_77 = x_76 + x_67
        x_76 = x_67 = None
        x_78 = torch.nn.functional.batch_norm(
            x_77,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_parameters_bias_ = (None)
        x_79 = torch.conv2d(
            x_78,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_78 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_1x1_parameters_weight_ = (None)
        x_80 = torch.nn.functional.batch_norm(
            x_79,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_79 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_bias_ = (None)
        x_81 = torch._C._nn.gelu(x_80)
        x_80 = None
        x_82 = torch.conv2d(
            x_81,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2048,
        )
        x_81 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_kxk_parameters_weight_ = (None)
        x_83 = torch.nn.functional.batch_norm(
            x_82,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_82 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_bias_ = (None)
        x_84 = torch._C._nn.gelu(x_83)
        x_83 = None
        x_se_28 = x_84.mean((2, 3), keepdim=True)
        x_se_29 = torch.conv2d(
            x_se_28,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_28 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc1_parameters_bias_ = (None)
        x_se_30 = torch.nn.functional.silu(x_se_29, inplace=True)
        x_se_29 = None
        x_se_31 = torch.conv2d(
            x_se_30,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_30 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc2_parameters_bias_ = (None)
        sigmoid_7 = x_se_31.sigmoid()
        x_se_31 = None
        x_85 = x_84 * sigmoid_7
        x_84 = sigmoid_7 = None
        x_86 = torch.conv2d(
            x_85,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_85 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_1x1_parameters_bias_ = (None)
        x_87 = x_86 + x_77
        x_86 = x_77 = None
        x_88 = torch.nn.functional.batch_norm(
            x_87,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_pre_norm_parameters_bias_ = (None)
        x_89 = torch.conv2d(
            x_88,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_88 = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_1x1_parameters_weight_ = (None)
        x_90 = torch.nn.functional.batch_norm(
            x_89,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_89 = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_bias_ = (None)
        x_91 = torch._C._nn.gelu(x_90)
        x_90 = None
        x_92 = torch.conv2d(
            x_91,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2048,
        )
        x_91 = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_kxk_parameters_weight_ = (None)
        x_93 = torch.nn.functional.batch_norm(
            x_92,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_92 = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_bias_ = (None)
        x_94 = torch._C._nn.gelu(x_93)
        x_93 = None
        x_se_32 = x_94.mean((2, 3), keepdim=True)
        x_se_33 = torch.conv2d(
            x_se_32,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_32 = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_se_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_se_modules_fc1_parameters_bias_ = (None)
        x_se_34 = torch.nn.functional.silu(x_se_33, inplace=True)
        x_se_33 = None
        x_se_35 = torch.conv2d(
            x_se_34,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_34 = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_se_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_se_modules_fc2_parameters_bias_ = (None)
        sigmoid_8 = x_se_35.sigmoid()
        x_se_35 = None
        x_95 = x_94 * sigmoid_8
        x_94 = sigmoid_8 = None
        x_96 = torch.conv2d(
            x_95,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_95 = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv3_1x1_parameters_bias_ = (None)
        x_97 = x_96 + x_87
        x_96 = x_87 = None
        x_98 = torch.nn.functional.batch_norm(
            x_97,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_pre_norm_parameters_bias_ = (None)
        x_99 = torch.conv2d(
            x_98,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_98 = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_conv1_1x1_parameters_weight_ = (None)
        x_100 = torch.nn.functional.batch_norm(
            x_99,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_99 = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_bias_ = (None)
        x_101 = torch._C._nn.gelu(x_100)
        x_100 = None
        x_102 = torch.conv2d(
            x_101,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2048,
        )
        x_101 = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_conv2_kxk_parameters_weight_ = (None)
        x_103 = torch.nn.functional.batch_norm(
            x_102,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_102 = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_bias_ = (None)
        x_104 = torch._C._nn.gelu(x_103)
        x_103 = None
        x_se_36 = x_104.mean((2, 3), keepdim=True)
        x_se_37 = torch.conv2d(
            x_se_36,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_36 = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_se_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_se_modules_fc1_parameters_bias_ = (None)
        x_se_38 = torch.nn.functional.silu(x_se_37, inplace=True)
        x_se_37 = None
        x_se_39 = torch.conv2d(
            x_se_38,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_38 = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_se_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_se_modules_fc2_parameters_bias_ = (None)
        sigmoid_9 = x_se_39.sigmoid()
        x_se_39 = None
        x_105 = x_104 * sigmoid_9
        x_104 = sigmoid_9 = None
        x_106 = torch.conv2d(
            x_105,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_105 = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_conv3_1x1_parameters_bias_ = (None)
        x_107 = x_106 + x_97
        x_106 = x_97 = None
        x_108 = torch.nn.functional.batch_norm(
            x_107,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_pre_norm_parameters_bias_ = (None)
        x_109 = torch.conv2d(
            x_108,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_108 = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_conv1_1x1_parameters_weight_ = (None)
        x_110 = torch.nn.functional.batch_norm(
            x_109,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_109 = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_parameters_bias_ = (None)
        x_111 = torch._C._nn.gelu(x_110)
        x_110 = None
        x_112 = torch.conv2d(
            x_111,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2048,
        )
        x_111 = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_conv2_kxk_parameters_weight_ = (None)
        x_113 = torch.nn.functional.batch_norm(
            x_112,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_112 = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_parameters_bias_ = (None)
        x_114 = torch._C._nn.gelu(x_113)
        x_113 = None
        x_se_40 = x_114.mean((2, 3), keepdim=True)
        x_se_41 = torch.conv2d(
            x_se_40,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_40 = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_se_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_se_modules_fc1_parameters_bias_ = (None)
        x_se_42 = torch.nn.functional.silu(x_se_41, inplace=True)
        x_se_41 = None
        x_se_43 = torch.conv2d(
            x_se_42,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_42 = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_se_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_se_modules_fc2_parameters_bias_ = (None)
        sigmoid_10 = x_se_43.sigmoid()
        x_se_43 = None
        x_115 = x_114 * sigmoid_10
        x_114 = sigmoid_10 = None
        x_116 = torch.conv2d(
            x_115,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_115 = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_conv3_1x1_parameters_bias_ = (None)
        x_117 = x_116 + x_107
        x_116 = x_107 = None
        x_118 = torch.nn.functional.batch_norm(
            x_117,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_pre_norm_parameters_bias_ = (None)
        x_119 = torch.conv2d(
            x_118,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_118 = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_conv1_1x1_parameters_weight_ = (None)
        x_120 = torch.nn.functional.batch_norm(
            x_119,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_119 = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_parameters_bias_ = (None)
        x_121 = torch._C._nn.gelu(x_120)
        x_120 = None
        x_122 = torch.conv2d(
            x_121,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2048,
        )
        x_121 = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_conv2_kxk_parameters_weight_ = (None)
        x_123 = torch.nn.functional.batch_norm(
            x_122,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_122 = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_parameters_bias_ = (None)
        x_124 = torch._C._nn.gelu(x_123)
        x_123 = None
        x_se_44 = x_124.mean((2, 3), keepdim=True)
        x_se_45 = torch.conv2d(
            x_se_44,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_44 = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_se_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_se_modules_fc1_parameters_bias_ = (None)
        x_se_46 = torch.nn.functional.silu(x_se_45, inplace=True)
        x_se_45 = None
        x_se_47 = torch.conv2d(
            x_se_46,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_46 = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_se_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_se_modules_fc2_parameters_bias_ = (None)
        sigmoid_11 = x_se_47.sigmoid()
        x_se_47 = None
        x_125 = x_124 * sigmoid_11
        x_124 = sigmoid_11 = None
        x_126 = torch.conv2d(
            x_125,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_125 = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_conv3_1x1_parameters_bias_ = (None)
        x_127 = x_126 + x_117
        x_126 = x_117 = None
        x_128 = torch.nn.functional.batch_norm(
            x_127,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_pre_norm_parameters_bias_ = (None)
        x_129 = torch.conv2d(
            x_128,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_128 = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_conv1_1x1_parameters_weight_ = (None)
        x_130 = torch.nn.functional.batch_norm(
            x_129,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_129 = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_parameters_bias_ = (None)
        x_131 = torch._C._nn.gelu(x_130)
        x_130 = None
        x_132 = torch.conv2d(
            x_131,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2048,
        )
        x_131 = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_conv2_kxk_parameters_weight_ = (None)
        x_133 = torch.nn.functional.batch_norm(
            x_132,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_132 = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_parameters_bias_ = (None)
        x_134 = torch._C._nn.gelu(x_133)
        x_133 = None
        x_se_48 = x_134.mean((2, 3), keepdim=True)
        x_se_49 = torch.conv2d(
            x_se_48,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_48 = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_se_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_se_modules_fc1_parameters_bias_ = (None)
        x_se_50 = torch.nn.functional.silu(x_se_49, inplace=True)
        x_se_49 = None
        x_se_51 = torch.conv2d(
            x_se_50,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_50 = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_se_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_se_modules_fc2_parameters_bias_ = (None)
        sigmoid_12 = x_se_51.sigmoid()
        x_se_51 = None
        x_135 = x_134 * sigmoid_12
        x_134 = sigmoid_12 = None
        x_136 = torch.conv2d(
            x_135,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_135 = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_conv3_1x1_parameters_bias_ = (None)
        x_137 = x_136 + x_127
        x_136 = x_127 = None
        x_138 = torch.nn.functional.batch_norm(
            x_137,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_pre_norm_parameters_bias_ = (None)
        x_139 = torch.conv2d(
            x_138,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_138 = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_conv1_1x1_parameters_weight_ = (None)
        x_140 = torch.nn.functional.batch_norm(
            x_139,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_139 = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_parameters_bias_ = (None)
        x_141 = torch._C._nn.gelu(x_140)
        x_140 = None
        x_142 = torch.conv2d(
            x_141,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2048,
        )
        x_141 = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_conv2_kxk_parameters_weight_ = (None)
        x_143 = torch.nn.functional.batch_norm(
            x_142,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_142 = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_parameters_bias_ = (None)
        x_144 = torch._C._nn.gelu(x_143)
        x_143 = None
        x_se_52 = x_144.mean((2, 3), keepdim=True)
        x_se_53 = torch.conv2d(
            x_se_52,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_52 = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_se_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_se_modules_fc1_parameters_bias_ = (None)
        x_se_54 = torch.nn.functional.silu(x_se_53, inplace=True)
        x_se_53 = None
        x_se_55 = torch.conv2d(
            x_se_54,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_54 = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_se_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_se_modules_fc2_parameters_bias_ = (None)
        sigmoid_13 = x_se_55.sigmoid()
        x_se_55 = None
        x_145 = x_144 * sigmoid_13
        x_144 = sigmoid_13 = None
        x_146 = torch.conv2d(
            x_145,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_145 = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_conv3_1x1_parameters_bias_ = (None)
        x_147 = x_146 + x_137
        x_146 = x_137 = None
        x_148 = torch._C._nn.avg_pool2d(x_147, 2, 2, 0, False, True, None)
        x_149 = torch.conv2d(
            x_148,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_148 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_ = (None)
        x_150 = x_147.permute(0, 2, 3, 1)
        x_147 = None
        x_151 = torch.nn.functional.layer_norm(
            x_150,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_,
            1e-06,
        )
        x_150 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_ = (None)
        x_152 = x_151.permute(0, 3, 1, 2)
        x_151 = None
        x_153 = torch._C._nn.avg_pool2d(x_152, 2, 2, 0, False, True, None)
        x_152 = None
        conv2d_75 = torch.conv2d(
            x_153,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_153 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        view = conv2d_75.view(1, 40, 96, -1)
        conv2d_75 = None
        chunk = view.chunk(3, dim=2)
        view = None
        q = chunk[0]
        k = chunk[1]
        v = chunk[2]
        chunk = None
        relative_position_bias = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_1 = relative_position_bias.view((196, 196, 40))
        relative_position_bias = None
        relative_position_bias_1 = view_1.permute(2, 0, 1)
        view_1 = None
        unsqueeze = relative_position_bias_1.unsqueeze(0)
        relative_position_bias_1 = None
        attn_bias = unsqueeze.contiguous()
        unsqueeze = None
        transpose = q.transpose(-1, -2)
        q = None
        contiguous_1 = transpose.contiguous()
        transpose = None
        transpose_1 = k.transpose(-1, -2)
        k = None
        contiguous_2 = transpose_1.contiguous()
        transpose_1 = None
        transpose_2 = v.transpose(-1, -2)
        v = None
        contiguous_3 = transpose_2.contiguous()
        transpose_2 = None
        scaled_dot_product_attention = torch._C._nn.scaled_dot_product_attention(
            contiguous_1, contiguous_2, contiguous_3, attn_mask=attn_bias, dropout_p=0.0
        )
        contiguous_1 = contiguous_2 = contiguous_3 = attn_bias = None
        transpose_3 = scaled_dot_product_attention.transpose(-1, -2)
        scaled_dot_product_attention = None
        x_154 = transpose_3.reshape(1, -1, 14, 14)
        transpose_3 = None
        x_155 = torch.conv2d(
            x_154,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_154 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_156 = torch.nn.functional.dropout(x_155, 0.0, False, False)
        x_155 = None
        x_157 = x_149 + x_156
        x_149 = x_156 = None
        x_158 = x_157.permute(0, 2, 3, 1)
        x_159 = torch.nn.functional.layer_norm(
            x_158,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_158 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_160 = x_159.permute(0, 3, 1, 2)
        x_159 = None
        x_161 = torch.conv2d(
            x_160,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_160 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_162 = torch._C._nn.gelu(x_161)
        x_161 = None
        x_163 = torch.nn.functional.dropout(x_162, 0.0, False, False)
        x_162 = None
        x_164 = torch.conv2d(
            x_163,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_163 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_165 = x_157 + x_164
        x_157 = x_164 = None
        x_166 = x_165.permute(0, 2, 3, 1)
        x_167 = torch.nn.functional.layer_norm(
            x_166,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_166 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_168 = x_167.permute(0, 3, 1, 2)
        x_167 = None
        conv2d_79 = torch.conv2d(
            x_168,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_168 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_2 = conv2d_79.view(1, 40, 96, -1)
        conv2d_79 = None
        chunk_1 = view_2.chunk(3, dim=2)
        view_2 = None
        q_1 = chunk_1[0]
        k_1 = chunk_1[1]
        v_1 = chunk_1[2]
        chunk_1 = None
        relative_position_bias_2 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_3 = relative_position_bias_2.view((196, 196, 40))
        relative_position_bias_2 = None
        relative_position_bias_3 = view_3.permute(2, 0, 1)
        view_3 = None
        unsqueeze_1 = relative_position_bias_3.unsqueeze(0)
        relative_position_bias_3 = None
        attn_bias_1 = unsqueeze_1.contiguous()
        unsqueeze_1 = None
        transpose_4 = q_1.transpose(-1, -2)
        q_1 = None
        contiguous_5 = transpose_4.contiguous()
        transpose_4 = None
        transpose_5 = k_1.transpose(-1, -2)
        k_1 = None
        contiguous_6 = transpose_5.contiguous()
        transpose_5 = None
        transpose_6 = v_1.transpose(-1, -2)
        v_1 = None
        contiguous_7 = transpose_6.contiguous()
        transpose_6 = None
        scaled_dot_product_attention_1 = torch._C._nn.scaled_dot_product_attention(
            contiguous_5,
            contiguous_6,
            contiguous_7,
            attn_mask=attn_bias_1,
            dropout_p=0.0,
        )
        contiguous_5 = contiguous_6 = contiguous_7 = attn_bias_1 = None
        transpose_7 = scaled_dot_product_attention_1.transpose(-1, -2)
        scaled_dot_product_attention_1 = None
        x_169 = transpose_7.reshape(1, -1, 14, 14)
        transpose_7 = None
        x_170 = torch.conv2d(
            x_169,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_169 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_171 = torch.nn.functional.dropout(x_170, 0.0, False, False)
        x_170 = None
        x_172 = x_165 + x_171
        x_165 = x_171 = None
        x_173 = x_172.permute(0, 2, 3, 1)
        x_174 = torch.nn.functional.layer_norm(
            x_173,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_173 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_175 = x_174.permute(0, 3, 1, 2)
        x_174 = None
        x_176 = torch.conv2d(
            x_175,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_175 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_177 = torch._C._nn.gelu(x_176)
        x_176 = None
        x_178 = torch.nn.functional.dropout(x_177, 0.0, False, False)
        x_177 = None
        x_179 = torch.conv2d(
            x_178,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_178 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_180 = x_172 + x_179
        x_172 = x_179 = None
        x_181 = x_180.permute(0, 2, 3, 1)
        x_182 = torch.nn.functional.layer_norm(
            x_181,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_181 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (None)
        x_183 = x_182.permute(0, 3, 1, 2)
        x_182 = None
        conv2d_83 = torch.conv2d(
            x_183,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_183 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_4 = conv2d_83.view(1, 40, 96, -1)
        conv2d_83 = None
        chunk_2 = view_4.chunk(3, dim=2)
        view_4 = None
        q_2 = chunk_2[0]
        k_2 = chunk_2[1]
        v_2 = chunk_2[2]
        chunk_2 = None
        relative_position_bias_4 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_5 = relative_position_bias_4.view((196, 196, 40))
        relative_position_bias_4 = None
        relative_position_bias_5 = view_5.permute(2, 0, 1)
        view_5 = None
        unsqueeze_2 = relative_position_bias_5.unsqueeze(0)
        relative_position_bias_5 = None
        attn_bias_2 = unsqueeze_2.contiguous()
        unsqueeze_2 = None
        transpose_8 = q_2.transpose(-1, -2)
        q_2 = None
        contiguous_9 = transpose_8.contiguous()
        transpose_8 = None
        transpose_9 = k_2.transpose(-1, -2)
        k_2 = None
        contiguous_10 = transpose_9.contiguous()
        transpose_9 = None
        transpose_10 = v_2.transpose(-1, -2)
        v_2 = None
        contiguous_11 = transpose_10.contiguous()
        transpose_10 = None
        scaled_dot_product_attention_2 = torch._C._nn.scaled_dot_product_attention(
            contiguous_9,
            contiguous_10,
            contiguous_11,
            attn_mask=attn_bias_2,
            dropout_p=0.0,
        )
        contiguous_9 = contiguous_10 = contiguous_11 = attn_bias_2 = None
        transpose_11 = scaled_dot_product_attention_2.transpose(-1, -2)
        scaled_dot_product_attention_2 = None
        x_184 = transpose_11.reshape(1, -1, 14, 14)
        transpose_11 = None
        x_185 = torch.conv2d(
            x_184,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_184 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_ = (None)
        x_186 = torch.nn.functional.dropout(x_185, 0.0, False, False)
        x_185 = None
        x_187 = x_180 + x_186
        x_180 = x_186 = None
        x_188 = x_187.permute(0, 2, 3, 1)
        x_189 = torch.nn.functional.layer_norm(
            x_188,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_188 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (None)
        x_190 = x_189.permute(0, 3, 1, 2)
        x_189 = None
        x_191 = torch.conv2d(
            x_190,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_190 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_192 = torch._C._nn.gelu(x_191)
        x_191 = None
        x_193 = torch.nn.functional.dropout(x_192, 0.0, False, False)
        x_192 = None
        x_194 = torch.conv2d(
            x_193,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_193 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_195 = x_187 + x_194
        x_187 = x_194 = None
        x_196 = x_195.permute(0, 2, 3, 1)
        x_197 = torch.nn.functional.layer_norm(
            x_196,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_196 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_ = (None)
        x_198 = x_197.permute(0, 3, 1, 2)
        x_197 = None
        conv2d_87 = torch.conv2d(
            x_198,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_198 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_6 = conv2d_87.view(1, 40, 96, -1)
        conv2d_87 = None
        chunk_3 = view_6.chunk(3, dim=2)
        view_6 = None
        q_3 = chunk_3[0]
        k_3 = chunk_3[1]
        v_3 = chunk_3[2]
        chunk_3 = None
        relative_position_bias_6 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_7 = relative_position_bias_6.view((196, 196, 40))
        relative_position_bias_6 = None
        relative_position_bias_7 = view_7.permute(2, 0, 1)
        view_7 = None
        unsqueeze_3 = relative_position_bias_7.unsqueeze(0)
        relative_position_bias_7 = None
        attn_bias_3 = unsqueeze_3.contiguous()
        unsqueeze_3 = None
        transpose_12 = q_3.transpose(-1, -2)
        q_3 = None
        contiguous_13 = transpose_12.contiguous()
        transpose_12 = None
        transpose_13 = k_3.transpose(-1, -2)
        k_3 = None
        contiguous_14 = transpose_13.contiguous()
        transpose_13 = None
        transpose_14 = v_3.transpose(-1, -2)
        v_3 = None
        contiguous_15 = transpose_14.contiguous()
        transpose_14 = None
        scaled_dot_product_attention_3 = torch._C._nn.scaled_dot_product_attention(
            contiguous_13,
            contiguous_14,
            contiguous_15,
            attn_mask=attn_bias_3,
            dropout_p=0.0,
        )
        contiguous_13 = contiguous_14 = contiguous_15 = attn_bias_3 = None
        transpose_15 = scaled_dot_product_attention_3.transpose(-1, -2)
        scaled_dot_product_attention_3 = None
        x_199 = transpose_15.reshape(1, -1, 14, 14)
        transpose_15 = None
        x_200 = torch.conv2d(
            x_199,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_199 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_ = (None)
        x_201 = torch.nn.functional.dropout(x_200, 0.0, False, False)
        x_200 = None
        x_202 = x_195 + x_201
        x_195 = x_201 = None
        x_203 = x_202.permute(0, 2, 3, 1)
        x_204 = torch.nn.functional.layer_norm(
            x_203,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_203 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_ = (None)
        x_205 = x_204.permute(0, 3, 1, 2)
        x_204 = None
        x_206 = torch.conv2d(
            x_205,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_205 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_207 = torch._C._nn.gelu(x_206)
        x_206 = None
        x_208 = torch.nn.functional.dropout(x_207, 0.0, False, False)
        x_207 = None
        x_209 = torch.conv2d(
            x_208,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_208 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_210 = x_202 + x_209
        x_202 = x_209 = None
        x_211 = x_210.permute(0, 2, 3, 1)
        x_212 = torch.nn.functional.layer_norm(
            x_211,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_211 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_ = (None)
        x_213 = x_212.permute(0, 3, 1, 2)
        x_212 = None
        conv2d_91 = torch.conv2d(
            x_213,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_213 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_8 = conv2d_91.view(1, 40, 96, -1)
        conv2d_91 = None
        chunk_4 = view_8.chunk(3, dim=2)
        view_8 = None
        q_4 = chunk_4[0]
        k_4 = chunk_4[1]
        v_4 = chunk_4[2]
        chunk_4 = None
        relative_position_bias_8 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_9 = relative_position_bias_8.view((196, 196, 40))
        relative_position_bias_8 = None
        relative_position_bias_9 = view_9.permute(2, 0, 1)
        view_9 = None
        unsqueeze_4 = relative_position_bias_9.unsqueeze(0)
        relative_position_bias_9 = None
        attn_bias_4 = unsqueeze_4.contiguous()
        unsqueeze_4 = None
        transpose_16 = q_4.transpose(-1, -2)
        q_4 = None
        contiguous_17 = transpose_16.contiguous()
        transpose_16 = None
        transpose_17 = k_4.transpose(-1, -2)
        k_4 = None
        contiguous_18 = transpose_17.contiguous()
        transpose_17 = None
        transpose_18 = v_4.transpose(-1, -2)
        v_4 = None
        contiguous_19 = transpose_18.contiguous()
        transpose_18 = None
        scaled_dot_product_attention_4 = torch._C._nn.scaled_dot_product_attention(
            contiguous_17,
            contiguous_18,
            contiguous_19,
            attn_mask=attn_bias_4,
            dropout_p=0.0,
        )
        contiguous_17 = contiguous_18 = contiguous_19 = attn_bias_4 = None
        transpose_19 = scaled_dot_product_attention_4.transpose(-1, -2)
        scaled_dot_product_attention_4 = None
        x_214 = transpose_19.reshape(1, -1, 14, 14)
        transpose_19 = None
        x_215 = torch.conv2d(
            x_214,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_214 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_ = (None)
        x_216 = torch.nn.functional.dropout(x_215, 0.0, False, False)
        x_215 = None
        x_217 = x_210 + x_216
        x_210 = x_216 = None
        x_218 = x_217.permute(0, 2, 3, 1)
        x_219 = torch.nn.functional.layer_norm(
            x_218,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_218 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_ = (None)
        x_220 = x_219.permute(0, 3, 1, 2)
        x_219 = None
        x_221 = torch.conv2d(
            x_220,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_220 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_222 = torch._C._nn.gelu(x_221)
        x_221 = None
        x_223 = torch.nn.functional.dropout(x_222, 0.0, False, False)
        x_222 = None
        x_224 = torch.conv2d(
            x_223,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_223 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_225 = x_217 + x_224
        x_217 = x_224 = None
        x_226 = x_225.permute(0, 2, 3, 1)
        x_227 = torch.nn.functional.layer_norm(
            x_226,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_226 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_ = (None)
        x_228 = x_227.permute(0, 3, 1, 2)
        x_227 = None
        conv2d_95 = torch.conv2d(
            x_228,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_228 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_10 = conv2d_95.view(1, 40, 96, -1)
        conv2d_95 = None
        chunk_5 = view_10.chunk(3, dim=2)
        view_10 = None
        q_5 = chunk_5[0]
        k_5 = chunk_5[1]
        v_5 = chunk_5[2]
        chunk_5 = None
        relative_position_bias_10 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_11 = relative_position_bias_10.view((196, 196, 40))
        relative_position_bias_10 = None
        relative_position_bias_11 = view_11.permute(2, 0, 1)
        view_11 = None
        unsqueeze_5 = relative_position_bias_11.unsqueeze(0)
        relative_position_bias_11 = None
        attn_bias_5 = unsqueeze_5.contiguous()
        unsqueeze_5 = None
        transpose_20 = q_5.transpose(-1, -2)
        q_5 = None
        contiguous_21 = transpose_20.contiguous()
        transpose_20 = None
        transpose_21 = k_5.transpose(-1, -2)
        k_5 = None
        contiguous_22 = transpose_21.contiguous()
        transpose_21 = None
        transpose_22 = v_5.transpose(-1, -2)
        v_5 = None
        contiguous_23 = transpose_22.contiguous()
        transpose_22 = None
        scaled_dot_product_attention_5 = torch._C._nn.scaled_dot_product_attention(
            contiguous_21,
            contiguous_22,
            contiguous_23,
            attn_mask=attn_bias_5,
            dropout_p=0.0,
        )
        contiguous_21 = contiguous_22 = contiguous_23 = attn_bias_5 = None
        transpose_23 = scaled_dot_product_attention_5.transpose(-1, -2)
        scaled_dot_product_attention_5 = None
        x_229 = transpose_23.reshape(1, -1, 14, 14)
        transpose_23 = None
        x_230 = torch.conv2d(
            x_229,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_229 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_ = (None)
        x_231 = torch.nn.functional.dropout(x_230, 0.0, False, False)
        x_230 = None
        x_232 = x_225 + x_231
        x_225 = x_231 = None
        x_233 = x_232.permute(0, 2, 3, 1)
        x_234 = torch.nn.functional.layer_norm(
            x_233,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_233 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_ = (None)
        x_235 = x_234.permute(0, 3, 1, 2)
        x_234 = None
        x_236 = torch.conv2d(
            x_235,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_235 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_237 = torch._C._nn.gelu(x_236)
        x_236 = None
        x_238 = torch.nn.functional.dropout(x_237, 0.0, False, False)
        x_237 = None
        x_239 = torch.conv2d(
            x_238,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_238 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_240 = x_232 + x_239
        x_232 = x_239 = None
        x_241 = x_240.permute(0, 2, 3, 1)
        x_242 = torch.nn.functional.layer_norm(
            x_241,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_241 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_ = (None)
        x_243 = x_242.permute(0, 3, 1, 2)
        x_242 = None
        conv2d_99 = torch.conv2d(
            x_243,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_243 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_12 = conv2d_99.view(1, 40, 96, -1)
        conv2d_99 = None
        chunk_6 = view_12.chunk(3, dim=2)
        view_12 = None
        q_6 = chunk_6[0]
        k_6 = chunk_6[1]
        v_6 = chunk_6[2]
        chunk_6 = None
        relative_position_bias_12 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_13 = relative_position_bias_12.view((196, 196, 40))
        relative_position_bias_12 = None
        relative_position_bias_13 = view_13.permute(2, 0, 1)
        view_13 = None
        unsqueeze_6 = relative_position_bias_13.unsqueeze(0)
        relative_position_bias_13 = None
        attn_bias_6 = unsqueeze_6.contiguous()
        unsqueeze_6 = None
        transpose_24 = q_6.transpose(-1, -2)
        q_6 = None
        contiguous_25 = transpose_24.contiguous()
        transpose_24 = None
        transpose_25 = k_6.transpose(-1, -2)
        k_6 = None
        contiguous_26 = transpose_25.contiguous()
        transpose_25 = None
        transpose_26 = v_6.transpose(-1, -2)
        v_6 = None
        contiguous_27 = transpose_26.contiguous()
        transpose_26 = None
        scaled_dot_product_attention_6 = torch._C._nn.scaled_dot_product_attention(
            contiguous_25,
            contiguous_26,
            contiguous_27,
            attn_mask=attn_bias_6,
            dropout_p=0.0,
        )
        contiguous_25 = contiguous_26 = contiguous_27 = attn_bias_6 = None
        transpose_27 = scaled_dot_product_attention_6.transpose(-1, -2)
        scaled_dot_product_attention_6 = None
        x_244 = transpose_27.reshape(1, -1, 14, 14)
        transpose_27 = None
        x_245 = torch.conv2d(
            x_244,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_244 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_ = (None)
        x_246 = torch.nn.functional.dropout(x_245, 0.0, False, False)
        x_245 = None
        x_247 = x_240 + x_246
        x_240 = x_246 = None
        x_248 = x_247.permute(0, 2, 3, 1)
        x_249 = torch.nn.functional.layer_norm(
            x_248,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_248 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_ = (None)
        x_250 = x_249.permute(0, 3, 1, 2)
        x_249 = None
        x_251 = torch.conv2d(
            x_250,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_250 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_252 = torch._C._nn.gelu(x_251)
        x_251 = None
        x_253 = torch.nn.functional.dropout(x_252, 0.0, False, False)
        x_252 = None
        x_254 = torch.conv2d(
            x_253,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_253 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_255 = x_247 + x_254
        x_247 = x_254 = None
        x_256 = x_255.permute(0, 2, 3, 1)
        x_257 = torch.nn.functional.layer_norm(
            x_256,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_256 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_ = (None)
        x_258 = x_257.permute(0, 3, 1, 2)
        x_257 = None
        conv2d_103 = torch.conv2d(
            x_258,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_258 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_14 = conv2d_103.view(1, 40, 96, -1)
        conv2d_103 = None
        chunk_7 = view_14.chunk(3, dim=2)
        view_14 = None
        q_7 = chunk_7[0]
        k_7 = chunk_7[1]
        v_7 = chunk_7[2]
        chunk_7 = None
        relative_position_bias_14 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_15 = relative_position_bias_14.view((196, 196, 40))
        relative_position_bias_14 = None
        relative_position_bias_15 = view_15.permute(2, 0, 1)
        view_15 = None
        unsqueeze_7 = relative_position_bias_15.unsqueeze(0)
        relative_position_bias_15 = None
        attn_bias_7 = unsqueeze_7.contiguous()
        unsqueeze_7 = None
        transpose_28 = q_7.transpose(-1, -2)
        q_7 = None
        contiguous_29 = transpose_28.contiguous()
        transpose_28 = None
        transpose_29 = k_7.transpose(-1, -2)
        k_7 = None
        contiguous_30 = transpose_29.contiguous()
        transpose_29 = None
        transpose_30 = v_7.transpose(-1, -2)
        v_7 = None
        contiguous_31 = transpose_30.contiguous()
        transpose_30 = None
        scaled_dot_product_attention_7 = torch._C._nn.scaled_dot_product_attention(
            contiguous_29,
            contiguous_30,
            contiguous_31,
            attn_mask=attn_bias_7,
            dropout_p=0.0,
        )
        contiguous_29 = contiguous_30 = contiguous_31 = attn_bias_7 = None
        transpose_31 = scaled_dot_product_attention_7.transpose(-1, -2)
        scaled_dot_product_attention_7 = None
        x_259 = transpose_31.reshape(1, -1, 14, 14)
        transpose_31 = None
        x_260 = torch.conv2d(
            x_259,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_259 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_ = (None)
        x_261 = torch.nn.functional.dropout(x_260, 0.0, False, False)
        x_260 = None
        x_262 = x_255 + x_261
        x_255 = x_261 = None
        x_263 = x_262.permute(0, 2, 3, 1)
        x_264 = torch.nn.functional.layer_norm(
            x_263,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_263 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_ = (None)
        x_265 = x_264.permute(0, 3, 1, 2)
        x_264 = None
        x_266 = torch.conv2d(
            x_265,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_265 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_267 = torch._C._nn.gelu(x_266)
        x_266 = None
        x_268 = torch.nn.functional.dropout(x_267, 0.0, False, False)
        x_267 = None
        x_269 = torch.conv2d(
            x_268,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_268 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_270 = x_262 + x_269
        x_262 = x_269 = None
        x_271 = x_270.permute(0, 2, 3, 1)
        x_272 = torch.nn.functional.layer_norm(
            x_271,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_271 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_ = (None)
        x_273 = x_272.permute(0, 3, 1, 2)
        x_272 = None
        conv2d_107 = torch.conv2d(
            x_273,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_273 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_16 = conv2d_107.view(1, 40, 96, -1)
        conv2d_107 = None
        chunk_8 = view_16.chunk(3, dim=2)
        view_16 = None
        q_8 = chunk_8[0]
        k_8 = chunk_8[1]
        v_8 = chunk_8[2]
        chunk_8 = None
        relative_position_bias_16 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_17 = relative_position_bias_16.view((196, 196, 40))
        relative_position_bias_16 = None
        relative_position_bias_17 = view_17.permute(2, 0, 1)
        view_17 = None
        unsqueeze_8 = relative_position_bias_17.unsqueeze(0)
        relative_position_bias_17 = None
        attn_bias_8 = unsqueeze_8.contiguous()
        unsqueeze_8 = None
        transpose_32 = q_8.transpose(-1, -2)
        q_8 = None
        contiguous_33 = transpose_32.contiguous()
        transpose_32 = None
        transpose_33 = k_8.transpose(-1, -2)
        k_8 = None
        contiguous_34 = transpose_33.contiguous()
        transpose_33 = None
        transpose_34 = v_8.transpose(-1, -2)
        v_8 = None
        contiguous_35 = transpose_34.contiguous()
        transpose_34 = None
        scaled_dot_product_attention_8 = torch._C._nn.scaled_dot_product_attention(
            contiguous_33,
            contiguous_34,
            contiguous_35,
            attn_mask=attn_bias_8,
            dropout_p=0.0,
        )
        contiguous_33 = contiguous_34 = contiguous_35 = attn_bias_8 = None
        transpose_35 = scaled_dot_product_attention_8.transpose(-1, -2)
        scaled_dot_product_attention_8 = None
        x_274 = transpose_35.reshape(1, -1, 14, 14)
        transpose_35 = None
        x_275 = torch.conv2d(
            x_274,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_274 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_ = (None)
        x_276 = torch.nn.functional.dropout(x_275, 0.0, False, False)
        x_275 = None
        x_277 = x_270 + x_276
        x_270 = x_276 = None
        x_278 = x_277.permute(0, 2, 3, 1)
        x_279 = torch.nn.functional.layer_norm(
            x_278,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_278 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_ = (None)
        x_280 = x_279.permute(0, 3, 1, 2)
        x_279 = None
        x_281 = torch.conv2d(
            x_280,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_280 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_282 = torch._C._nn.gelu(x_281)
        x_281 = None
        x_283 = torch.nn.functional.dropout(x_282, 0.0, False, False)
        x_282 = None
        x_284 = torch.conv2d(
            x_283,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_283 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_285 = x_277 + x_284
        x_277 = x_284 = None
        x_286 = x_285.permute(0, 2, 3, 1)
        x_287 = torch.nn.functional.layer_norm(
            x_286,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_286 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_ = (None)
        x_288 = x_287.permute(0, 3, 1, 2)
        x_287 = None
        conv2d_111 = torch.conv2d(
            x_288,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_288 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_18 = conv2d_111.view(1, 40, 96, -1)
        conv2d_111 = None
        chunk_9 = view_18.chunk(3, dim=2)
        view_18 = None
        q_9 = chunk_9[0]
        k_9 = chunk_9[1]
        v_9 = chunk_9[2]
        chunk_9 = None
        relative_position_bias_18 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_19 = relative_position_bias_18.view((196, 196, 40))
        relative_position_bias_18 = None
        relative_position_bias_19 = view_19.permute(2, 0, 1)
        view_19 = None
        unsqueeze_9 = relative_position_bias_19.unsqueeze(0)
        relative_position_bias_19 = None
        attn_bias_9 = unsqueeze_9.contiguous()
        unsqueeze_9 = None
        transpose_36 = q_9.transpose(-1, -2)
        q_9 = None
        contiguous_37 = transpose_36.contiguous()
        transpose_36 = None
        transpose_37 = k_9.transpose(-1, -2)
        k_9 = None
        contiguous_38 = transpose_37.contiguous()
        transpose_37 = None
        transpose_38 = v_9.transpose(-1, -2)
        v_9 = None
        contiguous_39 = transpose_38.contiguous()
        transpose_38 = None
        scaled_dot_product_attention_9 = torch._C._nn.scaled_dot_product_attention(
            contiguous_37,
            contiguous_38,
            contiguous_39,
            attn_mask=attn_bias_9,
            dropout_p=0.0,
        )
        contiguous_37 = contiguous_38 = contiguous_39 = attn_bias_9 = None
        transpose_39 = scaled_dot_product_attention_9.transpose(-1, -2)
        scaled_dot_product_attention_9 = None
        x_289 = transpose_39.reshape(1, -1, 14, 14)
        transpose_39 = None
        x_290 = torch.conv2d(
            x_289,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_289 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_ = (None)
        x_291 = torch.nn.functional.dropout(x_290, 0.0, False, False)
        x_290 = None
        x_292 = x_285 + x_291
        x_285 = x_291 = None
        x_293 = x_292.permute(0, 2, 3, 1)
        x_294 = torch.nn.functional.layer_norm(
            x_293,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_293 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_ = (None)
        x_295 = x_294.permute(0, 3, 1, 2)
        x_294 = None
        x_296 = torch.conv2d(
            x_295,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_295 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_297 = torch._C._nn.gelu(x_296)
        x_296 = None
        x_298 = torch.nn.functional.dropout(x_297, 0.0, False, False)
        x_297 = None
        x_299 = torch.conv2d(
            x_298,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_298 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_300 = x_292 + x_299
        x_292 = x_299 = None
        x_301 = x_300.permute(0, 2, 3, 1)
        x_302 = torch.nn.functional.layer_norm(
            x_301,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_301 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_ = (None)
        x_303 = x_302.permute(0, 3, 1, 2)
        x_302 = None
        conv2d_115 = torch.conv2d(
            x_303,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_303 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_20 = conv2d_115.view(1, 40, 96, -1)
        conv2d_115 = None
        chunk_10 = view_20.chunk(3, dim=2)
        view_20 = None
        q_10 = chunk_10[0]
        k_10 = chunk_10[1]
        v_10 = chunk_10[2]
        chunk_10 = None
        relative_position_bias_20 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_21 = relative_position_bias_20.view((196, 196, 40))
        relative_position_bias_20 = None
        relative_position_bias_21 = view_21.permute(2, 0, 1)
        view_21 = None
        unsqueeze_10 = relative_position_bias_21.unsqueeze(0)
        relative_position_bias_21 = None
        attn_bias_10 = unsqueeze_10.contiguous()
        unsqueeze_10 = None
        transpose_40 = q_10.transpose(-1, -2)
        q_10 = None
        contiguous_41 = transpose_40.contiguous()
        transpose_40 = None
        transpose_41 = k_10.transpose(-1, -2)
        k_10 = None
        contiguous_42 = transpose_41.contiguous()
        transpose_41 = None
        transpose_42 = v_10.transpose(-1, -2)
        v_10 = None
        contiguous_43 = transpose_42.contiguous()
        transpose_42 = None
        scaled_dot_product_attention_10 = torch._C._nn.scaled_dot_product_attention(
            contiguous_41,
            contiguous_42,
            contiguous_43,
            attn_mask=attn_bias_10,
            dropout_p=0.0,
        )
        contiguous_41 = contiguous_42 = contiguous_43 = attn_bias_10 = None
        transpose_43 = scaled_dot_product_attention_10.transpose(-1, -2)
        scaled_dot_product_attention_10 = None
        x_304 = transpose_43.reshape(1, -1, 14, 14)
        transpose_43 = None
        x_305 = torch.conv2d(
            x_304,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_304 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_ = (None)
        x_306 = torch.nn.functional.dropout(x_305, 0.0, False, False)
        x_305 = None
        x_307 = x_300 + x_306
        x_300 = x_306 = None
        x_308 = x_307.permute(0, 2, 3, 1)
        x_309 = torch.nn.functional.layer_norm(
            x_308,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_308 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_ = (None)
        x_310 = x_309.permute(0, 3, 1, 2)
        x_309 = None
        x_311 = torch.conv2d(
            x_310,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_310 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_312 = torch._C._nn.gelu(x_311)
        x_311 = None
        x_313 = torch.nn.functional.dropout(x_312, 0.0, False, False)
        x_312 = None
        x_314 = torch.conv2d(
            x_313,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_313 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_315 = x_307 + x_314
        x_307 = x_314 = None
        x_316 = x_315.permute(0, 2, 3, 1)
        x_317 = torch.nn.functional.layer_norm(
            x_316,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_316 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_ = (None)
        x_318 = x_317.permute(0, 3, 1, 2)
        x_317 = None
        conv2d_119 = torch.conv2d(
            x_318,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_318 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_22 = conv2d_119.view(1, 40, 96, -1)
        conv2d_119 = None
        chunk_11 = view_22.chunk(3, dim=2)
        view_22 = None
        q_11 = chunk_11[0]
        k_11 = chunk_11[1]
        v_11 = chunk_11[2]
        chunk_11 = None
        relative_position_bias_22 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_23 = relative_position_bias_22.view((196, 196, 40))
        relative_position_bias_22 = None
        relative_position_bias_23 = view_23.permute(2, 0, 1)
        view_23 = None
        unsqueeze_11 = relative_position_bias_23.unsqueeze(0)
        relative_position_bias_23 = None
        attn_bias_11 = unsqueeze_11.contiguous()
        unsqueeze_11 = None
        transpose_44 = q_11.transpose(-1, -2)
        q_11 = None
        contiguous_45 = transpose_44.contiguous()
        transpose_44 = None
        transpose_45 = k_11.transpose(-1, -2)
        k_11 = None
        contiguous_46 = transpose_45.contiguous()
        transpose_45 = None
        transpose_46 = v_11.transpose(-1, -2)
        v_11 = None
        contiguous_47 = transpose_46.contiguous()
        transpose_46 = None
        scaled_dot_product_attention_11 = torch._C._nn.scaled_dot_product_attention(
            contiguous_45,
            contiguous_46,
            contiguous_47,
            attn_mask=attn_bias_11,
            dropout_p=0.0,
        )
        contiguous_45 = contiguous_46 = contiguous_47 = attn_bias_11 = None
        transpose_47 = scaled_dot_product_attention_11.transpose(-1, -2)
        scaled_dot_product_attention_11 = None
        x_319 = transpose_47.reshape(1, -1, 14, 14)
        transpose_47 = None
        x_320 = torch.conv2d(
            x_319,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_319 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_ = (None)
        x_321 = torch.nn.functional.dropout(x_320, 0.0, False, False)
        x_320 = None
        x_322 = x_315 + x_321
        x_315 = x_321 = None
        x_323 = x_322.permute(0, 2, 3, 1)
        x_324 = torch.nn.functional.layer_norm(
            x_323,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_323 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_ = (None)
        x_325 = x_324.permute(0, 3, 1, 2)
        x_324 = None
        x_326 = torch.conv2d(
            x_325,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_325 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_327 = torch._C._nn.gelu(x_326)
        x_326 = None
        x_328 = torch.nn.functional.dropout(x_327, 0.0, False, False)
        x_327 = None
        x_329 = torch.conv2d(
            x_328,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_328 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_330 = x_322 + x_329
        x_322 = x_329 = None
        x_331 = x_330.permute(0, 2, 3, 1)
        x_332 = torch.nn.functional.layer_norm(
            x_331,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_331 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_ = (None)
        x_333 = x_332.permute(0, 3, 1, 2)
        x_332 = None
        conv2d_123 = torch.conv2d(
            x_333,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_333 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_24 = conv2d_123.view(1, 40, 96, -1)
        conv2d_123 = None
        chunk_12 = view_24.chunk(3, dim=2)
        view_24 = None
        q_12 = chunk_12[0]
        k_12 = chunk_12[1]
        v_12 = chunk_12[2]
        chunk_12 = None
        relative_position_bias_24 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_25 = relative_position_bias_24.view((196, 196, 40))
        relative_position_bias_24 = None
        relative_position_bias_25 = view_25.permute(2, 0, 1)
        view_25 = None
        unsqueeze_12 = relative_position_bias_25.unsqueeze(0)
        relative_position_bias_25 = None
        attn_bias_12 = unsqueeze_12.contiguous()
        unsqueeze_12 = None
        transpose_48 = q_12.transpose(-1, -2)
        q_12 = None
        contiguous_49 = transpose_48.contiguous()
        transpose_48 = None
        transpose_49 = k_12.transpose(-1, -2)
        k_12 = None
        contiguous_50 = transpose_49.contiguous()
        transpose_49 = None
        transpose_50 = v_12.transpose(-1, -2)
        v_12 = None
        contiguous_51 = transpose_50.contiguous()
        transpose_50 = None
        scaled_dot_product_attention_12 = torch._C._nn.scaled_dot_product_attention(
            contiguous_49,
            contiguous_50,
            contiguous_51,
            attn_mask=attn_bias_12,
            dropout_p=0.0,
        )
        contiguous_49 = contiguous_50 = contiguous_51 = attn_bias_12 = None
        transpose_51 = scaled_dot_product_attention_12.transpose(-1, -2)
        scaled_dot_product_attention_12 = None
        x_334 = transpose_51.reshape(1, -1, 14, 14)
        transpose_51 = None
        x_335 = torch.conv2d(
            x_334,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_334 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_ = (None)
        x_336 = torch.nn.functional.dropout(x_335, 0.0, False, False)
        x_335 = None
        x_337 = x_330 + x_336
        x_330 = x_336 = None
        x_338 = x_337.permute(0, 2, 3, 1)
        x_339 = torch.nn.functional.layer_norm(
            x_338,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_338 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_ = (None)
        x_340 = x_339.permute(0, 3, 1, 2)
        x_339 = None
        x_341 = torch.conv2d(
            x_340,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_340 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_342 = torch._C._nn.gelu(x_341)
        x_341 = None
        x_343 = torch.nn.functional.dropout(x_342, 0.0, False, False)
        x_342 = None
        x_344 = torch.conv2d(
            x_343,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_343 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_345 = x_337 + x_344
        x_337 = x_344 = None
        x_346 = x_345.permute(0, 2, 3, 1)
        x_347 = torch.nn.functional.layer_norm(
            x_346,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_346 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_ = (None)
        x_348 = x_347.permute(0, 3, 1, 2)
        x_347 = None
        conv2d_127 = torch.conv2d(
            x_348,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_348 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_26 = conv2d_127.view(1, 40, 96, -1)
        conv2d_127 = None
        chunk_13 = view_26.chunk(3, dim=2)
        view_26 = None
        q_13 = chunk_13[0]
        k_13 = chunk_13[1]
        v_13 = chunk_13[2]
        chunk_13 = None
        relative_position_bias_26 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_27 = relative_position_bias_26.view((196, 196, 40))
        relative_position_bias_26 = None
        relative_position_bias_27 = view_27.permute(2, 0, 1)
        view_27 = None
        unsqueeze_13 = relative_position_bias_27.unsqueeze(0)
        relative_position_bias_27 = None
        attn_bias_13 = unsqueeze_13.contiguous()
        unsqueeze_13 = None
        transpose_52 = q_13.transpose(-1, -2)
        q_13 = None
        contiguous_53 = transpose_52.contiguous()
        transpose_52 = None
        transpose_53 = k_13.transpose(-1, -2)
        k_13 = None
        contiguous_54 = transpose_53.contiguous()
        transpose_53 = None
        transpose_54 = v_13.transpose(-1, -2)
        v_13 = None
        contiguous_55 = transpose_54.contiguous()
        transpose_54 = None
        scaled_dot_product_attention_13 = torch._C._nn.scaled_dot_product_attention(
            contiguous_53,
            contiguous_54,
            contiguous_55,
            attn_mask=attn_bias_13,
            dropout_p=0.0,
        )
        contiguous_53 = contiguous_54 = contiguous_55 = attn_bias_13 = None
        transpose_55 = scaled_dot_product_attention_13.transpose(-1, -2)
        scaled_dot_product_attention_13 = None
        x_349 = transpose_55.reshape(1, -1, 14, 14)
        transpose_55 = None
        x_350 = torch.conv2d(
            x_349,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_349 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_ = (None)
        x_351 = torch.nn.functional.dropout(x_350, 0.0, False, False)
        x_350 = None
        x_352 = x_345 + x_351
        x_345 = x_351 = None
        x_353 = x_352.permute(0, 2, 3, 1)
        x_354 = torch.nn.functional.layer_norm(
            x_353,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_353 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_ = (None)
        x_355 = x_354.permute(0, 3, 1, 2)
        x_354 = None
        x_356 = torch.conv2d(
            x_355,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_355 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_357 = torch._C._nn.gelu(x_356)
        x_356 = None
        x_358 = torch.nn.functional.dropout(x_357, 0.0, False, False)
        x_357 = None
        x_359 = torch.conv2d(
            x_358,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_358 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_360 = x_352 + x_359
        x_352 = x_359 = None
        x_361 = x_360.permute(0, 2, 3, 1)
        x_362 = torch.nn.functional.layer_norm(
            x_361,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_361 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_bias_ = (None)
        x_363 = x_362.permute(0, 3, 1, 2)
        x_362 = None
        conv2d_131 = torch.conv2d(
            x_363,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_363 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_28 = conv2d_131.view(1, 40, 96, -1)
        conv2d_131 = None
        chunk_14 = view_28.chunk(3, dim=2)
        view_28 = None
        q_14 = chunk_14[0]
        k_14 = chunk_14[1]
        v_14 = chunk_14[2]
        chunk_14 = None
        relative_position_bias_28 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_29 = relative_position_bias_28.view((196, 196, 40))
        relative_position_bias_28 = None
        relative_position_bias_29 = view_29.permute(2, 0, 1)
        view_29 = None
        unsqueeze_14 = relative_position_bias_29.unsqueeze(0)
        relative_position_bias_29 = None
        attn_bias_14 = unsqueeze_14.contiguous()
        unsqueeze_14 = None
        transpose_56 = q_14.transpose(-1, -2)
        q_14 = None
        contiguous_57 = transpose_56.contiguous()
        transpose_56 = None
        transpose_57 = k_14.transpose(-1, -2)
        k_14 = None
        contiguous_58 = transpose_57.contiguous()
        transpose_57 = None
        transpose_58 = v_14.transpose(-1, -2)
        v_14 = None
        contiguous_59 = transpose_58.contiguous()
        transpose_58 = None
        scaled_dot_product_attention_14 = torch._C._nn.scaled_dot_product_attention(
            contiguous_57,
            contiguous_58,
            contiguous_59,
            attn_mask=attn_bias_14,
            dropout_p=0.0,
        )
        contiguous_57 = contiguous_58 = contiguous_59 = attn_bias_14 = None
        transpose_59 = scaled_dot_product_attention_14.transpose(-1, -2)
        scaled_dot_product_attention_14 = None
        x_364 = transpose_59.reshape(1, -1, 14, 14)
        transpose_59 = None
        x_365 = torch.conv2d(
            x_364,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_364 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_ = (None)
        x_366 = torch.nn.functional.dropout(x_365, 0.0, False, False)
        x_365 = None
        x_367 = x_360 + x_366
        x_360 = x_366 = None
        x_368 = x_367.permute(0, 2, 3, 1)
        x_369 = torch.nn.functional.layer_norm(
            x_368,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_368 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_bias_ = (None)
        x_370 = x_369.permute(0, 3, 1, 2)
        x_369 = None
        x_371 = torch.conv2d(
            x_370,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_370 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_372 = torch._C._nn.gelu(x_371)
        x_371 = None
        x_373 = torch.nn.functional.dropout(x_372, 0.0, False, False)
        x_372 = None
        x_374 = torch.conv2d(
            x_373,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_373 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_375 = x_367 + x_374
        x_367 = x_374 = None
        x_376 = x_375.permute(0, 2, 3, 1)
        x_377 = torch.nn.functional.layer_norm(
            x_376,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_376 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_bias_ = (None)
        x_378 = x_377.permute(0, 3, 1, 2)
        x_377 = None
        conv2d_135 = torch.conv2d(
            x_378,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_378 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_30 = conv2d_135.view(1, 40, 96, -1)
        conv2d_135 = None
        chunk_15 = view_30.chunk(3, dim=2)
        view_30 = None
        q_15 = chunk_15[0]
        k_15 = chunk_15[1]
        v_15 = chunk_15[2]
        chunk_15 = None
        relative_position_bias_30 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_31 = relative_position_bias_30.view((196, 196, 40))
        relative_position_bias_30 = None
        relative_position_bias_31 = view_31.permute(2, 0, 1)
        view_31 = None
        unsqueeze_15 = relative_position_bias_31.unsqueeze(0)
        relative_position_bias_31 = None
        attn_bias_15 = unsqueeze_15.contiguous()
        unsqueeze_15 = None
        transpose_60 = q_15.transpose(-1, -2)
        q_15 = None
        contiguous_61 = transpose_60.contiguous()
        transpose_60 = None
        transpose_61 = k_15.transpose(-1, -2)
        k_15 = None
        contiguous_62 = transpose_61.contiguous()
        transpose_61 = None
        transpose_62 = v_15.transpose(-1, -2)
        v_15 = None
        contiguous_63 = transpose_62.contiguous()
        transpose_62 = None
        scaled_dot_product_attention_15 = torch._C._nn.scaled_dot_product_attention(
            contiguous_61,
            contiguous_62,
            contiguous_63,
            attn_mask=attn_bias_15,
            dropout_p=0.0,
        )
        contiguous_61 = contiguous_62 = contiguous_63 = attn_bias_15 = None
        transpose_63 = scaled_dot_product_attention_15.transpose(-1, -2)
        scaled_dot_product_attention_15 = None
        x_379 = transpose_63.reshape(1, -1, 14, 14)
        transpose_63 = None
        x_380 = torch.conv2d(
            x_379,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_379 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_ = (None)
        x_381 = torch.nn.functional.dropout(x_380, 0.0, False, False)
        x_380 = None
        x_382 = x_375 + x_381
        x_375 = x_381 = None
        x_383 = x_382.permute(0, 2, 3, 1)
        x_384 = torch.nn.functional.layer_norm(
            x_383,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_383 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_bias_ = (None)
        x_385 = x_384.permute(0, 3, 1, 2)
        x_384 = None
        x_386 = torch.conv2d(
            x_385,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_385 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_387 = torch._C._nn.gelu(x_386)
        x_386 = None
        x_388 = torch.nn.functional.dropout(x_387, 0.0, False, False)
        x_387 = None
        x_389 = torch.conv2d(
            x_388,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_388 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_390 = x_382 + x_389
        x_382 = x_389 = None
        x_391 = x_390.permute(0, 2, 3, 1)
        x_392 = torch.nn.functional.layer_norm(
            x_391,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_391 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_bias_ = (None)
        x_393 = x_392.permute(0, 3, 1, 2)
        x_392 = None
        conv2d_139 = torch.conv2d(
            x_393,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_393 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_32 = conv2d_139.view(1, 40, 96, -1)
        conv2d_139 = None
        chunk_16 = view_32.chunk(3, dim=2)
        view_32 = None
        q_16 = chunk_16[0]
        k_16 = chunk_16[1]
        v_16 = chunk_16[2]
        chunk_16 = None
        relative_position_bias_32 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_33 = relative_position_bias_32.view((196, 196, 40))
        relative_position_bias_32 = None
        relative_position_bias_33 = view_33.permute(2, 0, 1)
        view_33 = None
        unsqueeze_16 = relative_position_bias_33.unsqueeze(0)
        relative_position_bias_33 = None
        attn_bias_16 = unsqueeze_16.contiguous()
        unsqueeze_16 = None
        transpose_64 = q_16.transpose(-1, -2)
        q_16 = None
        contiguous_65 = transpose_64.contiguous()
        transpose_64 = None
        transpose_65 = k_16.transpose(-1, -2)
        k_16 = None
        contiguous_66 = transpose_65.contiguous()
        transpose_65 = None
        transpose_66 = v_16.transpose(-1, -2)
        v_16 = None
        contiguous_67 = transpose_66.contiguous()
        transpose_66 = None
        scaled_dot_product_attention_16 = torch._C._nn.scaled_dot_product_attention(
            contiguous_65,
            contiguous_66,
            contiguous_67,
            attn_mask=attn_bias_16,
            dropout_p=0.0,
        )
        contiguous_65 = contiguous_66 = contiguous_67 = attn_bias_16 = None
        transpose_67 = scaled_dot_product_attention_16.transpose(-1, -2)
        scaled_dot_product_attention_16 = None
        x_394 = transpose_67.reshape(1, -1, 14, 14)
        transpose_67 = None
        x_395 = torch.conv2d(
            x_394,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_394 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_ = (None)
        x_396 = torch.nn.functional.dropout(x_395, 0.0, False, False)
        x_395 = None
        x_397 = x_390 + x_396
        x_390 = x_396 = None
        x_398 = x_397.permute(0, 2, 3, 1)
        x_399 = torch.nn.functional.layer_norm(
            x_398,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_398 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_bias_ = (None)
        x_400 = x_399.permute(0, 3, 1, 2)
        x_399 = None
        x_401 = torch.conv2d(
            x_400,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_400 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_402 = torch._C._nn.gelu(x_401)
        x_401 = None
        x_403 = torch.nn.functional.dropout(x_402, 0.0, False, False)
        x_402 = None
        x_404 = torch.conv2d(
            x_403,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_403 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_405 = x_397 + x_404
        x_397 = x_404 = None
        x_406 = x_405.permute(0, 2, 3, 1)
        x_407 = torch.nn.functional.layer_norm(
            x_406,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_406 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_bias_ = (None)
        x_408 = x_407.permute(0, 3, 1, 2)
        x_407 = None
        conv2d_143 = torch.conv2d(
            x_408,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_408 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_34 = conv2d_143.view(1, 40, 96, -1)
        conv2d_143 = None
        chunk_17 = view_34.chunk(3, dim=2)
        view_34 = None
        q_17 = chunk_17[0]
        k_17 = chunk_17[1]
        v_17 = chunk_17[2]
        chunk_17 = None
        relative_position_bias_34 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_35 = relative_position_bias_34.view((196, 196, 40))
        relative_position_bias_34 = None
        relative_position_bias_35 = view_35.permute(2, 0, 1)
        view_35 = None
        unsqueeze_17 = relative_position_bias_35.unsqueeze(0)
        relative_position_bias_35 = None
        attn_bias_17 = unsqueeze_17.contiguous()
        unsqueeze_17 = None
        transpose_68 = q_17.transpose(-1, -2)
        q_17 = None
        contiguous_69 = transpose_68.contiguous()
        transpose_68 = None
        transpose_69 = k_17.transpose(-1, -2)
        k_17 = None
        contiguous_70 = transpose_69.contiguous()
        transpose_69 = None
        transpose_70 = v_17.transpose(-1, -2)
        v_17 = None
        contiguous_71 = transpose_70.contiguous()
        transpose_70 = None
        scaled_dot_product_attention_17 = torch._C._nn.scaled_dot_product_attention(
            contiguous_69,
            contiguous_70,
            contiguous_71,
            attn_mask=attn_bias_17,
            dropout_p=0.0,
        )
        contiguous_69 = contiguous_70 = contiguous_71 = attn_bias_17 = None
        transpose_71 = scaled_dot_product_attention_17.transpose(-1, -2)
        scaled_dot_product_attention_17 = None
        x_409 = transpose_71.reshape(1, -1, 14, 14)
        transpose_71 = None
        x_410 = torch.conv2d(
            x_409,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_409 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_ = (None)
        x_411 = torch.nn.functional.dropout(x_410, 0.0, False, False)
        x_410 = None
        x_412 = x_405 + x_411
        x_405 = x_411 = None
        x_413 = x_412.permute(0, 2, 3, 1)
        x_414 = torch.nn.functional.layer_norm(
            x_413,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_413 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_bias_ = (None)
        x_415 = x_414.permute(0, 3, 1, 2)
        x_414 = None
        x_416 = torch.conv2d(
            x_415,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_415 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_417 = torch._C._nn.gelu(x_416)
        x_416 = None
        x_418 = torch.nn.functional.dropout(x_417, 0.0, False, False)
        x_417 = None
        x_419 = torch.conv2d(
            x_418,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_418 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_420 = x_412 + x_419
        x_412 = x_419 = None
        x_421 = x_420.permute(0, 2, 3, 1)
        x_422 = torch.nn.functional.layer_norm(
            x_421,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_421 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm1_parameters_bias_ = (None)
        x_423 = x_422.permute(0, 3, 1, 2)
        x_422 = None
        conv2d_147 = torch.conv2d(
            x_423,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_423 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_36 = conv2d_147.view(1, 40, 96, -1)
        conv2d_147 = None
        chunk_18 = view_36.chunk(3, dim=2)
        view_36 = None
        q_18 = chunk_18[0]
        k_18 = chunk_18[1]
        v_18 = chunk_18[2]
        chunk_18 = None
        relative_position_bias_36 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_37 = relative_position_bias_36.view((196, 196, 40))
        relative_position_bias_36 = None
        relative_position_bias_37 = view_37.permute(2, 0, 1)
        view_37 = None
        unsqueeze_18 = relative_position_bias_37.unsqueeze(0)
        relative_position_bias_37 = None
        attn_bias_18 = unsqueeze_18.contiguous()
        unsqueeze_18 = None
        transpose_72 = q_18.transpose(-1, -2)
        q_18 = None
        contiguous_73 = transpose_72.contiguous()
        transpose_72 = None
        transpose_73 = k_18.transpose(-1, -2)
        k_18 = None
        contiguous_74 = transpose_73.contiguous()
        transpose_73 = None
        transpose_74 = v_18.transpose(-1, -2)
        v_18 = None
        contiguous_75 = transpose_74.contiguous()
        transpose_74 = None
        scaled_dot_product_attention_18 = torch._C._nn.scaled_dot_product_attention(
            contiguous_73,
            contiguous_74,
            contiguous_75,
            attn_mask=attn_bias_18,
            dropout_p=0.0,
        )
        contiguous_73 = contiguous_74 = contiguous_75 = attn_bias_18 = None
        transpose_75 = scaled_dot_product_attention_18.transpose(-1, -2)
        scaled_dot_product_attention_18 = None
        x_424 = transpose_75.reshape(1, -1, 14, 14)
        transpose_75 = None
        x_425 = torch.conv2d(
            x_424,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_424 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_ = (None)
        x_426 = torch.nn.functional.dropout(x_425, 0.0, False, False)
        x_425 = None
        x_427 = x_420 + x_426
        x_420 = x_426 = None
        x_428 = x_427.permute(0, 2, 3, 1)
        x_429 = torch.nn.functional.layer_norm(
            x_428,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_428 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm2_parameters_bias_ = (None)
        x_430 = x_429.permute(0, 3, 1, 2)
        x_429 = None
        x_431 = torch.conv2d(
            x_430,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_430 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_432 = torch._C._nn.gelu(x_431)
        x_431 = None
        x_433 = torch.nn.functional.dropout(x_432, 0.0, False, False)
        x_432 = None
        x_434 = torch.conv2d(
            x_433,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_433 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_435 = x_427 + x_434
        x_427 = x_434 = None
        x_436 = x_435.permute(0, 2, 3, 1)
        x_437 = torch.nn.functional.layer_norm(
            x_436,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_436 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm1_parameters_bias_ = (None)
        x_438 = x_437.permute(0, 3, 1, 2)
        x_437 = None
        conv2d_151 = torch.conv2d(
            x_438,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_438 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_38 = conv2d_151.view(1, 40, 96, -1)
        conv2d_151 = None
        chunk_19 = view_38.chunk(3, dim=2)
        view_38 = None
        q_19 = chunk_19[0]
        k_19 = chunk_19[1]
        v_19 = chunk_19[2]
        chunk_19 = None
        relative_position_bias_38 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_39 = relative_position_bias_38.view((196, 196, 40))
        relative_position_bias_38 = None
        relative_position_bias_39 = view_39.permute(2, 0, 1)
        view_39 = None
        unsqueeze_19 = relative_position_bias_39.unsqueeze(0)
        relative_position_bias_39 = None
        attn_bias_19 = unsqueeze_19.contiguous()
        unsqueeze_19 = None
        transpose_76 = q_19.transpose(-1, -2)
        q_19 = None
        contiguous_77 = transpose_76.contiguous()
        transpose_76 = None
        transpose_77 = k_19.transpose(-1, -2)
        k_19 = None
        contiguous_78 = transpose_77.contiguous()
        transpose_77 = None
        transpose_78 = v_19.transpose(-1, -2)
        v_19 = None
        contiguous_79 = transpose_78.contiguous()
        transpose_78 = None
        scaled_dot_product_attention_19 = torch._C._nn.scaled_dot_product_attention(
            contiguous_77,
            contiguous_78,
            contiguous_79,
            attn_mask=attn_bias_19,
            dropout_p=0.0,
        )
        contiguous_77 = contiguous_78 = contiguous_79 = attn_bias_19 = None
        transpose_79 = scaled_dot_product_attention_19.transpose(-1, -2)
        scaled_dot_product_attention_19 = None
        x_439 = transpose_79.reshape(1, -1, 14, 14)
        transpose_79 = None
        x_440 = torch.conv2d(
            x_439,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_439 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_ = (None)
        x_441 = torch.nn.functional.dropout(x_440, 0.0, False, False)
        x_440 = None
        x_442 = x_435 + x_441
        x_435 = x_441 = None
        x_443 = x_442.permute(0, 2, 3, 1)
        x_444 = torch.nn.functional.layer_norm(
            x_443,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_443 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm2_parameters_bias_ = (None)
        x_445 = x_444.permute(0, 3, 1, 2)
        x_444 = None
        x_446 = torch.conv2d(
            x_445,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_445 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_447 = torch._C._nn.gelu(x_446)
        x_446 = None
        x_448 = torch.nn.functional.dropout(x_447, 0.0, False, False)
        x_447 = None
        x_449 = torch.conv2d(
            x_448,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_448 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_450 = x_442 + x_449
        x_442 = x_449 = None
        x_451 = x_450.permute(0, 2, 3, 1)
        x_452 = torch.nn.functional.layer_norm(
            x_451,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_451 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm1_parameters_bias_ = (None)
        x_453 = x_452.permute(0, 3, 1, 2)
        x_452 = None
        conv2d_155 = torch.conv2d(
            x_453,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_453 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_40 = conv2d_155.view(1, 40, 96, -1)
        conv2d_155 = None
        chunk_20 = view_40.chunk(3, dim=2)
        view_40 = None
        q_20 = chunk_20[0]
        k_20 = chunk_20[1]
        v_20 = chunk_20[2]
        chunk_20 = None
        relative_position_bias_40 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_41 = relative_position_bias_40.view((196, 196, 40))
        relative_position_bias_40 = None
        relative_position_bias_41 = view_41.permute(2, 0, 1)
        view_41 = None
        unsqueeze_20 = relative_position_bias_41.unsqueeze(0)
        relative_position_bias_41 = None
        attn_bias_20 = unsqueeze_20.contiguous()
        unsqueeze_20 = None
        transpose_80 = q_20.transpose(-1, -2)
        q_20 = None
        contiguous_81 = transpose_80.contiguous()
        transpose_80 = None
        transpose_81 = k_20.transpose(-1, -2)
        k_20 = None
        contiguous_82 = transpose_81.contiguous()
        transpose_81 = None
        transpose_82 = v_20.transpose(-1, -2)
        v_20 = None
        contiguous_83 = transpose_82.contiguous()
        transpose_82 = None
        scaled_dot_product_attention_20 = torch._C._nn.scaled_dot_product_attention(
            contiguous_81,
            contiguous_82,
            contiguous_83,
            attn_mask=attn_bias_20,
            dropout_p=0.0,
        )
        contiguous_81 = contiguous_82 = contiguous_83 = attn_bias_20 = None
        transpose_83 = scaled_dot_product_attention_20.transpose(-1, -2)
        scaled_dot_product_attention_20 = None
        x_454 = transpose_83.reshape(1, -1, 14, 14)
        transpose_83 = None
        x_455 = torch.conv2d(
            x_454,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_454 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_ = (None)
        x_456 = torch.nn.functional.dropout(x_455, 0.0, False, False)
        x_455 = None
        x_457 = x_450 + x_456
        x_450 = x_456 = None
        x_458 = x_457.permute(0, 2, 3, 1)
        x_459 = torch.nn.functional.layer_norm(
            x_458,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_458 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm2_parameters_bias_ = (None)
        x_460 = x_459.permute(0, 3, 1, 2)
        x_459 = None
        x_461 = torch.conv2d(
            x_460,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_460 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_462 = torch._C._nn.gelu(x_461)
        x_461 = None
        x_463 = torch.nn.functional.dropout(x_462, 0.0, False, False)
        x_462 = None
        x_464 = torch.conv2d(
            x_463,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_463 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_465 = x_457 + x_464
        x_457 = x_464 = None
        x_466 = x_465.permute(0, 2, 3, 1)
        x_467 = torch.nn.functional.layer_norm(
            x_466,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_466 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm1_parameters_bias_ = (None)
        x_468 = x_467.permute(0, 3, 1, 2)
        x_467 = None
        conv2d_159 = torch.conv2d(
            x_468,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_468 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_42 = conv2d_159.view(1, 40, 96, -1)
        conv2d_159 = None
        chunk_21 = view_42.chunk(3, dim=2)
        view_42 = None
        q_21 = chunk_21[0]
        k_21 = chunk_21[1]
        v_21 = chunk_21[2]
        chunk_21 = None
        relative_position_bias_42 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_43 = relative_position_bias_42.view((196, 196, 40))
        relative_position_bias_42 = None
        relative_position_bias_43 = view_43.permute(2, 0, 1)
        view_43 = None
        unsqueeze_21 = relative_position_bias_43.unsqueeze(0)
        relative_position_bias_43 = None
        attn_bias_21 = unsqueeze_21.contiguous()
        unsqueeze_21 = None
        transpose_84 = q_21.transpose(-1, -2)
        q_21 = None
        contiguous_85 = transpose_84.contiguous()
        transpose_84 = None
        transpose_85 = k_21.transpose(-1, -2)
        k_21 = None
        contiguous_86 = transpose_85.contiguous()
        transpose_85 = None
        transpose_86 = v_21.transpose(-1, -2)
        v_21 = None
        contiguous_87 = transpose_86.contiguous()
        transpose_86 = None
        scaled_dot_product_attention_21 = torch._C._nn.scaled_dot_product_attention(
            contiguous_85,
            contiguous_86,
            contiguous_87,
            attn_mask=attn_bias_21,
            dropout_p=0.0,
        )
        contiguous_85 = contiguous_86 = contiguous_87 = attn_bias_21 = None
        transpose_87 = scaled_dot_product_attention_21.transpose(-1, -2)
        scaled_dot_product_attention_21 = None
        x_469 = transpose_87.reshape(1, -1, 14, 14)
        transpose_87 = None
        x_470 = torch.conv2d(
            x_469,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_469 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_ = (None)
        x_471 = torch.nn.functional.dropout(x_470, 0.0, False, False)
        x_470 = None
        x_472 = x_465 + x_471
        x_465 = x_471 = None
        x_473 = x_472.permute(0, 2, 3, 1)
        x_474 = torch.nn.functional.layer_norm(
            x_473,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_473 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm2_parameters_bias_ = (None)
        x_475 = x_474.permute(0, 3, 1, 2)
        x_474 = None
        x_476 = torch.conv2d(
            x_475,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_475 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_477 = torch._C._nn.gelu(x_476)
        x_476 = None
        x_478 = torch.nn.functional.dropout(x_477, 0.0, False, False)
        x_477 = None
        x_479 = torch.conv2d(
            x_478,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_478 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_480 = x_472 + x_479
        x_472 = x_479 = None
        x_481 = x_480.permute(0, 2, 3, 1)
        x_482 = torch.nn.functional.layer_norm(
            x_481,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_481 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm1_parameters_bias_ = (None)
        x_483 = x_482.permute(0, 3, 1, 2)
        x_482 = None
        conv2d_163 = torch.conv2d(
            x_483,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_483 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_44 = conv2d_163.view(1, 40, 96, -1)
        conv2d_163 = None
        chunk_22 = view_44.chunk(3, dim=2)
        view_44 = None
        q_22 = chunk_22[0]
        k_22 = chunk_22[1]
        v_22 = chunk_22[2]
        chunk_22 = None
        relative_position_bias_44 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_45 = relative_position_bias_44.view((196, 196, 40))
        relative_position_bias_44 = None
        relative_position_bias_45 = view_45.permute(2, 0, 1)
        view_45 = None
        unsqueeze_22 = relative_position_bias_45.unsqueeze(0)
        relative_position_bias_45 = None
        attn_bias_22 = unsqueeze_22.contiguous()
        unsqueeze_22 = None
        transpose_88 = q_22.transpose(-1, -2)
        q_22 = None
        contiguous_89 = transpose_88.contiguous()
        transpose_88 = None
        transpose_89 = k_22.transpose(-1, -2)
        k_22 = None
        contiguous_90 = transpose_89.contiguous()
        transpose_89 = None
        transpose_90 = v_22.transpose(-1, -2)
        v_22 = None
        contiguous_91 = transpose_90.contiguous()
        transpose_90 = None
        scaled_dot_product_attention_22 = torch._C._nn.scaled_dot_product_attention(
            contiguous_89,
            contiguous_90,
            contiguous_91,
            attn_mask=attn_bias_22,
            dropout_p=0.0,
        )
        contiguous_89 = contiguous_90 = contiguous_91 = attn_bias_22 = None
        transpose_91 = scaled_dot_product_attention_22.transpose(-1, -2)
        scaled_dot_product_attention_22 = None
        x_484 = transpose_91.reshape(1, -1, 14, 14)
        transpose_91 = None
        x_485 = torch.conv2d(
            x_484,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_484 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_ = (None)
        x_486 = torch.nn.functional.dropout(x_485, 0.0, False, False)
        x_485 = None
        x_487 = x_480 + x_486
        x_480 = x_486 = None
        x_488 = x_487.permute(0, 2, 3, 1)
        x_489 = torch.nn.functional.layer_norm(
            x_488,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_488 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm2_parameters_bias_ = (None)
        x_490 = x_489.permute(0, 3, 1, 2)
        x_489 = None
        x_491 = torch.conv2d(
            x_490,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_490 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_492 = torch._C._nn.gelu(x_491)
        x_491 = None
        x_493 = torch.nn.functional.dropout(x_492, 0.0, False, False)
        x_492 = None
        x_494 = torch.conv2d(
            x_493,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_493 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_495 = x_487 + x_494
        x_487 = x_494 = None
        x_496 = x_495.permute(0, 2, 3, 1)
        x_497 = torch.nn.functional.layer_norm(
            x_496,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_496 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm1_parameters_bias_ = (None)
        x_498 = x_497.permute(0, 3, 1, 2)
        x_497 = None
        conv2d_167 = torch.conv2d(
            x_498,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_498 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_46 = conv2d_167.view(1, 40, 96, -1)
        conv2d_167 = None
        chunk_23 = view_46.chunk(3, dim=2)
        view_46 = None
        q_23 = chunk_23[0]
        k_23 = chunk_23[1]
        v_23 = chunk_23[2]
        chunk_23 = None
        relative_position_bias_46 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_47 = relative_position_bias_46.view((196, 196, 40))
        relative_position_bias_46 = None
        relative_position_bias_47 = view_47.permute(2, 0, 1)
        view_47 = None
        unsqueeze_23 = relative_position_bias_47.unsqueeze(0)
        relative_position_bias_47 = None
        attn_bias_23 = unsqueeze_23.contiguous()
        unsqueeze_23 = None
        transpose_92 = q_23.transpose(-1, -2)
        q_23 = None
        contiguous_93 = transpose_92.contiguous()
        transpose_92 = None
        transpose_93 = k_23.transpose(-1, -2)
        k_23 = None
        contiguous_94 = transpose_93.contiguous()
        transpose_93 = None
        transpose_94 = v_23.transpose(-1, -2)
        v_23 = None
        contiguous_95 = transpose_94.contiguous()
        transpose_94 = None
        scaled_dot_product_attention_23 = torch._C._nn.scaled_dot_product_attention(
            contiguous_93,
            contiguous_94,
            contiguous_95,
            attn_mask=attn_bias_23,
            dropout_p=0.0,
        )
        contiguous_93 = contiguous_94 = contiguous_95 = attn_bias_23 = None
        transpose_95 = scaled_dot_product_attention_23.transpose(-1, -2)
        scaled_dot_product_attention_23 = None
        x_499 = transpose_95.reshape(1, -1, 14, 14)
        transpose_95 = None
        x_500 = torch.conv2d(
            x_499,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_499 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_ = (None)
        x_501 = torch.nn.functional.dropout(x_500, 0.0, False, False)
        x_500 = None
        x_502 = x_495 + x_501
        x_495 = x_501 = None
        x_503 = x_502.permute(0, 2, 3, 1)
        x_504 = torch.nn.functional.layer_norm(
            x_503,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_503 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm2_parameters_bias_ = (None)
        x_505 = x_504.permute(0, 3, 1, 2)
        x_504 = None
        x_506 = torch.conv2d(
            x_505,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_505 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_507 = torch._C._nn.gelu(x_506)
        x_506 = None
        x_508 = torch.nn.functional.dropout(x_507, 0.0, False, False)
        x_507 = None
        x_509 = torch.conv2d(
            x_508,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_508 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_510 = x_502 + x_509
        x_502 = x_509 = None
        x_511 = x_510.permute(0, 2, 3, 1)
        x_512 = torch.nn.functional.layer_norm(
            x_511,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_511 = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm1_parameters_bias_ = (None)
        x_513 = x_512.permute(0, 3, 1, 2)
        x_512 = None
        conv2d_171 = torch.conv2d(
            x_513,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_513 = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_48 = conv2d_171.view(1, 40, 96, -1)
        conv2d_171 = None
        chunk_24 = view_48.chunk(3, dim=2)
        view_48 = None
        q_24 = chunk_24[0]
        k_24 = chunk_24[1]
        v_24 = chunk_24[2]
        chunk_24 = None
        relative_position_bias_48 = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_49 = relative_position_bias_48.view((196, 196, 40))
        relative_position_bias_48 = None
        relative_position_bias_49 = view_49.permute(2, 0, 1)
        view_49 = None
        unsqueeze_24 = relative_position_bias_49.unsqueeze(0)
        relative_position_bias_49 = None
        attn_bias_24 = unsqueeze_24.contiguous()
        unsqueeze_24 = None
        transpose_96 = q_24.transpose(-1, -2)
        q_24 = None
        contiguous_97 = transpose_96.contiguous()
        transpose_96 = None
        transpose_97 = k_24.transpose(-1, -2)
        k_24 = None
        contiguous_98 = transpose_97.contiguous()
        transpose_97 = None
        transpose_98 = v_24.transpose(-1, -2)
        v_24 = None
        contiguous_99 = transpose_98.contiguous()
        transpose_98 = None
        scaled_dot_product_attention_24 = torch._C._nn.scaled_dot_product_attention(
            contiguous_97,
            contiguous_98,
            contiguous_99,
            attn_mask=attn_bias_24,
            dropout_p=0.0,
        )
        contiguous_97 = contiguous_98 = contiguous_99 = attn_bias_24 = None
        transpose_99 = scaled_dot_product_attention_24.transpose(-1, -2)
        scaled_dot_product_attention_24 = None
        x_514 = transpose_99.reshape(1, -1, 14, 14)
        transpose_99 = None
        x_515 = torch.conv2d(
            x_514,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_514 = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_proj_parameters_bias_ = (None)
        x_516 = torch.nn.functional.dropout(x_515, 0.0, False, False)
        x_515 = None
        x_517 = x_510 + x_516
        x_510 = x_516 = None
        x_518 = x_517.permute(0, 2, 3, 1)
        x_519 = torch.nn.functional.layer_norm(
            x_518,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_518 = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm2_parameters_bias_ = (None)
        x_520 = x_519.permute(0, 3, 1, 2)
        x_519 = None
        x_521 = torch.conv2d(
            x_520,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_520 = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_522 = torch._C._nn.gelu(x_521)
        x_521 = None
        x_523 = torch.nn.functional.dropout(x_522, 0.0, False, False)
        x_522 = None
        x_524 = torch.conv2d(
            x_523,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_523 = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_525 = x_517 + x_524
        x_517 = x_524 = None
        x_526 = x_525.permute(0, 2, 3, 1)
        x_527 = torch.nn.functional.layer_norm(
            x_526,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_526 = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm1_parameters_bias_ = (None)
        x_528 = x_527.permute(0, 3, 1, 2)
        x_527 = None
        conv2d_175 = torch.conv2d(
            x_528,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_528 = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_50 = conv2d_175.view(1, 40, 96, -1)
        conv2d_175 = None
        chunk_25 = view_50.chunk(3, dim=2)
        view_50 = None
        q_25 = chunk_25[0]
        k_25 = chunk_25[1]
        v_25 = chunk_25[2]
        chunk_25 = None
        relative_position_bias_50 = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_51 = relative_position_bias_50.view((196, 196, 40))
        relative_position_bias_50 = None
        relative_position_bias_51 = view_51.permute(2, 0, 1)
        view_51 = None
        unsqueeze_25 = relative_position_bias_51.unsqueeze(0)
        relative_position_bias_51 = None
        attn_bias_25 = unsqueeze_25.contiguous()
        unsqueeze_25 = None
        transpose_100 = q_25.transpose(-1, -2)
        q_25 = None
        contiguous_101 = transpose_100.contiguous()
        transpose_100 = None
        transpose_101 = k_25.transpose(-1, -2)
        k_25 = None
        contiguous_102 = transpose_101.contiguous()
        transpose_101 = None
        transpose_102 = v_25.transpose(-1, -2)
        v_25 = None
        contiguous_103 = transpose_102.contiguous()
        transpose_102 = None
        scaled_dot_product_attention_25 = torch._C._nn.scaled_dot_product_attention(
            contiguous_101,
            contiguous_102,
            contiguous_103,
            attn_mask=attn_bias_25,
            dropout_p=0.0,
        )
        contiguous_101 = contiguous_102 = contiguous_103 = attn_bias_25 = None
        transpose_103 = scaled_dot_product_attention_25.transpose(-1, -2)
        scaled_dot_product_attention_25 = None
        x_529 = transpose_103.reshape(1, -1, 14, 14)
        transpose_103 = None
        x_530 = torch.conv2d(
            x_529,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_529 = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_proj_parameters_bias_ = (None)
        x_531 = torch.nn.functional.dropout(x_530, 0.0, False, False)
        x_530 = None
        x_532 = x_525 + x_531
        x_525 = x_531 = None
        x_533 = x_532.permute(0, 2, 3, 1)
        x_534 = torch.nn.functional.layer_norm(
            x_533,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_533 = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm2_parameters_bias_ = (None)
        x_535 = x_534.permute(0, 3, 1, 2)
        x_534 = None
        x_536 = torch.conv2d(
            x_535,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_535 = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_537 = torch._C._nn.gelu(x_536)
        x_536 = None
        x_538 = torch.nn.functional.dropout(x_537, 0.0, False, False)
        x_537 = None
        x_539 = torch.conv2d(
            x_538,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_538 = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_540 = x_532 + x_539
        x_532 = x_539 = None
        x_541 = x_540.permute(0, 2, 3, 1)
        x_542 = torch.nn.functional.layer_norm(
            x_541,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_541 = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm1_parameters_bias_ = (None)
        x_543 = x_542.permute(0, 3, 1, 2)
        x_542 = None
        conv2d_179 = torch.conv2d(
            x_543,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_543 = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_52 = conv2d_179.view(1, 40, 96, -1)
        conv2d_179 = None
        chunk_26 = view_52.chunk(3, dim=2)
        view_52 = None
        q_26 = chunk_26[0]
        k_26 = chunk_26[1]
        v_26 = chunk_26[2]
        chunk_26 = None
        relative_position_bias_52 = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_53 = relative_position_bias_52.view((196, 196, 40))
        relative_position_bias_52 = None
        relative_position_bias_53 = view_53.permute(2, 0, 1)
        view_53 = None
        unsqueeze_26 = relative_position_bias_53.unsqueeze(0)
        relative_position_bias_53 = None
        attn_bias_26 = unsqueeze_26.contiguous()
        unsqueeze_26 = None
        transpose_104 = q_26.transpose(-1, -2)
        q_26 = None
        contiguous_105 = transpose_104.contiguous()
        transpose_104 = None
        transpose_105 = k_26.transpose(-1, -2)
        k_26 = None
        contiguous_106 = transpose_105.contiguous()
        transpose_105 = None
        transpose_106 = v_26.transpose(-1, -2)
        v_26 = None
        contiguous_107 = transpose_106.contiguous()
        transpose_106 = None
        scaled_dot_product_attention_26 = torch._C._nn.scaled_dot_product_attention(
            contiguous_105,
            contiguous_106,
            contiguous_107,
            attn_mask=attn_bias_26,
            dropout_p=0.0,
        )
        contiguous_105 = contiguous_106 = contiguous_107 = attn_bias_26 = None
        transpose_107 = scaled_dot_product_attention_26.transpose(-1, -2)
        scaled_dot_product_attention_26 = None
        x_544 = transpose_107.reshape(1, -1, 14, 14)
        transpose_107 = None
        x_545 = torch.conv2d(
            x_544,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_544 = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_proj_parameters_bias_ = (None)
        x_546 = torch.nn.functional.dropout(x_545, 0.0, False, False)
        x_545 = None
        x_547 = x_540 + x_546
        x_540 = x_546 = None
        x_548 = x_547.permute(0, 2, 3, 1)
        x_549 = torch.nn.functional.layer_norm(
            x_548,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_548 = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm2_parameters_bias_ = (None)
        x_550 = x_549.permute(0, 3, 1, 2)
        x_549 = None
        x_551 = torch.conv2d(
            x_550,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_550 = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_552 = torch._C._nn.gelu(x_551)
        x_551 = None
        x_553 = torch.nn.functional.dropout(x_552, 0.0, False, False)
        x_552 = None
        x_554 = torch.conv2d(
            x_553,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_553 = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_555 = x_547 + x_554
        x_547 = x_554 = None
        x_556 = x_555.permute(0, 2, 3, 1)
        x_557 = torch.nn.functional.layer_norm(
            x_556,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_556 = l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_norm1_parameters_bias_ = (None)
        x_558 = x_557.permute(0, 3, 1, 2)
        x_557 = None
        conv2d_183 = torch.conv2d(
            x_558,
            l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_558 = l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_54 = conv2d_183.view(1, 40, 96, -1)
        conv2d_183 = None
        chunk_27 = view_54.chunk(3, dim=2)
        view_54 = None
        q_27 = chunk_27[0]
        k_27 = chunk_27[1]
        v_27 = chunk_27[2]
        chunk_27 = None
        relative_position_bias_54 = l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_55 = relative_position_bias_54.view((196, 196, 40))
        relative_position_bias_54 = None
        relative_position_bias_55 = view_55.permute(2, 0, 1)
        view_55 = None
        unsqueeze_27 = relative_position_bias_55.unsqueeze(0)
        relative_position_bias_55 = None
        attn_bias_27 = unsqueeze_27.contiguous()
        unsqueeze_27 = None
        transpose_108 = q_27.transpose(-1, -2)
        q_27 = None
        contiguous_109 = transpose_108.contiguous()
        transpose_108 = None
        transpose_109 = k_27.transpose(-1, -2)
        k_27 = None
        contiguous_110 = transpose_109.contiguous()
        transpose_109 = None
        transpose_110 = v_27.transpose(-1, -2)
        v_27 = None
        contiguous_111 = transpose_110.contiguous()
        transpose_110 = None
        scaled_dot_product_attention_27 = torch._C._nn.scaled_dot_product_attention(
            contiguous_109,
            contiguous_110,
            contiguous_111,
            attn_mask=attn_bias_27,
            dropout_p=0.0,
        )
        contiguous_109 = contiguous_110 = contiguous_111 = attn_bias_27 = None
        transpose_111 = scaled_dot_product_attention_27.transpose(-1, -2)
        scaled_dot_product_attention_27 = None
        x_559 = transpose_111.reshape(1, -1, 14, 14)
        transpose_111 = None
        x_560 = torch.conv2d(
            x_559,
            l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_559 = l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_proj_parameters_bias_ = (None)
        x_561 = torch.nn.functional.dropout(x_560, 0.0, False, False)
        x_560 = None
        x_562 = x_555 + x_561
        x_555 = x_561 = None
        x_563 = x_562.permute(0, 2, 3, 1)
        x_564 = torch.nn.functional.layer_norm(
            x_563,
            (1280,),
            l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_563 = l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_norm2_parameters_bias_ = (None)
        x_565 = x_564.permute(0, 3, 1, 2)
        x_564 = None
        x_566 = torch.conv2d(
            x_565,
            l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_565 = l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_567 = torch._C._nn.gelu(x_566)
        x_566 = None
        x_568 = torch.nn.functional.dropout(x_567, 0.0, False, False)
        x_567 = None
        x_569 = torch.conv2d(
            x_568,
            l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_568 = l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_570 = x_562 + x_569
        x_562 = x_569 = None
        x_571 = torch._C._nn.avg_pool2d(x_570, 2, 2, 0, False, True, None)
        x_572 = torch.conv2d(
            x_571,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_571 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_ = (None)
        x_573 = x_570.permute(0, 2, 3, 1)
        x_570 = None
        x_574 = torch.nn.functional.layer_norm(
            x_573,
            (1280,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_,
            1e-06,
        )
        x_573 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_ = (None)
        x_575 = x_574.permute(0, 3, 1, 2)
        x_574 = None
        x_576 = torch._C._nn.avg_pool2d(x_575, 2, 2, 0, False, True, None)
        x_575 = None
        conv2d_188 = torch.conv2d(
            x_576,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_576 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_56 = conv2d_188.view(1, 64, 96, -1)
        conv2d_188 = None
        chunk_28 = view_56.chunk(3, dim=2)
        view_56 = None
        q_28 = chunk_28[0]
        k_28 = chunk_28[1]
        v_28 = chunk_28[2]
        chunk_28 = None
        relative_position_bias_56 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_57 = relative_position_bias_56.view((49, 49, 64))
        relative_position_bias_56 = None
        relative_position_bias_57 = view_57.permute(2, 0, 1)
        view_57 = None
        unsqueeze_28 = relative_position_bias_57.unsqueeze(0)
        relative_position_bias_57 = None
        attn_bias_28 = unsqueeze_28.contiguous()
        unsqueeze_28 = None
        transpose_112 = q_28.transpose(-1, -2)
        q_28 = None
        contiguous_113 = transpose_112.contiguous()
        transpose_112 = None
        transpose_113 = k_28.transpose(-1, -2)
        k_28 = None
        contiguous_114 = transpose_113.contiguous()
        transpose_113 = None
        transpose_114 = v_28.transpose(-1, -2)
        v_28 = None
        contiguous_115 = transpose_114.contiguous()
        transpose_114 = None
        scaled_dot_product_attention_28 = torch._C._nn.scaled_dot_product_attention(
            contiguous_113,
            contiguous_114,
            contiguous_115,
            attn_mask=attn_bias_28,
            dropout_p=0.0,
        )
        contiguous_113 = contiguous_114 = contiguous_115 = attn_bias_28 = None
        transpose_115 = scaled_dot_product_attention_28.transpose(-1, -2)
        scaled_dot_product_attention_28 = None
        x_577 = transpose_115.reshape(1, -1, 7, 7)
        transpose_115 = None
        x_578 = torch.conv2d(
            x_577,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_577 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_579 = torch.nn.functional.dropout(x_578, 0.0, False, False)
        x_578 = None
        x_580 = x_572 + x_579
        x_572 = x_579 = None
        x_581 = x_580.permute(0, 2, 3, 1)
        x_582 = torch.nn.functional.layer_norm(
            x_581,
            (2048,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_581 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_583 = x_582.permute(0, 3, 1, 2)
        x_582 = None
        x_584 = torch.conv2d(
            x_583,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_583 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_585 = torch._C._nn.gelu(x_584)
        x_584 = None
        x_586 = torch.nn.functional.dropout(x_585, 0.0, False, False)
        x_585 = None
        x_587 = torch.conv2d(
            x_586,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_586 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_588 = x_580 + x_587
        x_580 = x_587 = None
        x_589 = x_588.permute(0, 2, 3, 1)
        x_590 = torch.nn.functional.layer_norm(
            x_589,
            (2048,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_589 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_591 = x_590.permute(0, 3, 1, 2)
        x_590 = None
        conv2d_192 = torch.conv2d(
            x_591,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_591 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_58 = conv2d_192.view(1, 64, 96, -1)
        conv2d_192 = None
        chunk_29 = view_58.chunk(3, dim=2)
        view_58 = None
        q_29 = chunk_29[0]
        k_29 = chunk_29[1]
        v_29 = chunk_29[2]
        chunk_29 = None
        relative_position_bias_58 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_59 = relative_position_bias_58.view((49, 49, 64))
        relative_position_bias_58 = None
        relative_position_bias_59 = view_59.permute(2, 0, 1)
        view_59 = None
        unsqueeze_29 = relative_position_bias_59.unsqueeze(0)
        relative_position_bias_59 = None
        attn_bias_29 = unsqueeze_29.contiguous()
        unsqueeze_29 = None
        transpose_116 = q_29.transpose(-1, -2)
        q_29 = None
        contiguous_117 = transpose_116.contiguous()
        transpose_116 = None
        transpose_117 = k_29.transpose(-1, -2)
        k_29 = None
        contiguous_118 = transpose_117.contiguous()
        transpose_117 = None
        transpose_118 = v_29.transpose(-1, -2)
        v_29 = None
        contiguous_119 = transpose_118.contiguous()
        transpose_118 = None
        scaled_dot_product_attention_29 = torch._C._nn.scaled_dot_product_attention(
            contiguous_117,
            contiguous_118,
            contiguous_119,
            attn_mask=attn_bias_29,
            dropout_p=0.0,
        )
        contiguous_117 = contiguous_118 = contiguous_119 = attn_bias_29 = None
        transpose_119 = scaled_dot_product_attention_29.transpose(-1, -2)
        scaled_dot_product_attention_29 = None
        x_592 = transpose_119.reshape(1, -1, 7, 7)
        transpose_119 = None
        x_593 = torch.conv2d(
            x_592,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_592 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_594 = torch.nn.functional.dropout(x_593, 0.0, False, False)
        x_593 = None
        x_595 = x_588 + x_594
        x_588 = x_594 = None
        x_596 = x_595.permute(0, 2, 3, 1)
        x_597 = torch.nn.functional.layer_norm(
            x_596,
            (2048,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_596 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_598 = x_597.permute(0, 3, 1, 2)
        x_597 = None
        x_599 = torch.conv2d(
            x_598,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_598 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_600 = torch._C._nn.gelu(x_599)
        x_599 = None
        x_601 = torch.nn.functional.dropout(x_600, 0.0, False, False)
        x_600 = None
        x_602 = torch.conv2d(
            x_601,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_601 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_603 = x_595 + x_602
        x_595 = x_602 = None
        x_604 = torch.nn.functional.adaptive_avg_pool2d(x_603, 1)
        x_603 = None
        x_605 = x_604.permute(0, 2, 3, 1)
        x_604 = None
        x_606 = torch.nn.functional.layer_norm(
            x_605,
            (2048,),
            l_self_modules_head_modules_norm_parameters_weight_,
            l_self_modules_head_modules_norm_parameters_bias_,
            1e-06,
        )
        x_605 = (
            l_self_modules_head_modules_norm_parameters_weight_
        ) = l_self_modules_head_modules_norm_parameters_bias_ = None
        x_607 = x_606.permute(0, 3, 1, 2)
        x_606 = None
        x_608 = x_607.flatten(1, -1)
        x_607 = None
        input_1 = torch._C._nn.linear(
            x_608,
            l_self_modules_head_modules_pre_logits_modules_fc_parameters_weight_,
            l_self_modules_head_modules_pre_logits_modules_fc_parameters_bias_,
        )
        x_608 = (
            l_self_modules_head_modules_pre_logits_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_pre_logits_modules_fc_parameters_bias_ = None
        input_2 = input_1.tanh()
        input_1 = None
        x_609 = torch.nn.functional.dropout(input_2, 0.0, False, False)
        input_2 = None
        x_610 = torch._C._nn.linear(
            x_609,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_609 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_610,)
