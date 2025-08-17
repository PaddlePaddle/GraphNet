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
        x_5 = torch.nn.functional.batch_norm(
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
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_5 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_ = (None)
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_6 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        x_8 = torch._C._nn.gelu(x_7)
        x_7 = None
        x_9 = torch.conv2d(
            x_8,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            768,
        )
        x_8 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_ = (None)
        x_10 = torch.nn.functional.batch_norm(
            x_9,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_9 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_11 = torch._C._nn.gelu(x_10)
        x_10 = None
        x_se = x_11.mean((2, 3), keepdim=True)
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
        x_12 = x_11 * sigmoid
        x_11 = sigmoid = None
        x_13 = torch.conv2d(
            x_12,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_12 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_bias_ = (None)
        x_14 = x_13 + x_4
        x_13 = x_4 = None
        x_15 = torch.nn.functional.batch_norm(
            x_14,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_bias_ = (None)
        x_16 = torch.conv2d(
            x_15,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_15 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_ = (None)
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_16 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_18 = torch._C._nn.gelu(x_17)
        x_17 = None
        x_19 = torch.conv2d(
            x_18,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_18 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_ = (None)
        x_20 = torch.nn.functional.batch_norm(
            x_19,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_19 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_21 = torch._C._nn.gelu(x_20)
        x_20 = None
        x_se_4 = x_21.mean((2, 3), keepdim=True)
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
        x_22 = x_21 * sigmoid_1
        x_21 = sigmoid_1 = None
        x_23 = torch.conv2d(
            x_22,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_22 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_bias_ = (None)
        x_24 = x_23 + x_14
        x_23 = x_14 = None
        x_25 = torch._C._nn.avg_pool2d(x_24, 2, 2, 0, False, True, None)
        x_26 = torch.conv2d(
            x_25,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_25 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_ = (None)
        x_27 = torch.nn.functional.batch_norm(
            x_24,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_24 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_bias_ = (None)
        x_28 = torch.conv2d(
            x_27,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_27 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_ = (None)
        x_29 = torch.nn.functional.batch_norm(
            x_28,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_28 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        x_30 = torch._C._nn.gelu(x_29)
        x_29 = None
        x_31 = torch.conv2d(
            x_30,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1536,
        )
        x_30 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_ = (None)
        x_32 = torch.nn.functional.batch_norm(
            x_31,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_31 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_33 = torch._C._nn.gelu(x_32)
        x_32 = None
        x_se_8 = x_33.mean((2, 3), keepdim=True)
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
        x_34 = x_33 * sigmoid_2
        x_33 = sigmoid_2 = None
        x_35 = torch.conv2d(
            x_34,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_34 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_bias_ = (None)
        x_36 = x_35 + x_26
        x_35 = x_26 = None
        x_37 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_bias_ = (None)
        x_38 = torch.conv2d(
            x_37,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_37 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_ = (None)
        x_39 = torch.nn.functional.batch_norm(
            x_38,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_38 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_40 = torch._C._nn.gelu(x_39)
        x_39 = None
        x_41 = torch.conv2d(
            x_40,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        x_40 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_ = (None)
        x_42 = torch.nn.functional.batch_norm(
            x_41,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_41 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_43 = torch._C._nn.gelu(x_42)
        x_42 = None
        x_se_12 = x_43.mean((2, 3), keepdim=True)
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
        x_44 = x_43 * sigmoid_3
        x_43 = sigmoid_3 = None
        x_45 = torch.conv2d(
            x_44,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_44 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_bias_ = (None)
        x_46 = x_45 + x_36
        x_45 = x_36 = None
        x_47 = torch.nn.functional.batch_norm(
            x_46,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_bias_ = (None)
        x_48 = torch.conv2d(
            x_47,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_47 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_1x1_parameters_weight_ = (None)
        x_49 = torch.nn.functional.batch_norm(
            x_48,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_48 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (None)
        x_50 = torch._C._nn.gelu(x_49)
        x_49 = None
        x_51 = torch.conv2d(
            x_50,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        x_50 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_kxk_parameters_weight_ = (None)
        x_52 = torch.nn.functional.batch_norm(
            x_51,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_51 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (None)
        x_53 = torch._C._nn.gelu(x_52)
        x_52 = None
        x_se_16 = x_53.mean((2, 3), keepdim=True)
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
        x_54 = x_53 * sigmoid_4
        x_53 = sigmoid_4 = None
        x_55 = torch.conv2d(
            x_54,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_54 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_bias_ = (None)
        x_56 = x_55 + x_46
        x_55 = x_46 = None
        x_57 = torch.nn.functional.batch_norm(
            x_56,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_parameters_bias_ = (None)
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_57 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_1x1_parameters_weight_ = (None)
        x_59 = torch.nn.functional.batch_norm(
            x_58,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_58 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_bias_ = (None)
        x_60 = torch._C._nn.gelu(x_59)
        x_59 = None
        x_61 = torch.conv2d(
            x_60,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        x_60 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_kxk_parameters_weight_ = (None)
        x_62 = torch.nn.functional.batch_norm(
            x_61,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_61 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_bias_ = (None)
        x_63 = torch._C._nn.gelu(x_62)
        x_62 = None
        x_se_20 = x_63.mean((2, 3), keepdim=True)
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
        x_64 = x_63 * sigmoid_5
        x_63 = sigmoid_5 = None
        x_65 = torch.conv2d(
            x_64,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_64 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_1x1_parameters_bias_ = (None)
        x_66 = x_65 + x_56
        x_65 = x_56 = None
        x_67 = torch.nn.functional.batch_norm(
            x_66,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_parameters_bias_ = (None)
        x_68 = torch.conv2d(
            x_67,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_67 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_1x1_parameters_weight_ = (None)
        x_69 = torch.nn.functional.batch_norm(
            x_68,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_68 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_bias_ = (None)
        x_70 = torch._C._nn.gelu(x_69)
        x_69 = None
        x_71 = torch.conv2d(
            x_70,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        x_70 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_kxk_parameters_weight_ = (None)
        x_72 = torch.nn.functional.batch_norm(
            x_71,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_71 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_bias_ = (None)
        x_73 = torch._C._nn.gelu(x_72)
        x_72 = None
        x_se_24 = x_73.mean((2, 3), keepdim=True)
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
        x_74 = x_73 * sigmoid_6
        x_73 = sigmoid_6 = None
        x_75 = torch.conv2d(
            x_74,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_74 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_1x1_parameters_bias_ = (None)
        x_76 = x_75 + x_66
        x_75 = x_66 = None
        x_77 = torch.nn.functional.batch_norm(
            x_76,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_parameters_bias_ = (None)
        x_78 = torch.conv2d(
            x_77,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_77 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_1x1_parameters_weight_ = (None)
        x_79 = torch.nn.functional.batch_norm(
            x_78,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_78 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_bias_ = (None)
        x_80 = torch._C._nn.gelu(x_79)
        x_79 = None
        x_81 = torch.conv2d(
            x_80,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        x_80 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_kxk_parameters_weight_ = (None)
        x_82 = torch.nn.functional.batch_norm(
            x_81,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_81 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_bias_ = (None)
        x_83 = torch._C._nn.gelu(x_82)
        x_82 = None
        x_se_28 = x_83.mean((2, 3), keepdim=True)
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
        x_84 = x_83 * sigmoid_7
        x_83 = sigmoid_7 = None
        x_85 = torch.conv2d(
            x_84,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_84 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_1x1_parameters_bias_ = (None)
        x_86 = x_85 + x_76
        x_85 = x_76 = None
        x_87 = torch.nn.functional.batch_norm(
            x_86,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_pre_norm_parameters_bias_ = (None)
        x_88 = torch.conv2d(
            x_87,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_87 = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_1x1_parameters_weight_ = (None)
        x_89 = torch.nn.functional.batch_norm(
            x_88,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_88 = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_bias_ = (None)
        x_90 = torch._C._nn.gelu(x_89)
        x_89 = None
        x_91 = torch.conv2d(
            x_90,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        x_90 = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_kxk_parameters_weight_ = (None)
        x_92 = torch.nn.functional.batch_norm(
            x_91,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_91 = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_bias_ = (None)
        x_93 = torch._C._nn.gelu(x_92)
        x_92 = None
        x_se_32 = x_93.mean((2, 3), keepdim=True)
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
        x_94 = x_93 * sigmoid_8
        x_93 = sigmoid_8 = None
        x_95 = torch.conv2d(
            x_94,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_94 = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv3_1x1_parameters_bias_ = (None)
        x_96 = x_95 + x_86
        x_95 = x_86 = None
        x_97 = torch.nn.functional.batch_norm(
            x_96,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_pre_norm_parameters_bias_ = (None)
        x_98 = torch.conv2d(
            x_97,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_97 = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_conv1_1x1_parameters_weight_ = (None)
        x_99 = torch.nn.functional.batch_norm(
            x_98,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_98 = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_bias_ = (None)
        x_100 = torch._C._nn.gelu(x_99)
        x_99 = None
        x_101 = torch.conv2d(
            x_100,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        x_100 = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_conv2_kxk_parameters_weight_ = (None)
        x_102 = torch.nn.functional.batch_norm(
            x_101,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_101 = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_bias_ = (None)
        x_103 = torch._C._nn.gelu(x_102)
        x_102 = None
        x_se_36 = x_103.mean((2, 3), keepdim=True)
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
        x_104 = x_103 * sigmoid_9
        x_103 = sigmoid_9 = None
        x_105 = torch.conv2d(
            x_104,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_104 = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_conv3_1x1_parameters_bias_ = (None)
        x_106 = x_105 + x_96
        x_105 = x_96 = None
        x_107 = torch.nn.functional.batch_norm(
            x_106,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_pre_norm_parameters_bias_ = (None)
        x_108 = torch.conv2d(
            x_107,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_107 = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_conv1_1x1_parameters_weight_ = (None)
        x_109 = torch.nn.functional.batch_norm(
            x_108,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_108 = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm1_parameters_bias_ = (None)
        x_110 = torch._C._nn.gelu(x_109)
        x_109 = None
        x_111 = torch.conv2d(
            x_110,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        x_110 = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_conv2_kxk_parameters_weight_ = (None)
        x_112 = torch.nn.functional.batch_norm(
            x_111,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_111 = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_norm2_parameters_bias_ = (None)
        x_113 = torch._C._nn.gelu(x_112)
        x_112 = None
        x_se_40 = x_113.mean((2, 3), keepdim=True)
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
        x_114 = x_113 * sigmoid_10
        x_113 = sigmoid_10 = None
        x_115 = torch.conv2d(
            x_114,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_114 = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_8_modules_conv3_1x1_parameters_bias_ = (None)
        x_116 = x_115 + x_106
        x_115 = x_106 = None
        x_117 = torch.nn.functional.batch_norm(
            x_116,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_pre_norm_parameters_bias_ = (None)
        x_118 = torch.conv2d(
            x_117,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_117 = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_conv1_1x1_parameters_weight_ = (None)
        x_119 = torch.nn.functional.batch_norm(
            x_118,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_118 = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm1_parameters_bias_ = (None)
        x_120 = torch._C._nn.gelu(x_119)
        x_119 = None
        x_121 = torch.conv2d(
            x_120,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        x_120 = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_conv2_kxk_parameters_weight_ = (None)
        x_122 = torch.nn.functional.batch_norm(
            x_121,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_121 = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_norm2_parameters_bias_ = (None)
        x_123 = torch._C._nn.gelu(x_122)
        x_122 = None
        x_se_44 = x_123.mean((2, 3), keepdim=True)
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
        x_124 = x_123 * sigmoid_11
        x_123 = sigmoid_11 = None
        x_125 = torch.conv2d(
            x_124,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_124 = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_9_modules_conv3_1x1_parameters_bias_ = (None)
        x_126 = x_125 + x_116
        x_125 = x_116 = None
        x_127 = torch.nn.functional.batch_norm(
            x_126,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_pre_norm_parameters_bias_ = (None)
        x_128 = torch.conv2d(
            x_127,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_127 = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_conv1_1x1_parameters_weight_ = (None)
        x_129 = torch.nn.functional.batch_norm(
            x_128,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_128 = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm1_parameters_bias_ = (None)
        x_130 = torch._C._nn.gelu(x_129)
        x_129 = None
        x_131 = torch.conv2d(
            x_130,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        x_130 = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_conv2_kxk_parameters_weight_ = (None)
        x_132 = torch.nn.functional.batch_norm(
            x_131,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_131 = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_norm2_parameters_bias_ = (None)
        x_133 = torch._C._nn.gelu(x_132)
        x_132 = None
        x_se_48 = x_133.mean((2, 3), keepdim=True)
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
        x_134 = x_133 * sigmoid_12
        x_133 = sigmoid_12 = None
        x_135 = torch.conv2d(
            x_134,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_134 = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_10_modules_conv3_1x1_parameters_bias_ = (None)
        x_136 = x_135 + x_126
        x_135 = x_126 = None
        x_137 = torch.nn.functional.batch_norm(
            x_136,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_pre_norm_parameters_bias_ = (None)
        x_138 = torch.conv2d(
            x_137,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_137 = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_conv1_1x1_parameters_weight_ = (None)
        x_139 = torch.nn.functional.batch_norm(
            x_138,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_138 = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm1_parameters_bias_ = (None)
        x_140 = torch._C._nn.gelu(x_139)
        x_139 = None
        x_141 = torch.conv2d(
            x_140,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        x_140 = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_conv2_kxk_parameters_weight_ = (None)
        x_142 = torch.nn.functional.batch_norm(
            x_141,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_141 = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_norm2_parameters_bias_ = (None)
        x_143 = torch._C._nn.gelu(x_142)
        x_142 = None
        x_se_52 = x_143.mean((2, 3), keepdim=True)
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
        x_144 = x_143 * sigmoid_13
        x_143 = sigmoid_13 = None
        x_145 = torch.conv2d(
            x_144,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_144 = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_11_modules_conv3_1x1_parameters_bias_ = (None)
        x_146 = x_145 + x_136
        x_145 = x_136 = None
        x_147 = torch._C._nn.avg_pool2d(x_146, 2, 2, 0, False, True, None)
        x_148 = torch.conv2d(
            x_147,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_147 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_ = (None)
        x_149 = x_146.permute(0, 2, 3, 1)
        x_146 = None
        x_150 = torch.nn.functional.layer_norm(
            x_149,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_,
            1e-06,
        )
        x_149 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_ = (None)
        x_151 = x_150.permute(0, 3, 1, 2)
        x_150 = None
        x_152 = torch._C._nn.avg_pool2d(x_151, 2, 2, 0, False, True, None)
        x_151 = None
        conv2d_74 = torch.conv2d(
            x_152,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_152 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        view = conv2d_74.view(1, 24, 96, -1)
        conv2d_74 = None
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
        view_1 = relative_position_bias.view((196, 196, 24))
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
        x_153 = transpose_3.reshape(1, -1, 14, 14)
        transpose_3 = None
        x_154 = torch.conv2d(
            x_153,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_153 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_155 = torch.nn.functional.dropout(x_154, 0.0, False, False)
        x_154 = None
        x_156 = x_148 + x_155
        x_148 = x_155 = None
        x_157 = x_156.permute(0, 2, 3, 1)
        x_158 = torch.nn.functional.layer_norm(
            x_157,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_157 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_159 = x_158.permute(0, 3, 1, 2)
        x_158 = None
        x_160 = torch.conv2d(
            x_159,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_159 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_161 = torch._C._nn.gelu(x_160)
        x_160 = None
        x_162 = torch.nn.functional.dropout(x_161, 0.0, False, False)
        x_161 = None
        x_163 = torch.conv2d(
            x_162,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_162 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_164 = x_156 + x_163
        x_156 = x_163 = None
        x_165 = x_164.permute(0, 2, 3, 1)
        x_166 = torch.nn.functional.layer_norm(
            x_165,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_165 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_167 = x_166.permute(0, 3, 1, 2)
        x_166 = None
        conv2d_78 = torch.conv2d(
            x_167,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_167 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_2 = conv2d_78.view(1, 24, 96, -1)
        conv2d_78 = None
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
        view_3 = relative_position_bias_2.view((196, 196, 24))
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
        x_168 = transpose_7.reshape(1, -1, 14, 14)
        transpose_7 = None
        x_169 = torch.conv2d(
            x_168,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_168 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_170 = torch.nn.functional.dropout(x_169, 0.0, False, False)
        x_169 = None
        x_171 = x_164 + x_170
        x_164 = x_170 = None
        x_172 = x_171.permute(0, 2, 3, 1)
        x_173 = torch.nn.functional.layer_norm(
            x_172,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_172 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_174 = x_173.permute(0, 3, 1, 2)
        x_173 = None
        x_175 = torch.conv2d(
            x_174,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_174 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_176 = torch._C._nn.gelu(x_175)
        x_175 = None
        x_177 = torch.nn.functional.dropout(x_176, 0.0, False, False)
        x_176 = None
        x_178 = torch.conv2d(
            x_177,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_177 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_179 = x_171 + x_178
        x_171 = x_178 = None
        x_180 = x_179.permute(0, 2, 3, 1)
        x_181 = torch.nn.functional.layer_norm(
            x_180,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_180 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (None)
        x_182 = x_181.permute(0, 3, 1, 2)
        x_181 = None
        conv2d_82 = torch.conv2d(
            x_182,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_182 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_4 = conv2d_82.view(1, 24, 96, -1)
        conv2d_82 = None
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
        view_5 = relative_position_bias_4.view((196, 196, 24))
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
        x_183 = transpose_11.reshape(1, -1, 14, 14)
        transpose_11 = None
        x_184 = torch.conv2d(
            x_183,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_183 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_ = (None)
        x_185 = torch.nn.functional.dropout(x_184, 0.0, False, False)
        x_184 = None
        x_186 = x_179 + x_185
        x_179 = x_185 = None
        x_187 = x_186.permute(0, 2, 3, 1)
        x_188 = torch.nn.functional.layer_norm(
            x_187,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_187 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (None)
        x_189 = x_188.permute(0, 3, 1, 2)
        x_188 = None
        x_190 = torch.conv2d(
            x_189,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_189 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_191 = torch._C._nn.gelu(x_190)
        x_190 = None
        x_192 = torch.nn.functional.dropout(x_191, 0.0, False, False)
        x_191 = None
        x_193 = torch.conv2d(
            x_192,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_192 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_194 = x_186 + x_193
        x_186 = x_193 = None
        x_195 = x_194.permute(0, 2, 3, 1)
        x_196 = torch.nn.functional.layer_norm(
            x_195,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_195 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_ = (None)
        x_197 = x_196.permute(0, 3, 1, 2)
        x_196 = None
        conv2d_86 = torch.conv2d(
            x_197,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_197 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_6 = conv2d_86.view(1, 24, 96, -1)
        conv2d_86 = None
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
        view_7 = relative_position_bias_6.view((196, 196, 24))
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
        x_198 = transpose_15.reshape(1, -1, 14, 14)
        transpose_15 = None
        x_199 = torch.conv2d(
            x_198,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_198 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_ = (None)
        x_200 = torch.nn.functional.dropout(x_199, 0.0, False, False)
        x_199 = None
        x_201 = x_194 + x_200
        x_194 = x_200 = None
        x_202 = x_201.permute(0, 2, 3, 1)
        x_203 = torch.nn.functional.layer_norm(
            x_202,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_202 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_ = (None)
        x_204 = x_203.permute(0, 3, 1, 2)
        x_203 = None
        x_205 = torch.conv2d(
            x_204,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_204 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_206 = torch._C._nn.gelu(x_205)
        x_205 = None
        x_207 = torch.nn.functional.dropout(x_206, 0.0, False, False)
        x_206 = None
        x_208 = torch.conv2d(
            x_207,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_207 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_209 = x_201 + x_208
        x_201 = x_208 = None
        x_210 = x_209.permute(0, 2, 3, 1)
        x_211 = torch.nn.functional.layer_norm(
            x_210,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_210 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_ = (None)
        x_212 = x_211.permute(0, 3, 1, 2)
        x_211 = None
        conv2d_90 = torch.conv2d(
            x_212,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_212 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_8 = conv2d_90.view(1, 24, 96, -1)
        conv2d_90 = None
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
        view_9 = relative_position_bias_8.view((196, 196, 24))
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
        x_213 = transpose_19.reshape(1, -1, 14, 14)
        transpose_19 = None
        x_214 = torch.conv2d(
            x_213,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_213 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_ = (None)
        x_215 = torch.nn.functional.dropout(x_214, 0.0, False, False)
        x_214 = None
        x_216 = x_209 + x_215
        x_209 = x_215 = None
        x_217 = x_216.permute(0, 2, 3, 1)
        x_218 = torch.nn.functional.layer_norm(
            x_217,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_217 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_ = (None)
        x_219 = x_218.permute(0, 3, 1, 2)
        x_218 = None
        x_220 = torch.conv2d(
            x_219,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_219 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_221 = torch._C._nn.gelu(x_220)
        x_220 = None
        x_222 = torch.nn.functional.dropout(x_221, 0.0, False, False)
        x_221 = None
        x_223 = torch.conv2d(
            x_222,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_222 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_224 = x_216 + x_223
        x_216 = x_223 = None
        x_225 = x_224.permute(0, 2, 3, 1)
        x_226 = torch.nn.functional.layer_norm(
            x_225,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_225 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_ = (None)
        x_227 = x_226.permute(0, 3, 1, 2)
        x_226 = None
        conv2d_94 = torch.conv2d(
            x_227,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_227 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_10 = conv2d_94.view(1, 24, 96, -1)
        conv2d_94 = None
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
        view_11 = relative_position_bias_10.view((196, 196, 24))
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
        x_228 = transpose_23.reshape(1, -1, 14, 14)
        transpose_23 = None
        x_229 = torch.conv2d(
            x_228,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_228 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_ = (None)
        x_230 = torch.nn.functional.dropout(x_229, 0.0, False, False)
        x_229 = None
        x_231 = x_224 + x_230
        x_224 = x_230 = None
        x_232 = x_231.permute(0, 2, 3, 1)
        x_233 = torch.nn.functional.layer_norm(
            x_232,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_232 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_ = (None)
        x_234 = x_233.permute(0, 3, 1, 2)
        x_233 = None
        x_235 = torch.conv2d(
            x_234,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_234 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_236 = torch._C._nn.gelu(x_235)
        x_235 = None
        x_237 = torch.nn.functional.dropout(x_236, 0.0, False, False)
        x_236 = None
        x_238 = torch.conv2d(
            x_237,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_237 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_239 = x_231 + x_238
        x_231 = x_238 = None
        x_240 = x_239.permute(0, 2, 3, 1)
        x_241 = torch.nn.functional.layer_norm(
            x_240,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_240 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_ = (None)
        x_242 = x_241.permute(0, 3, 1, 2)
        x_241 = None
        conv2d_98 = torch.conv2d(
            x_242,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_242 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_12 = conv2d_98.view(1, 24, 96, -1)
        conv2d_98 = None
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
        view_13 = relative_position_bias_12.view((196, 196, 24))
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
        x_243 = transpose_27.reshape(1, -1, 14, 14)
        transpose_27 = None
        x_244 = torch.conv2d(
            x_243,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_243 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_ = (None)
        x_245 = torch.nn.functional.dropout(x_244, 0.0, False, False)
        x_244 = None
        x_246 = x_239 + x_245
        x_239 = x_245 = None
        x_247 = x_246.permute(0, 2, 3, 1)
        x_248 = torch.nn.functional.layer_norm(
            x_247,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_247 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_ = (None)
        x_249 = x_248.permute(0, 3, 1, 2)
        x_248 = None
        x_250 = torch.conv2d(
            x_249,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_249 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_251 = torch._C._nn.gelu(x_250)
        x_250 = None
        x_252 = torch.nn.functional.dropout(x_251, 0.0, False, False)
        x_251 = None
        x_253 = torch.conv2d(
            x_252,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_252 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_254 = x_246 + x_253
        x_246 = x_253 = None
        x_255 = x_254.permute(0, 2, 3, 1)
        x_256 = torch.nn.functional.layer_norm(
            x_255,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_255 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_ = (None)
        x_257 = x_256.permute(0, 3, 1, 2)
        x_256 = None
        conv2d_102 = torch.conv2d(
            x_257,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_257 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_14 = conv2d_102.view(1, 24, 96, -1)
        conv2d_102 = None
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
        view_15 = relative_position_bias_14.view((196, 196, 24))
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
        x_258 = transpose_31.reshape(1, -1, 14, 14)
        transpose_31 = None
        x_259 = torch.conv2d(
            x_258,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_258 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_ = (None)
        x_260 = torch.nn.functional.dropout(x_259, 0.0, False, False)
        x_259 = None
        x_261 = x_254 + x_260
        x_254 = x_260 = None
        x_262 = x_261.permute(0, 2, 3, 1)
        x_263 = torch.nn.functional.layer_norm(
            x_262,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_262 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_ = (None)
        x_264 = x_263.permute(0, 3, 1, 2)
        x_263 = None
        x_265 = torch.conv2d(
            x_264,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_264 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_266 = torch._C._nn.gelu(x_265)
        x_265 = None
        x_267 = torch.nn.functional.dropout(x_266, 0.0, False, False)
        x_266 = None
        x_268 = torch.conv2d(
            x_267,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_267 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_269 = x_261 + x_268
        x_261 = x_268 = None
        x_270 = x_269.permute(0, 2, 3, 1)
        x_271 = torch.nn.functional.layer_norm(
            x_270,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_270 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_ = (None)
        x_272 = x_271.permute(0, 3, 1, 2)
        x_271 = None
        conv2d_106 = torch.conv2d(
            x_272,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_272 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_16 = conv2d_106.view(1, 24, 96, -1)
        conv2d_106 = None
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
        view_17 = relative_position_bias_16.view((196, 196, 24))
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
        x_273 = transpose_35.reshape(1, -1, 14, 14)
        transpose_35 = None
        x_274 = torch.conv2d(
            x_273,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_273 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_ = (None)
        x_275 = torch.nn.functional.dropout(x_274, 0.0, False, False)
        x_274 = None
        x_276 = x_269 + x_275
        x_269 = x_275 = None
        x_277 = x_276.permute(0, 2, 3, 1)
        x_278 = torch.nn.functional.layer_norm(
            x_277,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_277 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_ = (None)
        x_279 = x_278.permute(0, 3, 1, 2)
        x_278 = None
        x_280 = torch.conv2d(
            x_279,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_279 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_281 = torch._C._nn.gelu(x_280)
        x_280 = None
        x_282 = torch.nn.functional.dropout(x_281, 0.0, False, False)
        x_281 = None
        x_283 = torch.conv2d(
            x_282,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_282 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_284 = x_276 + x_283
        x_276 = x_283 = None
        x_285 = x_284.permute(0, 2, 3, 1)
        x_286 = torch.nn.functional.layer_norm(
            x_285,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_285 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_ = (None)
        x_287 = x_286.permute(0, 3, 1, 2)
        x_286 = None
        conv2d_110 = torch.conv2d(
            x_287,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_287 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_18 = conv2d_110.view(1, 24, 96, -1)
        conv2d_110 = None
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
        view_19 = relative_position_bias_18.view((196, 196, 24))
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
        x_288 = transpose_39.reshape(1, -1, 14, 14)
        transpose_39 = None
        x_289 = torch.conv2d(
            x_288,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_288 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_ = (None)
        x_290 = torch.nn.functional.dropout(x_289, 0.0, False, False)
        x_289 = None
        x_291 = x_284 + x_290
        x_284 = x_290 = None
        x_292 = x_291.permute(0, 2, 3, 1)
        x_293 = torch.nn.functional.layer_norm(
            x_292,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_292 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_ = (None)
        x_294 = x_293.permute(0, 3, 1, 2)
        x_293 = None
        x_295 = torch.conv2d(
            x_294,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_294 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_296 = torch._C._nn.gelu(x_295)
        x_295 = None
        x_297 = torch.nn.functional.dropout(x_296, 0.0, False, False)
        x_296 = None
        x_298 = torch.conv2d(
            x_297,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_297 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_299 = x_291 + x_298
        x_291 = x_298 = None
        x_300 = x_299.permute(0, 2, 3, 1)
        x_301 = torch.nn.functional.layer_norm(
            x_300,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_300 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_ = (None)
        x_302 = x_301.permute(0, 3, 1, 2)
        x_301 = None
        conv2d_114 = torch.conv2d(
            x_302,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_302 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_20 = conv2d_114.view(1, 24, 96, -1)
        conv2d_114 = None
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
        view_21 = relative_position_bias_20.view((196, 196, 24))
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
        x_303 = transpose_43.reshape(1, -1, 14, 14)
        transpose_43 = None
        x_304 = torch.conv2d(
            x_303,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_303 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_ = (None)
        x_305 = torch.nn.functional.dropout(x_304, 0.0, False, False)
        x_304 = None
        x_306 = x_299 + x_305
        x_299 = x_305 = None
        x_307 = x_306.permute(0, 2, 3, 1)
        x_308 = torch.nn.functional.layer_norm(
            x_307,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_307 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_ = (None)
        x_309 = x_308.permute(0, 3, 1, 2)
        x_308 = None
        x_310 = torch.conv2d(
            x_309,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_309 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_311 = torch._C._nn.gelu(x_310)
        x_310 = None
        x_312 = torch.nn.functional.dropout(x_311, 0.0, False, False)
        x_311 = None
        x_313 = torch.conv2d(
            x_312,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_312 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_314 = x_306 + x_313
        x_306 = x_313 = None
        x_315 = x_314.permute(0, 2, 3, 1)
        x_316 = torch.nn.functional.layer_norm(
            x_315,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_315 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_ = (None)
        x_317 = x_316.permute(0, 3, 1, 2)
        x_316 = None
        conv2d_118 = torch.conv2d(
            x_317,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_317 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_22 = conv2d_118.view(1, 24, 96, -1)
        conv2d_118 = None
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
        view_23 = relative_position_bias_22.view((196, 196, 24))
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
        x_318 = transpose_47.reshape(1, -1, 14, 14)
        transpose_47 = None
        x_319 = torch.conv2d(
            x_318,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_318 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_ = (None)
        x_320 = torch.nn.functional.dropout(x_319, 0.0, False, False)
        x_319 = None
        x_321 = x_314 + x_320
        x_314 = x_320 = None
        x_322 = x_321.permute(0, 2, 3, 1)
        x_323 = torch.nn.functional.layer_norm(
            x_322,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_322 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_ = (None)
        x_324 = x_323.permute(0, 3, 1, 2)
        x_323 = None
        x_325 = torch.conv2d(
            x_324,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_324 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_326 = torch._C._nn.gelu(x_325)
        x_325 = None
        x_327 = torch.nn.functional.dropout(x_326, 0.0, False, False)
        x_326 = None
        x_328 = torch.conv2d(
            x_327,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_327 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_329 = x_321 + x_328
        x_321 = x_328 = None
        x_330 = x_329.permute(0, 2, 3, 1)
        x_331 = torch.nn.functional.layer_norm(
            x_330,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_330 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_ = (None)
        x_332 = x_331.permute(0, 3, 1, 2)
        x_331 = None
        conv2d_122 = torch.conv2d(
            x_332,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_332 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_24 = conv2d_122.view(1, 24, 96, -1)
        conv2d_122 = None
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
        view_25 = relative_position_bias_24.view((196, 196, 24))
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
        x_333 = transpose_51.reshape(1, -1, 14, 14)
        transpose_51 = None
        x_334 = torch.conv2d(
            x_333,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_333 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_ = (None)
        x_335 = torch.nn.functional.dropout(x_334, 0.0, False, False)
        x_334 = None
        x_336 = x_329 + x_335
        x_329 = x_335 = None
        x_337 = x_336.permute(0, 2, 3, 1)
        x_338 = torch.nn.functional.layer_norm(
            x_337,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_337 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_ = (None)
        x_339 = x_338.permute(0, 3, 1, 2)
        x_338 = None
        x_340 = torch.conv2d(
            x_339,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_339 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_341 = torch._C._nn.gelu(x_340)
        x_340 = None
        x_342 = torch.nn.functional.dropout(x_341, 0.0, False, False)
        x_341 = None
        x_343 = torch.conv2d(
            x_342,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_342 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_344 = x_336 + x_343
        x_336 = x_343 = None
        x_345 = x_344.permute(0, 2, 3, 1)
        x_346 = torch.nn.functional.layer_norm(
            x_345,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_345 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_ = (None)
        x_347 = x_346.permute(0, 3, 1, 2)
        x_346 = None
        conv2d_126 = torch.conv2d(
            x_347,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_347 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_26 = conv2d_126.view(1, 24, 96, -1)
        conv2d_126 = None
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
        view_27 = relative_position_bias_26.view((196, 196, 24))
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
        x_348 = transpose_55.reshape(1, -1, 14, 14)
        transpose_55 = None
        x_349 = torch.conv2d(
            x_348,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_348 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_ = (None)
        x_350 = torch.nn.functional.dropout(x_349, 0.0, False, False)
        x_349 = None
        x_351 = x_344 + x_350
        x_344 = x_350 = None
        x_352 = x_351.permute(0, 2, 3, 1)
        x_353 = torch.nn.functional.layer_norm(
            x_352,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_352 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_ = (None)
        x_354 = x_353.permute(0, 3, 1, 2)
        x_353 = None
        x_355 = torch.conv2d(
            x_354,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_354 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_356 = torch._C._nn.gelu(x_355)
        x_355 = None
        x_357 = torch.nn.functional.dropout(x_356, 0.0, False, False)
        x_356 = None
        x_358 = torch.conv2d(
            x_357,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_357 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_359 = x_351 + x_358
        x_351 = x_358 = None
        x_360 = x_359.permute(0, 2, 3, 1)
        x_361 = torch.nn.functional.layer_norm(
            x_360,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_360 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_bias_ = (None)
        x_362 = x_361.permute(0, 3, 1, 2)
        x_361 = None
        conv2d_130 = torch.conv2d(
            x_362,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_362 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_28 = conv2d_130.view(1, 24, 96, -1)
        conv2d_130 = None
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
        view_29 = relative_position_bias_28.view((196, 196, 24))
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
        x_363 = transpose_59.reshape(1, -1, 14, 14)
        transpose_59 = None
        x_364 = torch.conv2d(
            x_363,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_363 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_ = (None)
        x_365 = torch.nn.functional.dropout(x_364, 0.0, False, False)
        x_364 = None
        x_366 = x_359 + x_365
        x_359 = x_365 = None
        x_367 = x_366.permute(0, 2, 3, 1)
        x_368 = torch.nn.functional.layer_norm(
            x_367,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_367 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_bias_ = (None)
        x_369 = x_368.permute(0, 3, 1, 2)
        x_368 = None
        x_370 = torch.conv2d(
            x_369,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_369 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_371 = torch._C._nn.gelu(x_370)
        x_370 = None
        x_372 = torch.nn.functional.dropout(x_371, 0.0, False, False)
        x_371 = None
        x_373 = torch.conv2d(
            x_372,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_372 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_374 = x_366 + x_373
        x_366 = x_373 = None
        x_375 = x_374.permute(0, 2, 3, 1)
        x_376 = torch.nn.functional.layer_norm(
            x_375,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_375 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_bias_ = (None)
        x_377 = x_376.permute(0, 3, 1, 2)
        x_376 = None
        conv2d_134 = torch.conv2d(
            x_377,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_377 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_30 = conv2d_134.view(1, 24, 96, -1)
        conv2d_134 = None
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
        view_31 = relative_position_bias_30.view((196, 196, 24))
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
        x_378 = transpose_63.reshape(1, -1, 14, 14)
        transpose_63 = None
        x_379 = torch.conv2d(
            x_378,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_378 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_ = (None)
        x_380 = torch.nn.functional.dropout(x_379, 0.0, False, False)
        x_379 = None
        x_381 = x_374 + x_380
        x_374 = x_380 = None
        x_382 = x_381.permute(0, 2, 3, 1)
        x_383 = torch.nn.functional.layer_norm(
            x_382,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_382 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_bias_ = (None)
        x_384 = x_383.permute(0, 3, 1, 2)
        x_383 = None
        x_385 = torch.conv2d(
            x_384,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_384 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_386 = torch._C._nn.gelu(x_385)
        x_385 = None
        x_387 = torch.nn.functional.dropout(x_386, 0.0, False, False)
        x_386 = None
        x_388 = torch.conv2d(
            x_387,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_387 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_389 = x_381 + x_388
        x_381 = x_388 = None
        x_390 = x_389.permute(0, 2, 3, 1)
        x_391 = torch.nn.functional.layer_norm(
            x_390,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_390 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_bias_ = (None)
        x_392 = x_391.permute(0, 3, 1, 2)
        x_391 = None
        conv2d_138 = torch.conv2d(
            x_392,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_392 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_32 = conv2d_138.view(1, 24, 96, -1)
        conv2d_138 = None
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
        view_33 = relative_position_bias_32.view((196, 196, 24))
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
        x_393 = transpose_67.reshape(1, -1, 14, 14)
        transpose_67 = None
        x_394 = torch.conv2d(
            x_393,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_393 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_ = (None)
        x_395 = torch.nn.functional.dropout(x_394, 0.0, False, False)
        x_394 = None
        x_396 = x_389 + x_395
        x_389 = x_395 = None
        x_397 = x_396.permute(0, 2, 3, 1)
        x_398 = torch.nn.functional.layer_norm(
            x_397,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_397 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_bias_ = (None)
        x_399 = x_398.permute(0, 3, 1, 2)
        x_398 = None
        x_400 = torch.conv2d(
            x_399,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_399 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_401 = torch._C._nn.gelu(x_400)
        x_400 = None
        x_402 = torch.nn.functional.dropout(x_401, 0.0, False, False)
        x_401 = None
        x_403 = torch.conv2d(
            x_402,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_402 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_404 = x_396 + x_403
        x_396 = x_403 = None
        x_405 = x_404.permute(0, 2, 3, 1)
        x_406 = torch.nn.functional.layer_norm(
            x_405,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_405 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_bias_ = (None)
        x_407 = x_406.permute(0, 3, 1, 2)
        x_406 = None
        conv2d_142 = torch.conv2d(
            x_407,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_407 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_34 = conv2d_142.view(1, 24, 96, -1)
        conv2d_142 = None
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
        view_35 = relative_position_bias_34.view((196, 196, 24))
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
        x_408 = transpose_71.reshape(1, -1, 14, 14)
        transpose_71 = None
        x_409 = torch.conv2d(
            x_408,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_408 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_ = (None)
        x_410 = torch.nn.functional.dropout(x_409, 0.0, False, False)
        x_409 = None
        x_411 = x_404 + x_410
        x_404 = x_410 = None
        x_412 = x_411.permute(0, 2, 3, 1)
        x_413 = torch.nn.functional.layer_norm(
            x_412,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_412 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_bias_ = (None)
        x_414 = x_413.permute(0, 3, 1, 2)
        x_413 = None
        x_415 = torch.conv2d(
            x_414,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_414 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_416 = torch._C._nn.gelu(x_415)
        x_415 = None
        x_417 = torch.nn.functional.dropout(x_416, 0.0, False, False)
        x_416 = None
        x_418 = torch.conv2d(
            x_417,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_417 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_419 = x_411 + x_418
        x_411 = x_418 = None
        x_420 = x_419.permute(0, 2, 3, 1)
        x_421 = torch.nn.functional.layer_norm(
            x_420,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_420 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm1_parameters_bias_ = (None)
        x_422 = x_421.permute(0, 3, 1, 2)
        x_421 = None
        conv2d_146 = torch.conv2d(
            x_422,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_422 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_36 = conv2d_146.view(1, 24, 96, -1)
        conv2d_146 = None
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
        view_37 = relative_position_bias_36.view((196, 196, 24))
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
        x_423 = transpose_75.reshape(1, -1, 14, 14)
        transpose_75 = None
        x_424 = torch.conv2d(
            x_423,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_423 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_ = (None)
        x_425 = torch.nn.functional.dropout(x_424, 0.0, False, False)
        x_424 = None
        x_426 = x_419 + x_425
        x_419 = x_425 = None
        x_427 = x_426.permute(0, 2, 3, 1)
        x_428 = torch.nn.functional.layer_norm(
            x_427,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_427 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm2_parameters_bias_ = (None)
        x_429 = x_428.permute(0, 3, 1, 2)
        x_428 = None
        x_430 = torch.conv2d(
            x_429,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_429 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_431 = torch._C._nn.gelu(x_430)
        x_430 = None
        x_432 = torch.nn.functional.dropout(x_431, 0.0, False, False)
        x_431 = None
        x_433 = torch.conv2d(
            x_432,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_432 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_434 = x_426 + x_433
        x_426 = x_433 = None
        x_435 = x_434.permute(0, 2, 3, 1)
        x_436 = torch.nn.functional.layer_norm(
            x_435,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_435 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm1_parameters_bias_ = (None)
        x_437 = x_436.permute(0, 3, 1, 2)
        x_436 = None
        conv2d_150 = torch.conv2d(
            x_437,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_437 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_38 = conv2d_150.view(1, 24, 96, -1)
        conv2d_150 = None
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
        view_39 = relative_position_bias_38.view((196, 196, 24))
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
        x_438 = transpose_79.reshape(1, -1, 14, 14)
        transpose_79 = None
        x_439 = torch.conv2d(
            x_438,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_438 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_ = (None)
        x_440 = torch.nn.functional.dropout(x_439, 0.0, False, False)
        x_439 = None
        x_441 = x_434 + x_440
        x_434 = x_440 = None
        x_442 = x_441.permute(0, 2, 3, 1)
        x_443 = torch.nn.functional.layer_norm(
            x_442,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_442 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm2_parameters_bias_ = (None)
        x_444 = x_443.permute(0, 3, 1, 2)
        x_443 = None
        x_445 = torch.conv2d(
            x_444,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_444 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_446 = torch._C._nn.gelu(x_445)
        x_445 = None
        x_447 = torch.nn.functional.dropout(x_446, 0.0, False, False)
        x_446 = None
        x_448 = torch.conv2d(
            x_447,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_447 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_449 = x_441 + x_448
        x_441 = x_448 = None
        x_450 = x_449.permute(0, 2, 3, 1)
        x_451 = torch.nn.functional.layer_norm(
            x_450,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_450 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm1_parameters_bias_ = (None)
        x_452 = x_451.permute(0, 3, 1, 2)
        x_451 = None
        conv2d_154 = torch.conv2d(
            x_452,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_452 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_40 = conv2d_154.view(1, 24, 96, -1)
        conv2d_154 = None
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
        view_41 = relative_position_bias_40.view((196, 196, 24))
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
        x_453 = transpose_83.reshape(1, -1, 14, 14)
        transpose_83 = None
        x_454 = torch.conv2d(
            x_453,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_453 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_ = (None)
        x_455 = torch.nn.functional.dropout(x_454, 0.0, False, False)
        x_454 = None
        x_456 = x_449 + x_455
        x_449 = x_455 = None
        x_457 = x_456.permute(0, 2, 3, 1)
        x_458 = torch.nn.functional.layer_norm(
            x_457,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_457 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm2_parameters_bias_ = (None)
        x_459 = x_458.permute(0, 3, 1, 2)
        x_458 = None
        x_460 = torch.conv2d(
            x_459,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_459 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_461 = torch._C._nn.gelu(x_460)
        x_460 = None
        x_462 = torch.nn.functional.dropout(x_461, 0.0, False, False)
        x_461 = None
        x_463 = torch.conv2d(
            x_462,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_462 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_464 = x_456 + x_463
        x_456 = x_463 = None
        x_465 = x_464.permute(0, 2, 3, 1)
        x_466 = torch.nn.functional.layer_norm(
            x_465,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_465 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm1_parameters_bias_ = (None)
        x_467 = x_466.permute(0, 3, 1, 2)
        x_466 = None
        conv2d_158 = torch.conv2d(
            x_467,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_467 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_42 = conv2d_158.view(1, 24, 96, -1)
        conv2d_158 = None
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
        view_43 = relative_position_bias_42.view((196, 196, 24))
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
        x_468 = transpose_87.reshape(1, -1, 14, 14)
        transpose_87 = None
        x_469 = torch.conv2d(
            x_468,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_468 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_ = (None)
        x_470 = torch.nn.functional.dropout(x_469, 0.0, False, False)
        x_469 = None
        x_471 = x_464 + x_470
        x_464 = x_470 = None
        x_472 = x_471.permute(0, 2, 3, 1)
        x_473 = torch.nn.functional.layer_norm(
            x_472,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_472 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm2_parameters_bias_ = (None)
        x_474 = x_473.permute(0, 3, 1, 2)
        x_473 = None
        x_475 = torch.conv2d(
            x_474,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_474 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_476 = torch._C._nn.gelu(x_475)
        x_475 = None
        x_477 = torch.nn.functional.dropout(x_476, 0.0, False, False)
        x_476 = None
        x_478 = torch.conv2d(
            x_477,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_477 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_479 = x_471 + x_478
        x_471 = x_478 = None
        x_480 = x_479.permute(0, 2, 3, 1)
        x_481 = torch.nn.functional.layer_norm(
            x_480,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_480 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm1_parameters_bias_ = (None)
        x_482 = x_481.permute(0, 3, 1, 2)
        x_481 = None
        conv2d_162 = torch.conv2d(
            x_482,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_482 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_44 = conv2d_162.view(1, 24, 96, -1)
        conv2d_162 = None
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
        view_45 = relative_position_bias_44.view((196, 196, 24))
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
        x_483 = transpose_91.reshape(1, -1, 14, 14)
        transpose_91 = None
        x_484 = torch.conv2d(
            x_483,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_483 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_ = (None)
        x_485 = torch.nn.functional.dropout(x_484, 0.0, False, False)
        x_484 = None
        x_486 = x_479 + x_485
        x_479 = x_485 = None
        x_487 = x_486.permute(0, 2, 3, 1)
        x_488 = torch.nn.functional.layer_norm(
            x_487,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_487 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm2_parameters_bias_ = (None)
        x_489 = x_488.permute(0, 3, 1, 2)
        x_488 = None
        x_490 = torch.conv2d(
            x_489,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_489 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_491 = torch._C._nn.gelu(x_490)
        x_490 = None
        x_492 = torch.nn.functional.dropout(x_491, 0.0, False, False)
        x_491 = None
        x_493 = torch.conv2d(
            x_492,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_492 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_494 = x_486 + x_493
        x_486 = x_493 = None
        x_495 = x_494.permute(0, 2, 3, 1)
        x_496 = torch.nn.functional.layer_norm(
            x_495,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_495 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm1_parameters_bias_ = (None)
        x_497 = x_496.permute(0, 3, 1, 2)
        x_496 = None
        conv2d_166 = torch.conv2d(
            x_497,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_497 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_46 = conv2d_166.view(1, 24, 96, -1)
        conv2d_166 = None
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
        view_47 = relative_position_bias_46.view((196, 196, 24))
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
        x_498 = transpose_95.reshape(1, -1, 14, 14)
        transpose_95 = None
        x_499 = torch.conv2d(
            x_498,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_498 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_ = (None)
        x_500 = torch.nn.functional.dropout(x_499, 0.0, False, False)
        x_499 = None
        x_501 = x_494 + x_500
        x_494 = x_500 = None
        x_502 = x_501.permute(0, 2, 3, 1)
        x_503 = torch.nn.functional.layer_norm(
            x_502,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_502 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm2_parameters_bias_ = (None)
        x_504 = x_503.permute(0, 3, 1, 2)
        x_503 = None
        x_505 = torch.conv2d(
            x_504,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_504 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_506 = torch._C._nn.gelu(x_505)
        x_505 = None
        x_507 = torch.nn.functional.dropout(x_506, 0.0, False, False)
        x_506 = None
        x_508 = torch.conv2d(
            x_507,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_507 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_509 = x_501 + x_508
        x_501 = x_508 = None
        x_510 = x_509.permute(0, 2, 3, 1)
        x_511 = torch.nn.functional.layer_norm(
            x_510,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_510 = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm1_parameters_bias_ = (None)
        x_512 = x_511.permute(0, 3, 1, 2)
        x_511 = None
        conv2d_170 = torch.conv2d(
            x_512,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_512 = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_48 = conv2d_170.view(1, 24, 96, -1)
        conv2d_170 = None
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
        view_49 = relative_position_bias_48.view((196, 196, 24))
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
        x_513 = transpose_99.reshape(1, -1, 14, 14)
        transpose_99 = None
        x_514 = torch.conv2d(
            x_513,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_513 = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_attn_modules_proj_parameters_bias_ = (None)
        x_515 = torch.nn.functional.dropout(x_514, 0.0, False, False)
        x_514 = None
        x_516 = x_509 + x_515
        x_509 = x_515 = None
        x_517 = x_516.permute(0, 2, 3, 1)
        x_518 = torch.nn.functional.layer_norm(
            x_517,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_517 = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm2_parameters_bias_ = (None)
        x_519 = x_518.permute(0, 3, 1, 2)
        x_518 = None
        x_520 = torch.conv2d(
            x_519,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_519 = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_521 = torch._C._nn.gelu(x_520)
        x_520 = None
        x_522 = torch.nn.functional.dropout(x_521, 0.0, False, False)
        x_521 = None
        x_523 = torch.conv2d(
            x_522,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_522 = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_524 = x_516 + x_523
        x_516 = x_523 = None
        x_525 = x_524.permute(0, 2, 3, 1)
        x_526 = torch.nn.functional.layer_norm(
            x_525,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_525 = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm1_parameters_bias_ = (None)
        x_527 = x_526.permute(0, 3, 1, 2)
        x_526 = None
        conv2d_174 = torch.conv2d(
            x_527,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_527 = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_50 = conv2d_174.view(1, 24, 96, -1)
        conv2d_174 = None
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
        view_51 = relative_position_bias_50.view((196, 196, 24))
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
        x_528 = transpose_103.reshape(1, -1, 14, 14)
        transpose_103 = None
        x_529 = torch.conv2d(
            x_528,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_528 = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_attn_modules_proj_parameters_bias_ = (None)
        x_530 = torch.nn.functional.dropout(x_529, 0.0, False, False)
        x_529 = None
        x_531 = x_524 + x_530
        x_524 = x_530 = None
        x_532 = x_531.permute(0, 2, 3, 1)
        x_533 = torch.nn.functional.layer_norm(
            x_532,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_532 = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm2_parameters_bias_ = (None)
        x_534 = x_533.permute(0, 3, 1, 2)
        x_533 = None
        x_535 = torch.conv2d(
            x_534,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_534 = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_536 = torch._C._nn.gelu(x_535)
        x_535 = None
        x_537 = torch.nn.functional.dropout(x_536, 0.0, False, False)
        x_536 = None
        x_538 = torch.conv2d(
            x_537,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_537 = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_539 = x_531 + x_538
        x_531 = x_538 = None
        x_540 = x_539.permute(0, 2, 3, 1)
        x_541 = torch.nn.functional.layer_norm(
            x_540,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_540 = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm1_parameters_bias_ = (None)
        x_542 = x_541.permute(0, 3, 1, 2)
        x_541 = None
        conv2d_178 = torch.conv2d(
            x_542,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_542 = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_52 = conv2d_178.view(1, 24, 96, -1)
        conv2d_178 = None
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
        view_53 = relative_position_bias_52.view((196, 196, 24))
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
        x_543 = transpose_107.reshape(1, -1, 14, 14)
        transpose_107 = None
        x_544 = torch.conv2d(
            x_543,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_543 = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_attn_modules_proj_parameters_bias_ = (None)
        x_545 = torch.nn.functional.dropout(x_544, 0.0, False, False)
        x_544 = None
        x_546 = x_539 + x_545
        x_539 = x_545 = None
        x_547 = x_546.permute(0, 2, 3, 1)
        x_548 = torch.nn.functional.layer_norm(
            x_547,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_547 = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm2_parameters_bias_ = (None)
        x_549 = x_548.permute(0, 3, 1, 2)
        x_548 = None
        x_550 = torch.conv2d(
            x_549,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_549 = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_551 = torch._C._nn.gelu(x_550)
        x_550 = None
        x_552 = torch.nn.functional.dropout(x_551, 0.0, False, False)
        x_551 = None
        x_553 = torch.conv2d(
            x_552,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_552 = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_554 = x_546 + x_553
        x_546 = x_553 = None
        x_555 = x_554.permute(0, 2, 3, 1)
        x_556 = torch.nn.functional.layer_norm(
            x_555,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_555 = l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_norm1_parameters_bias_ = (None)
        x_557 = x_556.permute(0, 3, 1, 2)
        x_556 = None
        conv2d_182 = torch.conv2d(
            x_557,
            l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_557 = l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_54 = conv2d_182.view(1, 24, 96, -1)
        conv2d_182 = None
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
        view_55 = relative_position_bias_54.view((196, 196, 24))
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
        x_558 = transpose_111.reshape(1, -1, 14, 14)
        transpose_111 = None
        x_559 = torch.conv2d(
            x_558,
            l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_558 = l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_attn_modules_proj_parameters_bias_ = (None)
        x_560 = torch.nn.functional.dropout(x_559, 0.0, False, False)
        x_559 = None
        x_561 = x_554 + x_560
        x_554 = x_560 = None
        x_562 = x_561.permute(0, 2, 3, 1)
        x_563 = torch.nn.functional.layer_norm(
            x_562,
            (768,),
            l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_562 = l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_norm2_parameters_bias_ = (None)
        x_564 = x_563.permute(0, 3, 1, 2)
        x_563 = None
        x_565 = torch.conv2d(
            x_564,
            l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_564 = l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_566 = torch._C._nn.gelu(x_565)
        x_565 = None
        x_567 = torch.nn.functional.dropout(x_566, 0.0, False, False)
        x_566 = None
        x_568 = torch.conv2d(
            x_567,
            l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_567 = l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_569 = x_561 + x_568
        x_561 = x_568 = None
        x_570 = torch._C._nn.avg_pool2d(x_569, 2, 2, 0, False, True, None)
        x_571 = torch.conv2d(
            x_570,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_570 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_ = (None)
        x_572 = x_569.permute(0, 2, 3, 1)
        x_569 = None
        x_573 = torch.nn.functional.layer_norm(
            x_572,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_,
            1e-06,
        )
        x_572 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_ = (None)
        x_574 = x_573.permute(0, 3, 1, 2)
        x_573 = None
        x_575 = torch._C._nn.avg_pool2d(x_574, 2, 2, 0, False, True, None)
        x_574 = None
        conv2d_187 = torch.conv2d(
            x_575,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_575 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_56 = conv2d_187.view(1, 48, 96, -1)
        conv2d_187 = None
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
        view_57 = relative_position_bias_56.view((49, 49, 48))
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
        x_576 = transpose_115.reshape(1, -1, 7, 7)
        transpose_115 = None
        x_577 = torch.conv2d(
            x_576,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_576 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_578 = torch.nn.functional.dropout(x_577, 0.0, False, False)
        x_577 = None
        x_579 = x_571 + x_578
        x_571 = x_578 = None
        x_580 = x_579.permute(0, 2, 3, 1)
        x_581 = torch.nn.functional.layer_norm(
            x_580,
            (1536,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_580 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_582 = x_581.permute(0, 3, 1, 2)
        x_581 = None
        x_583 = torch.conv2d(
            x_582,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_582 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_584 = torch._C._nn.gelu(x_583)
        x_583 = None
        x_585 = torch.nn.functional.dropout(x_584, 0.0, False, False)
        x_584 = None
        x_586 = torch.conv2d(
            x_585,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_585 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_587 = x_579 + x_586
        x_579 = x_586 = None
        x_588 = x_587.permute(0, 2, 3, 1)
        x_589 = torch.nn.functional.layer_norm(
            x_588,
            (1536,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_588 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_590 = x_589.permute(0, 3, 1, 2)
        x_589 = None
        conv2d_191 = torch.conv2d(
            x_590,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_590 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_58 = conv2d_191.view(1, 48, 96, -1)
        conv2d_191 = None
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
        view_59 = relative_position_bias_58.view((49, 49, 48))
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
        x_591 = transpose_119.reshape(1, -1, 7, 7)
        transpose_119 = None
        x_592 = torch.conv2d(
            x_591,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_591 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_593 = torch.nn.functional.dropout(x_592, 0.0, False, False)
        x_592 = None
        x_594 = x_587 + x_593
        x_587 = x_593 = None
        x_595 = x_594.permute(0, 2, 3, 1)
        x_596 = torch.nn.functional.layer_norm(
            x_595,
            (1536,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_595 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_597 = x_596.permute(0, 3, 1, 2)
        x_596 = None
        x_598 = torch.conv2d(
            x_597,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_597 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_599 = torch._C._nn.gelu(x_598)
        x_598 = None
        x_600 = torch.nn.functional.dropout(x_599, 0.0, False, False)
        x_599 = None
        x_601 = torch.conv2d(
            x_600,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_600 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_602 = x_594 + x_601
        x_594 = x_601 = None
        x_603 = torch.nn.functional.adaptive_avg_pool2d(x_602, 1)
        x_602 = None
        x_604 = x_603.permute(0, 2, 3, 1)
        x_603 = None
        x_605 = torch.nn.functional.layer_norm(
            x_604,
            (1536,),
            l_self_modules_head_modules_norm_parameters_weight_,
            l_self_modules_head_modules_norm_parameters_bias_,
            1e-06,
        )
        x_604 = (
            l_self_modules_head_modules_norm_parameters_weight_
        ) = l_self_modules_head_modules_norm_parameters_bias_ = None
        x_606 = x_605.permute(0, 3, 1, 2)
        x_605 = None
        x_607 = x_606.flatten(1, -1)
        x_606 = None
        input_1 = torch._C._nn.linear(
            x_607,
            l_self_modules_head_modules_pre_logits_modules_fc_parameters_weight_,
            l_self_modules_head_modules_pre_logits_modules_fc_parameters_bias_,
        )
        x_607 = (
            l_self_modules_head_modules_pre_logits_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_pre_logits_modules_fc_parameters_bias_ = None
        input_2 = input_1.tanh()
        input_1 = None
        x_608 = torch.nn.functional.dropout(input_2, 0.0, False, False)
        input_2 = None
        x_609 = torch._C._nn.linear(
            x_608,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_608 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_609,)
