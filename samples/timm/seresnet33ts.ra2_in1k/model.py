import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_final_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_final_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_final_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_final_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_final_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_conv1_modules_conv_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_stem_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_stem_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_stem_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_stem_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_stem_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_final_conv_modules_conv_parameters_weight_ = (
            L_self_modules_final_conv_modules_conv_parameters_weight_
        )
        l_self_modules_final_conv_modules_bn_buffers_running_mean_ = (
            L_self_modules_final_conv_modules_bn_buffers_running_mean_
        )
        l_self_modules_final_conv_modules_bn_buffers_running_var_ = (
            L_self_modules_final_conv_modules_bn_buffers_running_var_
        )
        l_self_modules_final_conv_modules_bn_parameters_weight_ = (
            L_self_modules_final_conv_modules_bn_parameters_weight_
        )
        l_self_modules_final_conv_modules_bn_parameters_bias_ = (
            L_self_modules_final_conv_modules_bn_parameters_bias_
        )
        l_self_modules_head_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_fc_parameters_bias_
        )
        x = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_conv1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_stem_modules_conv1_modules_conv_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_stem_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_self_modules_stem_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_conv1_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_conv1_modules_bn_parameters_bias_ = None
        x_2 = torch.nn.functional.silu(x_1, inplace=True)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_stem_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_stem_modules_conv2_modules_conv_parameters_weight_ = None
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_stem_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_3 = (
            l_self_modules_stem_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_conv2_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_conv2_modules_bn_parameters_bias_ = None
        x_5 = torch.nn.functional.silu(x_4, inplace=True)
        x_4 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_stem_modules_conv3_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_5 = l_self_modules_stem_modules_conv3_modules_conv_parameters_weight_ = None
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_stem_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_6 = (
            l_self_modules_stem_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_conv3_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_conv3_modules_bn_parameters_bias_ = None
        x_8 = torch.nn.functional.silu(x_7, inplace=True)
        x_7 = None
        x_9 = torch.conv2d(
            x_8,
            l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_10 = torch.nn.functional.batch_norm(
            x_9,
            l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_9 = l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_11 = torch.nn.functional.silu(x_10, inplace=True)
        x_10 = None
        x_12 = torch.conv2d(
            x_11,
            l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_11 = l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_13 = torch.nn.functional.batch_norm(
            x_12,
            l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_12 = l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_14 = torch.nn.functional.silu(x_13, inplace=True)
        x_13 = None
        x_se = x_14.mean((2, 3), keepdim=True)
        x_se_1 = torch.conv2d(
            x_se,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_2 = torch.nn.functional.relu(x_se_1, inplace=True)
        x_se_1 = None
        x_se_3 = torch.conv2d(
            x_se_2,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_2 = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid = x_se_3.sigmoid()
        x_se_3 = None
        x_15 = x_14 * sigmoid
        x_14 = sigmoid = None
        x_16 = torch.conv2d(
            x_15,
            l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_15 = l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_16 = l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_18 = torch.conv2d(
            x_8,
            l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_8 = l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_parameters_weight_ = (None)
        x_19 = torch.nn.functional.batch_norm(
            x_18,
            l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_18 = l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_bias_ = (None)
        x_20 = x_17 + x_19
        x_17 = x_19 = None
        input_1 = torch.nn.functional.silu(x_20, inplace=True)
        x_20 = None
        x_21 = torch.conv2d(
            input_1,
            l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_22 = torch.nn.functional.batch_norm(
            x_21,
            l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_21 = l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_23 = torch.nn.functional.silu(x_22, inplace=True)
        x_22 = None
        x_24 = torch.conv2d(
            x_23,
            l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_23 = l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_25 = torch.nn.functional.batch_norm(
            x_24,
            l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_24 = l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_26 = torch.nn.functional.silu(x_25, inplace=True)
        x_25 = None
        x_se_4 = x_26.mean((2, 3), keepdim=True)
        x_se_5 = torch.conv2d(
            x_se_4,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_4 = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_6 = torch.nn.functional.relu(x_se_5, inplace=True)
        x_se_5 = None
        x_se_7 = torch.conv2d(
            x_se_6,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_6 = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_1 = x_se_7.sigmoid()
        x_se_7 = None
        x_27 = x_26 * sigmoid_1
        x_26 = sigmoid_1 = None
        x_28 = torch.conv2d(
            x_27,
            l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_27 = l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_29 = torch.nn.functional.batch_norm(
            x_28,
            l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_28 = l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_30 = x_29 + input_1
        x_29 = input_1 = None
        input_2 = torch.nn.functional.silu(x_30, inplace=True)
        x_30 = None
        x_31 = torch.conv2d(
            input_2,
            l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_32 = torch.nn.functional.batch_norm(
            x_31,
            l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_31 = l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_33 = torch.nn.functional.silu(x_32, inplace=True)
        x_32 = None
        x_34 = torch.conv2d(
            x_33,
            l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_33 = l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_35 = torch.nn.functional.batch_norm(
            x_34,
            l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_34 = l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_36 = torch.nn.functional.silu(x_35, inplace=True)
        x_35 = None
        x_se_8 = x_36.mean((2, 3), keepdim=True)
        x_se_9 = torch.conv2d(
            x_se_8,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_8 = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_10 = torch.nn.functional.relu(x_se_9, inplace=True)
        x_se_9 = None
        x_se_11 = torch.conv2d(
            x_se_10,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_10 = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_2 = x_se_11.sigmoid()
        x_se_11 = None
        x_37 = x_36 * sigmoid_2
        x_36 = sigmoid_2 = None
        x_38 = torch.conv2d(
            x_37,
            l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_37 = l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_39 = torch.nn.functional.batch_norm(
            x_38,
            l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_38 = l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_40 = torch.conv2d(
            input_2,
            l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        input_2 = l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_parameters_weight_ = (None)
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_bias_ = (None)
        x_42 = x_39 + x_41
        x_39 = x_41 = None
        input_3 = torch.nn.functional.silu(x_42, inplace=True)
        x_42 = None
        x_43 = torch.conv2d(
            input_3,
            l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_44 = torch.nn.functional.batch_norm(
            x_43,
            l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_43 = l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_45 = torch.nn.functional.silu(x_44, inplace=True)
        x_44 = None
        x_46 = torch.conv2d(
            x_45,
            l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_45 = l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_47 = torch.nn.functional.batch_norm(
            x_46,
            l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_46 = l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_48 = torch.nn.functional.silu(x_47, inplace=True)
        x_47 = None
        x_se_12 = x_48.mean((2, 3), keepdim=True)
        x_se_13 = torch.conv2d(
            x_se_12,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_12 = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_14 = torch.nn.functional.relu(x_se_13, inplace=True)
        x_se_13 = None
        x_se_15 = torch.conv2d(
            x_se_14,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_14 = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_3 = x_se_15.sigmoid()
        x_se_15 = None
        x_49 = x_48 * sigmoid_3
        x_48 = sigmoid_3 = None
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_49 = l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_50 = l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_52 = x_51 + input_3
        x_51 = input_3 = None
        input_4 = torch.nn.functional.silu(x_52, inplace=True)
        x_52 = None
        x_53 = torch.conv2d(
            input_4,
            l_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_54 = torch.nn.functional.batch_norm(
            x_53,
            l_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_53 = l_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_55 = torch.nn.functional.silu(x_54, inplace=True)
        x_54 = None
        x_56 = torch.conv2d(
            x_55,
            l_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_55 = l_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_57 = torch.nn.functional.batch_norm(
            x_56,
            l_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_56 = l_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_58 = torch.nn.functional.silu(x_57, inplace=True)
        x_57 = None
        x_se_16 = x_58.mean((2, 3), keepdim=True)
        x_se_17 = torch.conv2d(
            x_se_16,
            l_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_16 = l_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_18 = torch.nn.functional.relu(x_se_17, inplace=True)
        x_se_17 = None
        x_se_19 = torch.conv2d(
            x_se_18,
            l_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_18 = l_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_4 = x_se_19.sigmoid()
        x_se_19 = None
        x_59 = x_58 * sigmoid_4
        x_58 = sigmoid_4 = None
        x_60 = torch.conv2d(
            x_59,
            l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_59 = l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_61 = torch.nn.functional.batch_norm(
            x_60,
            l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_60 = l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_62 = x_61 + input_4
        x_61 = input_4 = None
        input_5 = torch.nn.functional.silu(x_62, inplace=True)
        x_62 = None
        x_63 = torch.conv2d(
            input_5,
            l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_64 = torch.nn.functional.batch_norm(
            x_63,
            l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_63 = l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_65 = torch.nn.functional.silu(x_64, inplace=True)
        x_64 = None
        x_66 = torch.conv2d(
            x_65,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_65 = l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_67 = torch.nn.functional.batch_norm(
            x_66,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_66 = l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_68 = torch.nn.functional.silu(x_67, inplace=True)
        x_67 = None
        x_se_20 = x_68.mean((2, 3), keepdim=True)
        x_se_21 = torch.conv2d(
            x_se_20,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_20 = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_22 = torch.nn.functional.relu(x_se_21, inplace=True)
        x_se_21 = None
        x_se_23 = torch.conv2d(
            x_se_22,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_22 = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_5 = x_se_23.sigmoid()
        x_se_23 = None
        x_69 = x_68 * sigmoid_5
        x_68 = sigmoid_5 = None
        x_70 = torch.conv2d(
            x_69,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_69 = l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_71 = torch.nn.functional.batch_norm(
            x_70,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_70 = l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_72 = torch.conv2d(
            input_5,
            l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        input_5 = l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_parameters_weight_ = (None)
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_72 = l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_bias_ = (None)
        x_74 = x_71 + x_73
        x_71 = x_73 = None
        input_6 = torch.nn.functional.silu(x_74, inplace=True)
        x_74 = None
        x_75 = torch.conv2d(
            input_6,
            l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_76 = torch.nn.functional.batch_norm(
            x_75,
            l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_75 = l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_77 = torch.nn.functional.silu(x_76, inplace=True)
        x_76 = None
        x_78 = torch.conv2d(
            x_77,
            l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_77 = l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_79 = torch.nn.functional.batch_norm(
            x_78,
            l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_78 = l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_80 = torch.nn.functional.silu(x_79, inplace=True)
        x_79 = None
        x_se_24 = x_80.mean((2, 3), keepdim=True)
        x_se_25 = torch.conv2d(
            x_se_24,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_24 = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_26 = torch.nn.functional.relu(x_se_25, inplace=True)
        x_se_25 = None
        x_se_27 = torch.conv2d(
            x_se_26,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_26 = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_6 = x_se_27.sigmoid()
        x_se_27 = None
        x_81 = x_80 * sigmoid_6
        x_80 = sigmoid_6 = None
        x_82 = torch.conv2d(
            x_81,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_81 = l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_83 = torch.nn.functional.batch_norm(
            x_82,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_82 = l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_84 = x_83 + input_6
        x_83 = input_6 = None
        input_7 = torch.nn.functional.silu(x_84, inplace=True)
        x_84 = None
        x_85 = torch.conv2d(
            input_7,
            l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_86 = torch.nn.functional.batch_norm(
            x_85,
            l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_85 = l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_87 = torch.nn.functional.silu(x_86, inplace=True)
        x_86 = None
        x_88 = torch.conv2d(
            x_87,
            l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_87 = l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_89 = torch.nn.functional.batch_norm(
            x_88,
            l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_88 = l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_90 = torch.nn.functional.silu(x_89, inplace=True)
        x_89 = None
        x_se_28 = x_90.mean((2, 3), keepdim=True)
        x_se_29 = torch.conv2d(
            x_se_28,
            l_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_28 = l_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_30 = torch.nn.functional.relu(x_se_29, inplace=True)
        x_se_29 = None
        x_se_31 = torch.conv2d(
            x_se_30,
            l_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_30 = l_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_7 = x_se_31.sigmoid()
        x_se_31 = None
        x_91 = x_90 * sigmoid_7
        x_90 = sigmoid_7 = None
        x_92 = torch.conv2d(
            x_91,
            l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_91 = l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_93 = torch.nn.functional.batch_norm(
            x_92,
            l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_92 = l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_94 = x_93 + input_7
        x_93 = input_7 = None
        input_8 = torch.nn.functional.silu(x_94, inplace=True)
        x_94 = None
        x_95 = torch.conv2d(
            input_8,
            l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_96 = torch.nn.functional.batch_norm(
            x_95,
            l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_95 = l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_97 = torch.nn.functional.silu(x_96, inplace=True)
        x_96 = None
        x_98 = torch.conv2d(
            x_97,
            l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_97 = l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_99 = torch.nn.functional.batch_norm(
            x_98,
            l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_98 = l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_100 = torch.nn.functional.silu(x_99, inplace=True)
        x_99 = None
        x_se_32 = x_100.mean((2, 3), keepdim=True)
        x_se_33 = torch.conv2d(
            x_se_32,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_32 = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_34 = torch.nn.functional.relu(x_se_33, inplace=True)
        x_se_33 = None
        x_se_35 = torch.conv2d(
            x_se_34,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_34 = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_8 = x_se_35.sigmoid()
        x_se_35 = None
        x_101 = x_100 * sigmoid_8
        x_100 = sigmoid_8 = None
        x_102 = torch.conv2d(
            x_101,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_101 = l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_103 = torch.nn.functional.batch_norm(
            x_102,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_102 = l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_104 = torch.conv2d(
            input_8,
            l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        input_8 = l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_parameters_weight_ = (None)
        x_105 = torch.nn.functional.batch_norm(
            x_104,
            l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_104 = l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_bias_ = (None)
        x_106 = x_103 + x_105
        x_103 = x_105 = None
        input_9 = torch.nn.functional.silu(x_106, inplace=True)
        x_106 = None
        x_107 = torch.conv2d(
            input_9,
            l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_108 = torch.nn.functional.batch_norm(
            x_107,
            l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_107 = l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_109 = torch.nn.functional.silu(x_108, inplace=True)
        x_108 = None
        x_110 = torch.conv2d(
            x_109,
            l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_109 = l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_110 = l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_112 = torch.nn.functional.silu(x_111, inplace=True)
        x_111 = None
        x_se_36 = x_112.mean((2, 3), keepdim=True)
        x_se_37 = torch.conv2d(
            x_se_36,
            l_self_modules_stages_modules_3_modules_1_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_36 = l_self_modules_stages_modules_3_modules_1_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_38 = torch.nn.functional.relu(x_se_37, inplace=True)
        x_se_37 = None
        x_se_39 = torch.conv2d(
            x_se_38,
            l_self_modules_stages_modules_3_modules_1_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_38 = l_self_modules_stages_modules_3_modules_1_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_9 = x_se_39.sigmoid()
        x_se_39 = None
        x_113 = x_112 * sigmoid_9
        x_112 = sigmoid_9 = None
        x_114 = torch.conv2d(
            x_113,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_113 = l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_115 = torch.nn.functional.batch_norm(
            x_114,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_114 = l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_116 = x_115 + input_9
        x_115 = input_9 = None
        input_10 = torch.nn.functional.silu(x_116, inplace=True)
        x_116 = None
        x_117 = torch.conv2d(
            input_10,
            l_self_modules_final_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_10 = l_self_modules_final_conv_modules_conv_parameters_weight_ = None
        x_118 = torch.nn.functional.batch_norm(
            x_117,
            l_self_modules_final_conv_modules_bn_buffers_running_mean_,
            l_self_modules_final_conv_modules_bn_buffers_running_var_,
            l_self_modules_final_conv_modules_bn_parameters_weight_,
            l_self_modules_final_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_117 = (
            l_self_modules_final_conv_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_final_conv_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_final_conv_modules_bn_parameters_weight_
        ) = l_self_modules_final_conv_modules_bn_parameters_bias_ = None
        x_119 = torch.nn.functional.silu(x_118, inplace=True)
        x_118 = None
        x_120 = torch.nn.functional.adaptive_avg_pool2d(x_119, 1)
        x_119 = None
        x_121 = x_120.flatten(1, -1)
        x_120 = None
        x_122 = torch.nn.functional.dropout(x_121, 0.0, False, False)
        x_121 = None
        x_123 = torch._C._nn.linear(
            x_122,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_122 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_123,)
