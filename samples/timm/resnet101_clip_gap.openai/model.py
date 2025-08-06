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
        L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_2_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_2_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_2_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_2_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_2_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_2_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_2_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_2_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_2_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_2_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_2_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_2_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_3_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_3_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_3_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_3_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_3_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_3_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_3_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_3_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_3_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_3_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_3_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_3_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_3_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_3_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_3_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_3_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_3_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_3_modules_conv3_1x1_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_
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
        x_2 = torch.nn.functional.relu(x_1, inplace=True)
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
        x_5 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_stem_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
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
        x_8 = torch.nn.functional.relu(x_7, inplace=True)
        x_7 = None
        input_1 = torch._C._nn.avg_pool2d(x_8, 2, 2, 0, False, True, None)
        x_8 = None
        x_9 = torch.conv2d(
            input_1,
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
        x_11 = torch.nn.functional.relu(x_10, inplace=True)
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
        x_14 = torch.nn.functional.relu(x_13, inplace=True)
        x_13 = None
        x_15 = torch.conv2d(
            x_14,
            l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_14 = l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_16 = torch.nn.functional.batch_norm(
            x_15,
            l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_15 = l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_17 = torch.conv2d(
            input_1,
            l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_1 = l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_modules_conv_parameters_weight_ = (None)
        x_18 = torch.nn.functional.batch_norm(
            x_17,
            l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_17 = l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_bias_ = (None)
        x_19 = x_16 + x_18
        x_16 = x_18 = None
        input_2 = torch.nn.functional.relu(x_19, inplace=True)
        x_19 = None
        x_20 = torch.conv2d(
            input_2,
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
        x_21 = torch.nn.functional.batch_norm(
            x_20,
            l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_20 = l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_22 = torch.nn.functional.relu(x_21, inplace=True)
        x_21 = None
        x_23 = torch.conv2d(
            x_22,
            l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_22 = l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_24 = torch.nn.functional.batch_norm(
            x_23,
            l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_23 = l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_25 = torch.nn.functional.relu(x_24, inplace=True)
        x_24 = None
        x_26 = torch.conv2d(
            x_25,
            l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_25 = l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_27 = torch.nn.functional.batch_norm(
            x_26,
            l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_26 = l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_28 = x_27 + input_2
        x_27 = input_2 = None
        input_3 = torch.nn.functional.relu(x_28, inplace=True)
        x_28 = None
        x_29 = torch.conv2d(
            input_3,
            l_self_modules_stages_modules_0_modules_2_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_30 = torch.nn.functional.batch_norm(
            x_29,
            l_self_modules_stages_modules_0_modules_2_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_2_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_2_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_2_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_29 = l_self_modules_stages_modules_0_modules_2_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_2_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_2_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_2_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_31 = torch.nn.functional.relu(x_30, inplace=True)
        x_30 = None
        x_32 = torch.conv2d(
            x_31,
            l_self_modules_stages_modules_0_modules_2_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_31 = l_self_modules_stages_modules_0_modules_2_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_33 = torch.nn.functional.batch_norm(
            x_32,
            l_self_modules_stages_modules_0_modules_2_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_2_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_2_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_2_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_32 = l_self_modules_stages_modules_0_modules_2_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_2_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_2_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_2_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_34 = torch.nn.functional.relu(x_33, inplace=True)
        x_33 = None
        x_35 = torch.conv2d(
            x_34,
            l_self_modules_stages_modules_0_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_34 = l_self_modules_stages_modules_0_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_36 = torch.nn.functional.batch_norm(
            x_35,
            l_self_modules_stages_modules_0_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_35 = l_self_modules_stages_modules_0_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_37 = x_36 + input_3
        x_36 = input_3 = None
        input_4 = torch.nn.functional.relu(x_37, inplace=True)
        x_37 = None
        x_38 = torch.conv2d(
            input_4,
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
        x_39 = torch.nn.functional.batch_norm(
            x_38,
            l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_38 = l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_40 = torch.nn.functional.relu(x_39, inplace=True)
        x_39 = None
        x_41 = torch.conv2d(
            x_40,
            l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_40 = l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_42 = torch.nn.functional.batch_norm(
            x_41,
            l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_41 = l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_43 = torch.nn.functional.relu(x_42, inplace=True)
        x_42 = None
        x_44 = torch._C._nn.avg_pool2d(x_43, 2, 2, 0, False, True, None)
        x_43 = None
        x_45 = torch.conv2d(
            x_44,
            l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_44 = l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_46 = torch.nn.functional.batch_norm(
            x_45,
            l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_45 = l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        avg_pool2d_2 = torch._C._nn.avg_pool2d(input_4, 2, 2, 0, True, False, None)
        input_4 = None
        x_47 = torch.conv2d(
            avg_pool2d_2,
            l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        avg_pool2d_2 = l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_modules_conv_parameters_weight_ = (None)
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_bias_ = (None)
        x_49 = x_46 + x_48
        x_46 = x_48 = None
        input_5 = torch.nn.functional.relu(x_49, inplace=True)
        x_49 = None
        x_50 = torch.conv2d(
            input_5,
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
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_50 = l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_52 = torch.nn.functional.relu(x_51, inplace=True)
        x_51 = None
        x_53 = torch.conv2d(
            x_52,
            l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_52 = l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_54 = torch.nn.functional.batch_norm(
            x_53,
            l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_53 = l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_55 = torch.nn.functional.relu(x_54, inplace=True)
        x_54 = None
        x_56 = torch.conv2d(
            x_55,
            l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_55 = l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_57 = torch.nn.functional.batch_norm(
            x_56,
            l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_56 = l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_58 = x_57 + input_5
        x_57 = input_5 = None
        input_6 = torch.nn.functional.relu(x_58, inplace=True)
        x_58 = None
        x_59 = torch.conv2d(
            input_6,
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
        x_60 = torch.nn.functional.batch_norm(
            x_59,
            l_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_59 = l_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_61 = torch.nn.functional.relu(x_60, inplace=True)
        x_60 = None
        x_62 = torch.conv2d(
            x_61,
            l_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_61 = l_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_63 = torch.nn.functional.batch_norm(
            x_62,
            l_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_62 = l_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_64 = torch.nn.functional.relu(x_63, inplace=True)
        x_63 = None
        x_65 = torch.conv2d(
            x_64,
            l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_64 = l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_66 = torch.nn.functional.batch_norm(
            x_65,
            l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_65 = l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_67 = x_66 + input_6
        x_66 = input_6 = None
        input_7 = torch.nn.functional.relu(x_67, inplace=True)
        x_67 = None
        x_68 = torch.conv2d(
            input_7,
            l_self_modules_stages_modules_1_modules_3_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_69 = torch.nn.functional.batch_norm(
            x_68,
            l_self_modules_stages_modules_1_modules_3_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_3_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_3_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_3_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_68 = l_self_modules_stages_modules_1_modules_3_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_3_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_3_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_3_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_70 = torch.nn.functional.relu(x_69, inplace=True)
        x_69 = None
        x_71 = torch.conv2d(
            x_70,
            l_self_modules_stages_modules_1_modules_3_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_70 = l_self_modules_stages_modules_1_modules_3_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_72 = torch.nn.functional.batch_norm(
            x_71,
            l_self_modules_stages_modules_1_modules_3_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_3_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_3_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_3_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_71 = l_self_modules_stages_modules_1_modules_3_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_3_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_3_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_3_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_73 = torch.nn.functional.relu(x_72, inplace=True)
        x_72 = None
        x_74 = torch.conv2d(
            x_73,
            l_self_modules_stages_modules_1_modules_3_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_73 = l_self_modules_stages_modules_1_modules_3_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_75 = torch.nn.functional.batch_norm(
            x_74,
            l_self_modules_stages_modules_1_modules_3_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_3_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_3_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_3_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_74 = l_self_modules_stages_modules_1_modules_3_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_3_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_3_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_3_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_76 = x_75 + input_7
        x_75 = input_7 = None
        input_8 = torch.nn.functional.relu(x_76, inplace=True)
        x_76 = None
        x_77 = torch.conv2d(
            input_8,
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
        x_78 = torch.nn.functional.batch_norm(
            x_77,
            l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_77 = l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_79 = torch.nn.functional.relu(x_78, inplace=True)
        x_78 = None
        x_80 = torch.conv2d(
            x_79,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_79 = l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_81 = torch.nn.functional.batch_norm(
            x_80,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_80 = l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_82 = torch.nn.functional.relu(x_81, inplace=True)
        x_81 = None
        x_83 = torch._C._nn.avg_pool2d(x_82, 2, 2, 0, False, True, None)
        x_82 = None
        x_84 = torch.conv2d(
            x_83,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_83 = l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_85 = torch.nn.functional.batch_norm(
            x_84,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_84 = l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        avg_pool2d_4 = torch._C._nn.avg_pool2d(input_8, 2, 2, 0, True, False, None)
        input_8 = None
        x_86 = torch.conv2d(
            avg_pool2d_4,
            l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        avg_pool2d_4 = l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_conv_parameters_weight_ = (None)
        x_87 = torch.nn.functional.batch_norm(
            x_86,
            l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_86 = l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_bias_ = (None)
        x_88 = x_85 + x_87
        x_85 = x_87 = None
        input_9 = torch.nn.functional.relu(x_88, inplace=True)
        x_88 = None
        x_89 = torch.conv2d(
            input_9,
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
        x_90 = torch.nn.functional.batch_norm(
            x_89,
            l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_89 = l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_91 = torch.nn.functional.relu(x_90, inplace=True)
        x_90 = None
        x_92 = torch.conv2d(
            x_91,
            l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_91 = l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_93 = torch.nn.functional.batch_norm(
            x_92,
            l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_92 = l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_94 = torch.nn.functional.relu(x_93, inplace=True)
        x_93 = None
        x_95 = torch.conv2d(
            x_94,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_94 = l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_96 = torch.nn.functional.batch_norm(
            x_95,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_95 = l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_97 = x_96 + input_9
        x_96 = input_9 = None
        input_10 = torch.nn.functional.relu(x_97, inplace=True)
        x_97 = None
        x_98 = torch.conv2d(
            input_10,
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
        x_99 = torch.nn.functional.batch_norm(
            x_98,
            l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_98 = l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_100 = torch.nn.functional.relu(x_99, inplace=True)
        x_99 = None
        x_101 = torch.conv2d(
            x_100,
            l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_100 = l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_102 = torch.nn.functional.batch_norm(
            x_101,
            l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_101 = l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_103 = torch.nn.functional.relu(x_102, inplace=True)
        x_102 = None
        x_104 = torch.conv2d(
            x_103,
            l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_103 = l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_105 = torch.nn.functional.batch_norm(
            x_104,
            l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_104 = l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_106 = x_105 + input_10
        x_105 = input_10 = None
        input_11 = torch.nn.functional.relu(x_106, inplace=True)
        x_106 = None
        x_107 = torch.conv2d(
            input_11,
            l_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_108 = torch.nn.functional.batch_norm(
            x_107,
            l_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_107 = l_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_109 = torch.nn.functional.relu(x_108, inplace=True)
        x_108 = None
        x_110 = torch.conv2d(
            x_109,
            l_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_109 = l_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_110 = l_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_112 = torch.nn.functional.relu(x_111, inplace=True)
        x_111 = None
        x_113 = torch.conv2d(
            x_112,
            l_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_112 = l_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_114 = torch.nn.functional.batch_norm(
            x_113,
            l_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_113 = l_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_115 = x_114 + input_11
        x_114 = input_11 = None
        input_12 = torch.nn.functional.relu(x_115, inplace=True)
        x_115 = None
        x_116 = torch.conv2d(
            input_12,
            l_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_117 = torch.nn.functional.batch_norm(
            x_116,
            l_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_116 = l_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_118 = torch.nn.functional.relu(x_117, inplace=True)
        x_117 = None
        x_119 = torch.conv2d(
            x_118,
            l_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_118 = l_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_120 = torch.nn.functional.batch_norm(
            x_119,
            l_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_119 = l_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_121 = torch.nn.functional.relu(x_120, inplace=True)
        x_120 = None
        x_122 = torch.conv2d(
            x_121,
            l_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_121 = l_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_123 = torch.nn.functional.batch_norm(
            x_122,
            l_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_122 = l_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_124 = x_123 + input_12
        x_123 = input_12 = None
        input_13 = torch.nn.functional.relu(x_124, inplace=True)
        x_124 = None
        x_125 = torch.conv2d(
            input_13,
            l_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_126 = torch.nn.functional.batch_norm(
            x_125,
            l_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_125 = l_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_127 = torch.nn.functional.relu(x_126, inplace=True)
        x_126 = None
        x_128 = torch.conv2d(
            x_127,
            l_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_127 = l_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_129 = torch.nn.functional.batch_norm(
            x_128,
            l_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_128 = l_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_130 = torch.nn.functional.relu(x_129, inplace=True)
        x_129 = None
        x_131 = torch.conv2d(
            x_130,
            l_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_130 = l_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_132 = torch.nn.functional.batch_norm(
            x_131,
            l_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_131 = l_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_133 = x_132 + input_13
        x_132 = input_13 = None
        input_14 = torch.nn.functional.relu(x_133, inplace=True)
        x_133 = None
        x_134 = torch.conv2d(
            input_14,
            l_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_135 = torch.nn.functional.batch_norm(
            x_134,
            l_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_134 = l_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_136 = torch.nn.functional.relu(x_135, inplace=True)
        x_135 = None
        x_137 = torch.conv2d(
            x_136,
            l_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_136 = l_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_138 = torch.nn.functional.batch_norm(
            x_137,
            l_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_137 = l_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_139 = torch.nn.functional.relu(x_138, inplace=True)
        x_138 = None
        x_140 = torch.conv2d(
            x_139,
            l_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_139 = l_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_141 = torch.nn.functional.batch_norm(
            x_140,
            l_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_140 = l_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_142 = x_141 + input_14
        x_141 = input_14 = None
        input_15 = torch.nn.functional.relu(x_142, inplace=True)
        x_142 = None
        x_143 = torch.conv2d(
            input_15,
            l_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_144 = torch.nn.functional.batch_norm(
            x_143,
            l_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_143 = l_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_145 = torch.nn.functional.relu(x_144, inplace=True)
        x_144 = None
        x_146 = torch.conv2d(
            x_145,
            l_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_145 = l_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_147 = torch.nn.functional.batch_norm(
            x_146,
            l_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_146 = l_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_148 = torch.nn.functional.relu(x_147, inplace=True)
        x_147 = None
        x_149 = torch.conv2d(
            x_148,
            l_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_148 = l_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_150 = torch.nn.functional.batch_norm(
            x_149,
            l_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_149 = l_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_151 = x_150 + input_15
        x_150 = input_15 = None
        input_16 = torch.nn.functional.relu(x_151, inplace=True)
        x_151 = None
        x_152 = torch.conv2d(
            input_16,
            l_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_153 = torch.nn.functional.batch_norm(
            x_152,
            l_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_152 = l_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_154 = torch.nn.functional.relu(x_153, inplace=True)
        x_153 = None
        x_155 = torch.conv2d(
            x_154,
            l_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_154 = l_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_156 = torch.nn.functional.batch_norm(
            x_155,
            l_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_155 = l_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_157 = torch.nn.functional.relu(x_156, inplace=True)
        x_156 = None
        x_158 = torch.conv2d(
            x_157,
            l_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_157 = l_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_159 = torch.nn.functional.batch_norm(
            x_158,
            l_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_158 = l_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_160 = x_159 + input_16
        x_159 = input_16 = None
        input_17 = torch.nn.functional.relu(x_160, inplace=True)
        x_160 = None
        x_161 = torch.conv2d(
            input_17,
            l_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_162 = torch.nn.functional.batch_norm(
            x_161,
            l_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_161 = l_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_163 = torch.nn.functional.relu(x_162, inplace=True)
        x_162 = None
        x_164 = torch.conv2d(
            x_163,
            l_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_163 = l_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_165 = torch.nn.functional.batch_norm(
            x_164,
            l_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_164 = l_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_166 = torch.nn.functional.relu(x_165, inplace=True)
        x_165 = None
        x_167 = torch.conv2d(
            x_166,
            l_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_166 = l_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_168 = torch.nn.functional.batch_norm(
            x_167,
            l_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_167 = l_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_169 = x_168 + input_17
        x_168 = input_17 = None
        input_18 = torch.nn.functional.relu(x_169, inplace=True)
        x_169 = None
        x_170 = torch.conv2d(
            input_18,
            l_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_171 = torch.nn.functional.batch_norm(
            x_170,
            l_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_170 = l_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_172 = torch.nn.functional.relu(x_171, inplace=True)
        x_171 = None
        x_173 = torch.conv2d(
            x_172,
            l_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_172 = l_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_174 = torch.nn.functional.batch_norm(
            x_173,
            l_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_173 = l_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_175 = torch.nn.functional.relu(x_174, inplace=True)
        x_174 = None
        x_176 = torch.conv2d(
            x_175,
            l_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_175 = l_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_177 = torch.nn.functional.batch_norm(
            x_176,
            l_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_176 = l_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_178 = x_177 + input_18
        x_177 = input_18 = None
        input_19 = torch.nn.functional.relu(x_178, inplace=True)
        x_178 = None
        x_179 = torch.conv2d(
            input_19,
            l_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_180 = torch.nn.functional.batch_norm(
            x_179,
            l_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_179 = l_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_181 = torch.nn.functional.relu(x_180, inplace=True)
        x_180 = None
        x_182 = torch.conv2d(
            x_181,
            l_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_181 = l_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_183 = torch.nn.functional.batch_norm(
            x_182,
            l_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_182 = l_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_184 = torch.nn.functional.relu(x_183, inplace=True)
        x_183 = None
        x_185 = torch.conv2d(
            x_184,
            l_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_184 = l_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_186 = torch.nn.functional.batch_norm(
            x_185,
            l_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_185 = l_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_187 = x_186 + input_19
        x_186 = input_19 = None
        input_20 = torch.nn.functional.relu(x_187, inplace=True)
        x_187 = None
        x_188 = torch.conv2d(
            input_20,
            l_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_189 = torch.nn.functional.batch_norm(
            x_188,
            l_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_188 = l_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_190 = torch.nn.functional.relu(x_189, inplace=True)
        x_189 = None
        x_191 = torch.conv2d(
            x_190,
            l_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_190 = l_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_192 = torch.nn.functional.batch_norm(
            x_191,
            l_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_191 = l_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_193 = torch.nn.functional.relu(x_192, inplace=True)
        x_192 = None
        x_194 = torch.conv2d(
            x_193,
            l_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_193 = l_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_195 = torch.nn.functional.batch_norm(
            x_194,
            l_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_194 = l_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_196 = x_195 + input_20
        x_195 = input_20 = None
        input_21 = torch.nn.functional.relu(x_196, inplace=True)
        x_196 = None
        x_197 = torch.conv2d(
            input_21,
            l_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_198 = torch.nn.functional.batch_norm(
            x_197,
            l_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_197 = l_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_199 = torch.nn.functional.relu(x_198, inplace=True)
        x_198 = None
        x_200 = torch.conv2d(
            x_199,
            l_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_199 = l_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_201 = torch.nn.functional.batch_norm(
            x_200,
            l_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_200 = l_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_202 = torch.nn.functional.relu(x_201, inplace=True)
        x_201 = None
        x_203 = torch.conv2d(
            x_202,
            l_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_202 = l_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_204 = torch.nn.functional.batch_norm(
            x_203,
            l_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_203 = l_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_205 = x_204 + input_21
        x_204 = input_21 = None
        input_22 = torch.nn.functional.relu(x_205, inplace=True)
        x_205 = None
        x_206 = torch.conv2d(
            input_22,
            l_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_207 = torch.nn.functional.batch_norm(
            x_206,
            l_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_206 = l_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_208 = torch.nn.functional.relu(x_207, inplace=True)
        x_207 = None
        x_209 = torch.conv2d(
            x_208,
            l_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_208 = l_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_210 = torch.nn.functional.batch_norm(
            x_209,
            l_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_209 = l_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_211 = torch.nn.functional.relu(x_210, inplace=True)
        x_210 = None
        x_212 = torch.conv2d(
            x_211,
            l_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_211 = l_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_213 = torch.nn.functional.batch_norm(
            x_212,
            l_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_212 = l_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_214 = x_213 + input_22
        x_213 = input_22 = None
        input_23 = torch.nn.functional.relu(x_214, inplace=True)
        x_214 = None
        x_215 = torch.conv2d(
            input_23,
            l_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_216 = torch.nn.functional.batch_norm(
            x_215,
            l_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_215 = l_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_217 = torch.nn.functional.relu(x_216, inplace=True)
        x_216 = None
        x_218 = torch.conv2d(
            x_217,
            l_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_217 = l_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_219 = torch.nn.functional.batch_norm(
            x_218,
            l_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_218 = l_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_220 = torch.nn.functional.relu(x_219, inplace=True)
        x_219 = None
        x_221 = torch.conv2d(
            x_220,
            l_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_220 = l_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_222 = torch.nn.functional.batch_norm(
            x_221,
            l_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_221 = l_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_223 = x_222 + input_23
        x_222 = input_23 = None
        input_24 = torch.nn.functional.relu(x_223, inplace=True)
        x_223 = None
        x_224 = torch.conv2d(
            input_24,
            l_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_225 = torch.nn.functional.batch_norm(
            x_224,
            l_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_224 = l_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_226 = torch.nn.functional.relu(x_225, inplace=True)
        x_225 = None
        x_227 = torch.conv2d(
            x_226,
            l_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_226 = l_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_228 = torch.nn.functional.batch_norm(
            x_227,
            l_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_227 = l_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_229 = torch.nn.functional.relu(x_228, inplace=True)
        x_228 = None
        x_230 = torch.conv2d(
            x_229,
            l_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_229 = l_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_231 = torch.nn.functional.batch_norm(
            x_230,
            l_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_230 = l_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_232 = x_231 + input_24
        x_231 = input_24 = None
        input_25 = torch.nn.functional.relu(x_232, inplace=True)
        x_232 = None
        x_233 = torch.conv2d(
            input_25,
            l_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_234 = torch.nn.functional.batch_norm(
            x_233,
            l_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_233 = l_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_235 = torch.nn.functional.relu(x_234, inplace=True)
        x_234 = None
        x_236 = torch.conv2d(
            x_235,
            l_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_235 = l_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_237 = torch.nn.functional.batch_norm(
            x_236,
            l_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_236 = l_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_238 = torch.nn.functional.relu(x_237, inplace=True)
        x_237 = None
        x_239 = torch.conv2d(
            x_238,
            l_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_238 = l_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_240 = torch.nn.functional.batch_norm(
            x_239,
            l_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_239 = l_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_241 = x_240 + input_25
        x_240 = input_25 = None
        input_26 = torch.nn.functional.relu(x_241, inplace=True)
        x_241 = None
        x_242 = torch.conv2d(
            input_26,
            l_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_243 = torch.nn.functional.batch_norm(
            x_242,
            l_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_242 = l_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_244 = torch.nn.functional.relu(x_243, inplace=True)
        x_243 = None
        x_245 = torch.conv2d(
            x_244,
            l_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_244 = l_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_246 = torch.nn.functional.batch_norm(
            x_245,
            l_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_245 = l_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_247 = torch.nn.functional.relu(x_246, inplace=True)
        x_246 = None
        x_248 = torch.conv2d(
            x_247,
            l_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_247 = l_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_249 = torch.nn.functional.batch_norm(
            x_248,
            l_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_248 = l_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_250 = x_249 + input_26
        x_249 = input_26 = None
        input_27 = torch.nn.functional.relu(x_250, inplace=True)
        x_250 = None
        x_251 = torch.conv2d(
            input_27,
            l_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_252 = torch.nn.functional.batch_norm(
            x_251,
            l_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_251 = l_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_253 = torch.nn.functional.relu(x_252, inplace=True)
        x_252 = None
        x_254 = torch.conv2d(
            x_253,
            l_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_253 = l_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_255 = torch.nn.functional.batch_norm(
            x_254,
            l_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_254 = l_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_256 = torch.nn.functional.relu(x_255, inplace=True)
        x_255 = None
        x_257 = torch.conv2d(
            x_256,
            l_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_256 = l_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_258 = torch.nn.functional.batch_norm(
            x_257,
            l_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_257 = l_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_259 = x_258 + input_27
        x_258 = input_27 = None
        input_28 = torch.nn.functional.relu(x_259, inplace=True)
        x_259 = None
        x_260 = torch.conv2d(
            input_28,
            l_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_261 = torch.nn.functional.batch_norm(
            x_260,
            l_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_260 = l_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_262 = torch.nn.functional.relu(x_261, inplace=True)
        x_261 = None
        x_263 = torch.conv2d(
            x_262,
            l_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_262 = l_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_264 = torch.nn.functional.batch_norm(
            x_263,
            l_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_263 = l_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_265 = torch.nn.functional.relu(x_264, inplace=True)
        x_264 = None
        x_266 = torch.conv2d(
            x_265,
            l_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_265 = l_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_267 = torch.nn.functional.batch_norm(
            x_266,
            l_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_266 = l_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_268 = x_267 + input_28
        x_267 = input_28 = None
        input_29 = torch.nn.functional.relu(x_268, inplace=True)
        x_268 = None
        x_269 = torch.conv2d(
            input_29,
            l_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_270 = torch.nn.functional.batch_norm(
            x_269,
            l_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_269 = l_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_271 = torch.nn.functional.relu(x_270, inplace=True)
        x_270 = None
        x_272 = torch.conv2d(
            x_271,
            l_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_271 = l_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_273 = torch.nn.functional.batch_norm(
            x_272,
            l_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_272 = l_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_274 = torch.nn.functional.relu(x_273, inplace=True)
        x_273 = None
        x_275 = torch.conv2d(
            x_274,
            l_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_274 = l_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_276 = torch.nn.functional.batch_norm(
            x_275,
            l_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_275 = l_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_277 = x_276 + input_29
        x_276 = input_29 = None
        input_30 = torch.nn.functional.relu(x_277, inplace=True)
        x_277 = None
        x_278 = torch.conv2d(
            input_30,
            l_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_279 = torch.nn.functional.batch_norm(
            x_278,
            l_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_278 = l_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_280 = torch.nn.functional.relu(x_279, inplace=True)
        x_279 = None
        x_281 = torch.conv2d(
            x_280,
            l_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_280 = l_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_282 = torch.nn.functional.batch_norm(
            x_281,
            l_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_281 = l_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_283 = torch.nn.functional.relu(x_282, inplace=True)
        x_282 = None
        x_284 = torch.conv2d(
            x_283,
            l_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_283 = l_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_285 = torch.nn.functional.batch_norm(
            x_284,
            l_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_284 = l_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_286 = x_285 + input_30
        x_285 = input_30 = None
        input_31 = torch.nn.functional.relu(x_286, inplace=True)
        x_286 = None
        x_287 = torch.conv2d(
            input_31,
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
        x_288 = torch.nn.functional.batch_norm(
            x_287,
            l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_287 = l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_289 = torch.nn.functional.relu(x_288, inplace=True)
        x_288 = None
        x_290 = torch.conv2d(
            x_289,
            l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_289 = l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_291 = torch.nn.functional.batch_norm(
            x_290,
            l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_290 = l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_292 = torch.nn.functional.relu(x_291, inplace=True)
        x_291 = None
        x_293 = torch._C._nn.avg_pool2d(x_292, 2, 2, 0, False, True, None)
        x_292 = None
        x_294 = torch.conv2d(
            x_293,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_293 = l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_295 = torch.nn.functional.batch_norm(
            x_294,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_294 = l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        avg_pool2d_6 = torch._C._nn.avg_pool2d(input_31, 2, 2, 0, True, False, None)
        input_31 = None
        x_296 = torch.conv2d(
            avg_pool2d_6,
            l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        avg_pool2d_6 = l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_conv_parameters_weight_ = (None)
        x_297 = torch.nn.functional.batch_norm(
            x_296,
            l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_296 = l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_bias_ = (None)
        x_298 = x_295 + x_297
        x_295 = x_297 = None
        input_32 = torch.nn.functional.relu(x_298, inplace=True)
        x_298 = None
        x_299 = torch.conv2d(
            input_32,
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
        x_300 = torch.nn.functional.batch_norm(
            x_299,
            l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_299 = l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_301 = torch.nn.functional.relu(x_300, inplace=True)
        x_300 = None
        x_302 = torch.conv2d(
            x_301,
            l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_301 = l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_303 = torch.nn.functional.batch_norm(
            x_302,
            l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_302 = l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_304 = torch.nn.functional.relu(x_303, inplace=True)
        x_303 = None
        x_305 = torch.conv2d(
            x_304,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_304 = l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_306 = torch.nn.functional.batch_norm(
            x_305,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_305 = l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_307 = x_306 + input_32
        x_306 = input_32 = None
        input_33 = torch.nn.functional.relu(x_307, inplace=True)
        x_307 = None
        x_308 = torch.conv2d(
            input_33,
            l_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_309 = torch.nn.functional.batch_norm(
            x_308,
            l_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_308 = l_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_310 = torch.nn.functional.relu(x_309, inplace=True)
        x_309 = None
        x_311 = torch.conv2d(
            x_310,
            l_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_310 = l_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_312 = torch.nn.functional.batch_norm(
            x_311,
            l_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_311 = l_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_313 = torch.nn.functional.relu(x_312, inplace=True)
        x_312 = None
        x_314 = torch.conv2d(
            x_313,
            l_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_313 = l_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_315 = torch.nn.functional.batch_norm(
            x_314,
            l_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_314 = l_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_316 = x_315 + input_33
        x_315 = input_33 = None
        input_34 = torch.nn.functional.relu(x_316, inplace=True)
        x_316 = None
        x_317 = torch.nn.functional.adaptive_avg_pool2d(input_34, 1)
        input_34 = None
        x_318 = x_317.flatten(1, -1)
        x_317 = None
        x_319 = torch.nn.functional.dropout(x_318, 0.0, False, False)
        x_318 = None
        return (x_319,)
