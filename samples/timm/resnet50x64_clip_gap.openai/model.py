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
        L_self_modules_stages_modules_1_modules_4_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_6_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_6_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_6_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_6_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_6_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_6_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_7_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_7_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_7_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_7_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_7_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_7_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_8_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_8_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_8_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_8_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_8_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_8_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_9_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_9_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_9_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_9_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_9_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_9_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_10_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_10_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_10_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_10_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_10_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_10_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_11_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_11_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_11_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_11_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_11_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_11_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_12_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_12_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_12_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_12_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_12_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_12_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_13_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_13_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_13_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_13_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_13_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_13_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_14_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_14_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_14_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_14_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_14_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_14_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_14_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_14_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_14_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_14_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_14_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_14_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_14_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_14_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_14_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_23_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_23_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_23_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_23_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_23_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_23_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_23_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_24_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_24_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_24_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_24_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_24_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_24_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_24_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_24_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_24_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_24_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_24_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_24_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_24_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_24_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_24_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_25_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_25_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_25_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_25_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_25_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_25_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_25_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_25_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_25_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_25_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_25_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_25_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_25_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_25_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_25_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_26_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_26_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_26_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_26_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_26_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_26_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_26_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_26_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_26_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_26_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_26_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_26_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_26_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_26_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_26_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_27_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_27_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_27_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_27_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_27_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_27_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_27_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_27_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_27_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_27_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_27_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_27_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_27_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_27_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_27_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_28_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_28_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_28_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_28_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_28_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_28_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_28_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_28_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_28_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_28_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_28_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_28_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_28_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_28_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_28_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_29_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_29_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_29_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_29_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_29_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_29_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_29_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_29_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_29_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_29_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_29_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_29_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_29_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_29_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_29_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_30_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_30_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_30_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_30_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_30_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_30_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_30_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_30_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_30_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_30_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_30_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_30_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_30_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_30_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_30_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_31_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_31_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_31_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_31_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_31_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_31_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_31_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_31_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_31_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_31_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_31_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_31_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_31_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_31_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_31_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_32_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_32_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_32_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_32_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_32_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_32_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_32_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_32_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_32_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_32_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_32_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_32_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_32_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_32_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_32_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_33_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_33_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_33_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_33_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_33_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_33_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_33_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_33_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_33_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_33_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_33_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_33_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_33_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_33_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_33_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_34_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_34_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_34_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_34_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_34_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_34_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_34_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_34_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_34_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_34_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_34_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_34_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_34_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_34_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_34_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_35_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_35_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_35_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_35_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_35_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_35_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_35_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_35_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_35_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_35_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_35_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_35_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_35_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_35_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_35_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_3_modules_3_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_3_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_3_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_3_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_3_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_3_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_3_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_3_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_3_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_3_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_3_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_3_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_3_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_3_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_3_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_4_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_4_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_4_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_4_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_4_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_4_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_4_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_4_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_4_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_4_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_4_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_4_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_4_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_4_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_4_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_5_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_5_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_5_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_5_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_5_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_5_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_5_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_5_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_5_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_5_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_5_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_5_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_5_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_5_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_5_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_6_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_6_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_6_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_6_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_6_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_6_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_6_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_6_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_6_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_6_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_6_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_6_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_6_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_6_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_6_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_7_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_7_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_7_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_7_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_7_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_7_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_7_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_7_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_7_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_7_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_7_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_7_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_7_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_7_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_7_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_8_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_8_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_8_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_8_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_8_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_8_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_8_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_8_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_8_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_8_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_8_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_8_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_8_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_8_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_8_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_9_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_9_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_9_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_9_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_9_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_9_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_9_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_9_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_9_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_9_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_9_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_9_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_9_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_9_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_9_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_stages_modules_1_modules_4_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_4_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_4_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_4_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_4_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_4_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_4_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_4_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_4_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_4_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_4_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_4_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_4_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_4_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_4_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_4_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_4_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_4_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_5_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_5_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_5_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_5_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_5_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_5_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_5_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_5_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_5_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_5_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_5_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_5_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_5_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_5_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_5_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_5_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_5_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_5_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_6_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_6_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_6_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_6_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_6_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_6_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_6_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_6_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_6_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_6_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_6_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_6_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_6_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_6_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_6_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_6_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_6_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_6_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_6_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_6_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_6_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_6_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_6_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_6_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_6_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_6_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_6_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_6_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_6_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_6_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_7_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_7_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_7_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_7_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_7_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_7_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_7_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_7_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_7_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_7_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_7_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_7_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_7_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_7_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_7_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_7_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_7_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_7_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_7_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_7_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_7_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_7_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_7_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_7_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_7_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_7_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_7_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_7_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_7_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_7_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_8_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_8_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_8_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_8_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_8_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_8_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_8_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_8_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_8_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_8_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_8_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_8_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_8_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_8_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_8_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_8_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_8_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_8_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_8_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_8_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_8_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_8_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_8_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_8_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_8_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_8_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_8_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_8_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_8_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_8_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_9_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_9_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_9_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_9_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_9_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_9_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_9_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_9_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_9_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_9_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_9_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_9_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_9_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_9_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_9_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_9_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_9_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_9_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_9_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_9_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_9_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_9_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_9_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_9_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_9_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_9_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_9_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_9_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_9_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_9_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_10_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_10_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_10_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_10_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_10_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_10_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_10_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_10_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_10_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_10_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_10_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_10_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_10_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_10_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_10_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_10_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_10_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_10_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_10_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_10_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_10_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_10_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_10_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_10_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_10_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_10_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_10_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_10_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_10_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_10_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_11_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_11_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_11_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_11_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_11_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_11_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_11_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_11_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_11_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_11_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_11_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_11_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_11_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_11_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_11_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_11_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_11_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_11_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_11_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_11_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_11_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_11_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_11_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_11_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_11_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_11_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_11_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_11_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_11_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_11_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_12_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_12_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_12_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_12_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_12_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_12_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_12_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_12_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_12_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_12_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_12_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_12_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_12_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_12_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_12_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_12_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_12_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_12_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_12_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_12_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_12_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_12_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_12_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_12_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_12_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_12_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_12_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_12_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_12_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_12_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_13_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_13_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_13_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_13_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_13_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_13_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_13_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_13_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_13_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_13_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_13_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_13_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_13_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_13_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_13_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_13_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_13_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_13_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_13_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_13_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_13_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_13_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_13_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_13_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_13_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_13_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_13_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_13_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_13_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_13_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_14_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_14_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_14_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_14_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_14_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_14_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_14_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_14_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_14_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_14_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_14_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_14_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_14_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_14_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_14_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_14_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_14_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_14_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_14_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_14_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_14_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_14_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_14_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_14_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_14_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_14_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_14_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_14_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_14_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_14_modules_conv3_1x1_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_23_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_23_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_23_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_23_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_23_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_23_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_23_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_23_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_23_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_23_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_23_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_23_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_23_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_23_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_23_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_23_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_23_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_23_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_23_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_23_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_23_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_23_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_23_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_23_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_23_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_23_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_23_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_23_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_23_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_23_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_24_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_24_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_24_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_24_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_24_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_24_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_24_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_24_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_24_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_24_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_24_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_24_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_24_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_24_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_24_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_24_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_24_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_24_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_24_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_24_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_24_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_24_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_24_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_24_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_24_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_24_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_24_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_24_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_24_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_24_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_25_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_25_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_25_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_25_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_25_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_25_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_25_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_25_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_25_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_25_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_25_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_25_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_25_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_25_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_25_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_25_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_25_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_25_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_25_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_25_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_25_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_25_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_25_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_25_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_25_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_25_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_25_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_25_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_25_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_25_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_26_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_26_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_26_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_26_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_26_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_26_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_26_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_26_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_26_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_26_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_26_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_26_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_26_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_26_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_26_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_26_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_26_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_26_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_26_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_26_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_26_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_26_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_26_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_26_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_26_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_26_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_26_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_26_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_26_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_26_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_27_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_27_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_27_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_27_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_27_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_27_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_27_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_27_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_27_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_27_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_27_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_27_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_27_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_27_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_27_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_27_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_27_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_27_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_27_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_27_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_27_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_27_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_27_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_27_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_27_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_27_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_27_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_27_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_27_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_27_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_28_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_28_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_28_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_28_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_28_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_28_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_28_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_28_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_28_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_28_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_28_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_28_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_28_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_28_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_28_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_28_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_28_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_28_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_28_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_28_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_28_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_28_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_28_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_28_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_28_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_28_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_28_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_28_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_28_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_28_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_29_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_29_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_29_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_29_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_29_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_29_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_29_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_29_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_29_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_29_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_29_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_29_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_29_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_29_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_29_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_29_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_29_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_29_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_29_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_29_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_29_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_29_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_29_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_29_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_29_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_29_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_29_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_29_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_29_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_29_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_30_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_30_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_30_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_30_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_30_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_30_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_30_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_30_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_30_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_30_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_30_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_30_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_30_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_30_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_30_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_30_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_30_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_30_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_30_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_30_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_30_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_30_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_30_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_30_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_30_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_30_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_30_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_30_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_30_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_30_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_31_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_31_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_31_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_31_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_31_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_31_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_31_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_31_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_31_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_31_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_31_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_31_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_31_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_31_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_31_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_31_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_31_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_31_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_31_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_31_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_31_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_31_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_31_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_31_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_31_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_31_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_31_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_31_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_31_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_31_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_32_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_32_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_32_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_32_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_32_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_32_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_32_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_32_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_32_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_32_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_32_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_32_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_32_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_32_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_32_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_32_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_32_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_32_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_32_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_32_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_32_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_32_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_32_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_32_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_32_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_32_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_32_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_32_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_32_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_32_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_33_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_33_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_33_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_33_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_33_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_33_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_33_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_33_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_33_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_33_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_33_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_33_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_33_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_33_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_33_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_33_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_33_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_33_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_33_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_33_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_33_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_33_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_33_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_33_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_33_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_33_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_33_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_33_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_33_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_33_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_34_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_34_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_34_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_34_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_34_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_34_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_34_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_34_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_34_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_34_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_34_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_34_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_34_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_34_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_34_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_34_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_34_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_34_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_34_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_34_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_34_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_34_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_34_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_34_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_34_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_34_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_34_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_34_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_34_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_34_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_35_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_35_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_35_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_35_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_35_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_35_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_35_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_35_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_35_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_35_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_35_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_35_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_35_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_35_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_35_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_35_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_35_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_35_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_35_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_35_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_35_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_35_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_35_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_35_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_35_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_35_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_35_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_35_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_35_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_35_modules_conv3_1x1_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_3_modules_3_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_3_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_3_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_3_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_3_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_3_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_3_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_3_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_3_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_3_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_3_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_3_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_3_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_3_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_3_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_3_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_3_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_3_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_3_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_3_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_3_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_3_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_3_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_3_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_3_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_3_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_3_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_3_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_3_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_3_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_4_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_4_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_4_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_4_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_4_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_4_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_4_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_4_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_4_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_4_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_4_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_4_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_4_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_4_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_4_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_4_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_4_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_4_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_4_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_4_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_4_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_4_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_4_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_4_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_4_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_4_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_4_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_4_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_4_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_4_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_5_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_5_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_5_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_5_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_5_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_5_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_5_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_5_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_5_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_5_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_5_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_5_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_5_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_5_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_5_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_5_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_5_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_5_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_5_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_5_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_5_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_5_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_5_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_5_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_5_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_5_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_5_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_5_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_5_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_5_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_6_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_6_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_6_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_6_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_6_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_6_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_6_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_6_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_6_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_6_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_6_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_6_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_6_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_6_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_6_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_6_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_6_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_6_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_6_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_6_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_6_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_6_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_6_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_6_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_6_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_6_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_6_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_6_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_6_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_6_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_7_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_7_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_7_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_7_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_7_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_7_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_7_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_7_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_7_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_7_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_7_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_7_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_7_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_7_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_7_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_7_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_7_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_7_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_7_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_7_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_7_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_7_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_7_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_7_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_7_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_7_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_7_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_7_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_7_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_7_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_8_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_8_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_8_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_8_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_8_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_8_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_8_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_8_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_8_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_8_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_8_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_8_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_8_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_8_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_8_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_8_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_8_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_8_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_8_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_8_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_8_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_8_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_8_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_8_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_8_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_8_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_8_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_8_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_8_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_8_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_9_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_9_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_9_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_9_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_9_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_9_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_9_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_9_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_9_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_9_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_9_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_9_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_9_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_9_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_9_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_9_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_9_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_9_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_9_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_9_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_9_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_9_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_9_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_9_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_9_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_9_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_9_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_9_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_9_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_9_modules_conv3_1x1_modules_bn_parameters_bias_
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
            l_self_modules_stages_modules_1_modules_4_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_78 = torch.nn.functional.batch_norm(
            x_77,
            l_self_modules_stages_modules_1_modules_4_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_4_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_4_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_4_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_77 = l_self_modules_stages_modules_1_modules_4_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_4_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_4_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_4_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_79 = torch.nn.functional.relu(x_78, inplace=True)
        x_78 = None
        x_80 = torch.conv2d(
            x_79,
            l_self_modules_stages_modules_1_modules_4_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_79 = l_self_modules_stages_modules_1_modules_4_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_81 = torch.nn.functional.batch_norm(
            x_80,
            l_self_modules_stages_modules_1_modules_4_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_4_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_4_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_4_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_80 = l_self_modules_stages_modules_1_modules_4_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_4_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_4_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_4_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_82 = torch.nn.functional.relu(x_81, inplace=True)
        x_81 = None
        x_83 = torch.conv2d(
            x_82,
            l_self_modules_stages_modules_1_modules_4_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_82 = l_self_modules_stages_modules_1_modules_4_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_84 = torch.nn.functional.batch_norm(
            x_83,
            l_self_modules_stages_modules_1_modules_4_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_4_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_4_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_4_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_83 = l_self_modules_stages_modules_1_modules_4_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_4_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_4_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_4_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_85 = x_84 + input_8
        x_84 = input_8 = None
        input_9 = torch.nn.functional.relu(x_85, inplace=True)
        x_85 = None
        x_86 = torch.conv2d(
            input_9,
            l_self_modules_stages_modules_1_modules_5_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_5_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_87 = torch.nn.functional.batch_norm(
            x_86,
            l_self_modules_stages_modules_1_modules_5_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_5_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_5_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_5_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_86 = l_self_modules_stages_modules_1_modules_5_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_5_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_5_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_5_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_88 = torch.nn.functional.relu(x_87, inplace=True)
        x_87 = None
        x_89 = torch.conv2d(
            x_88,
            l_self_modules_stages_modules_1_modules_5_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_88 = l_self_modules_stages_modules_1_modules_5_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_90 = torch.nn.functional.batch_norm(
            x_89,
            l_self_modules_stages_modules_1_modules_5_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_5_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_5_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_5_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_89 = l_self_modules_stages_modules_1_modules_5_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_5_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_5_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_5_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_91 = torch.nn.functional.relu(x_90, inplace=True)
        x_90 = None
        x_92 = torch.conv2d(
            x_91,
            l_self_modules_stages_modules_1_modules_5_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_91 = l_self_modules_stages_modules_1_modules_5_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_93 = torch.nn.functional.batch_norm(
            x_92,
            l_self_modules_stages_modules_1_modules_5_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_5_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_5_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_5_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_92 = l_self_modules_stages_modules_1_modules_5_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_5_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_5_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_5_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_94 = x_93 + input_9
        x_93 = input_9 = None
        input_10 = torch.nn.functional.relu(x_94, inplace=True)
        x_94 = None
        x_95 = torch.conv2d(
            input_10,
            l_self_modules_stages_modules_1_modules_6_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_6_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_96 = torch.nn.functional.batch_norm(
            x_95,
            l_self_modules_stages_modules_1_modules_6_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_6_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_6_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_6_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_95 = l_self_modules_stages_modules_1_modules_6_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_6_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_6_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_6_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_97 = torch.nn.functional.relu(x_96, inplace=True)
        x_96 = None
        x_98 = torch.conv2d(
            x_97,
            l_self_modules_stages_modules_1_modules_6_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_97 = l_self_modules_stages_modules_1_modules_6_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_99 = torch.nn.functional.batch_norm(
            x_98,
            l_self_modules_stages_modules_1_modules_6_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_6_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_6_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_6_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_98 = l_self_modules_stages_modules_1_modules_6_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_6_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_6_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_6_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_100 = torch.nn.functional.relu(x_99, inplace=True)
        x_99 = None
        x_101 = torch.conv2d(
            x_100,
            l_self_modules_stages_modules_1_modules_6_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_100 = l_self_modules_stages_modules_1_modules_6_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_102 = torch.nn.functional.batch_norm(
            x_101,
            l_self_modules_stages_modules_1_modules_6_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_6_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_6_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_6_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_101 = l_self_modules_stages_modules_1_modules_6_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_6_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_6_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_6_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_103 = x_102 + input_10
        x_102 = input_10 = None
        input_11 = torch.nn.functional.relu(x_103, inplace=True)
        x_103 = None
        x_104 = torch.conv2d(
            input_11,
            l_self_modules_stages_modules_1_modules_7_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_7_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_105 = torch.nn.functional.batch_norm(
            x_104,
            l_self_modules_stages_modules_1_modules_7_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_7_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_7_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_7_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_104 = l_self_modules_stages_modules_1_modules_7_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_7_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_7_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_7_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_106 = torch.nn.functional.relu(x_105, inplace=True)
        x_105 = None
        x_107 = torch.conv2d(
            x_106,
            l_self_modules_stages_modules_1_modules_7_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_106 = l_self_modules_stages_modules_1_modules_7_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_108 = torch.nn.functional.batch_norm(
            x_107,
            l_self_modules_stages_modules_1_modules_7_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_7_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_7_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_7_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_107 = l_self_modules_stages_modules_1_modules_7_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_7_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_7_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_7_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_109 = torch.nn.functional.relu(x_108, inplace=True)
        x_108 = None
        x_110 = torch.conv2d(
            x_109,
            l_self_modules_stages_modules_1_modules_7_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_109 = l_self_modules_stages_modules_1_modules_7_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_stages_modules_1_modules_7_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_7_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_7_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_7_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_110 = l_self_modules_stages_modules_1_modules_7_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_7_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_7_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_7_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_112 = x_111 + input_11
        x_111 = input_11 = None
        input_12 = torch.nn.functional.relu(x_112, inplace=True)
        x_112 = None
        x_113 = torch.conv2d(
            input_12,
            l_self_modules_stages_modules_1_modules_8_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_8_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_114 = torch.nn.functional.batch_norm(
            x_113,
            l_self_modules_stages_modules_1_modules_8_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_8_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_8_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_8_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_113 = l_self_modules_stages_modules_1_modules_8_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_8_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_8_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_8_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_115 = torch.nn.functional.relu(x_114, inplace=True)
        x_114 = None
        x_116 = torch.conv2d(
            x_115,
            l_self_modules_stages_modules_1_modules_8_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_115 = l_self_modules_stages_modules_1_modules_8_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_117 = torch.nn.functional.batch_norm(
            x_116,
            l_self_modules_stages_modules_1_modules_8_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_8_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_8_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_8_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_116 = l_self_modules_stages_modules_1_modules_8_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_8_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_8_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_8_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_118 = torch.nn.functional.relu(x_117, inplace=True)
        x_117 = None
        x_119 = torch.conv2d(
            x_118,
            l_self_modules_stages_modules_1_modules_8_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_118 = l_self_modules_stages_modules_1_modules_8_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_120 = torch.nn.functional.batch_norm(
            x_119,
            l_self_modules_stages_modules_1_modules_8_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_8_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_8_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_8_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_119 = l_self_modules_stages_modules_1_modules_8_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_8_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_8_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_8_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_121 = x_120 + input_12
        x_120 = input_12 = None
        input_13 = torch.nn.functional.relu(x_121, inplace=True)
        x_121 = None
        x_122 = torch.conv2d(
            input_13,
            l_self_modules_stages_modules_1_modules_9_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_9_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_123 = torch.nn.functional.batch_norm(
            x_122,
            l_self_modules_stages_modules_1_modules_9_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_9_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_9_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_9_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_122 = l_self_modules_stages_modules_1_modules_9_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_9_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_9_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_9_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_124 = torch.nn.functional.relu(x_123, inplace=True)
        x_123 = None
        x_125 = torch.conv2d(
            x_124,
            l_self_modules_stages_modules_1_modules_9_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_124 = l_self_modules_stages_modules_1_modules_9_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_126 = torch.nn.functional.batch_norm(
            x_125,
            l_self_modules_stages_modules_1_modules_9_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_9_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_9_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_9_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_125 = l_self_modules_stages_modules_1_modules_9_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_9_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_9_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_9_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_127 = torch.nn.functional.relu(x_126, inplace=True)
        x_126 = None
        x_128 = torch.conv2d(
            x_127,
            l_self_modules_stages_modules_1_modules_9_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_127 = l_self_modules_stages_modules_1_modules_9_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_129 = torch.nn.functional.batch_norm(
            x_128,
            l_self_modules_stages_modules_1_modules_9_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_9_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_9_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_9_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_128 = l_self_modules_stages_modules_1_modules_9_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_9_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_9_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_9_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_130 = x_129 + input_13
        x_129 = input_13 = None
        input_14 = torch.nn.functional.relu(x_130, inplace=True)
        x_130 = None
        x_131 = torch.conv2d(
            input_14,
            l_self_modules_stages_modules_1_modules_10_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_10_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_132 = torch.nn.functional.batch_norm(
            x_131,
            l_self_modules_stages_modules_1_modules_10_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_10_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_10_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_10_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_131 = l_self_modules_stages_modules_1_modules_10_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_10_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_10_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_10_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_133 = torch.nn.functional.relu(x_132, inplace=True)
        x_132 = None
        x_134 = torch.conv2d(
            x_133,
            l_self_modules_stages_modules_1_modules_10_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_133 = l_self_modules_stages_modules_1_modules_10_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_135 = torch.nn.functional.batch_norm(
            x_134,
            l_self_modules_stages_modules_1_modules_10_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_10_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_10_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_10_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_134 = l_self_modules_stages_modules_1_modules_10_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_10_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_10_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_10_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_136 = torch.nn.functional.relu(x_135, inplace=True)
        x_135 = None
        x_137 = torch.conv2d(
            x_136,
            l_self_modules_stages_modules_1_modules_10_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_136 = l_self_modules_stages_modules_1_modules_10_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_138 = torch.nn.functional.batch_norm(
            x_137,
            l_self_modules_stages_modules_1_modules_10_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_10_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_10_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_10_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_137 = l_self_modules_stages_modules_1_modules_10_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_10_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_10_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_10_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_139 = x_138 + input_14
        x_138 = input_14 = None
        input_15 = torch.nn.functional.relu(x_139, inplace=True)
        x_139 = None
        x_140 = torch.conv2d(
            input_15,
            l_self_modules_stages_modules_1_modules_11_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_11_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_141 = torch.nn.functional.batch_norm(
            x_140,
            l_self_modules_stages_modules_1_modules_11_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_11_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_11_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_11_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_140 = l_self_modules_stages_modules_1_modules_11_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_11_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_11_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_11_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_142 = torch.nn.functional.relu(x_141, inplace=True)
        x_141 = None
        x_143 = torch.conv2d(
            x_142,
            l_self_modules_stages_modules_1_modules_11_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_142 = l_self_modules_stages_modules_1_modules_11_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_144 = torch.nn.functional.batch_norm(
            x_143,
            l_self_modules_stages_modules_1_modules_11_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_11_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_11_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_11_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_143 = l_self_modules_stages_modules_1_modules_11_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_11_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_11_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_11_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_145 = torch.nn.functional.relu(x_144, inplace=True)
        x_144 = None
        x_146 = torch.conv2d(
            x_145,
            l_self_modules_stages_modules_1_modules_11_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_145 = l_self_modules_stages_modules_1_modules_11_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_147 = torch.nn.functional.batch_norm(
            x_146,
            l_self_modules_stages_modules_1_modules_11_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_11_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_11_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_11_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_146 = l_self_modules_stages_modules_1_modules_11_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_11_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_11_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_11_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_148 = x_147 + input_15
        x_147 = input_15 = None
        input_16 = torch.nn.functional.relu(x_148, inplace=True)
        x_148 = None
        x_149 = torch.conv2d(
            input_16,
            l_self_modules_stages_modules_1_modules_12_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_12_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_150 = torch.nn.functional.batch_norm(
            x_149,
            l_self_modules_stages_modules_1_modules_12_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_12_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_12_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_12_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_149 = l_self_modules_stages_modules_1_modules_12_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_12_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_12_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_12_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_151 = torch.nn.functional.relu(x_150, inplace=True)
        x_150 = None
        x_152 = torch.conv2d(
            x_151,
            l_self_modules_stages_modules_1_modules_12_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_151 = l_self_modules_stages_modules_1_modules_12_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_153 = torch.nn.functional.batch_norm(
            x_152,
            l_self_modules_stages_modules_1_modules_12_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_12_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_12_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_12_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_152 = l_self_modules_stages_modules_1_modules_12_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_12_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_12_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_12_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_154 = torch.nn.functional.relu(x_153, inplace=True)
        x_153 = None
        x_155 = torch.conv2d(
            x_154,
            l_self_modules_stages_modules_1_modules_12_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_154 = l_self_modules_stages_modules_1_modules_12_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_156 = torch.nn.functional.batch_norm(
            x_155,
            l_self_modules_stages_modules_1_modules_12_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_12_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_12_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_12_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_155 = l_self_modules_stages_modules_1_modules_12_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_12_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_12_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_12_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_157 = x_156 + input_16
        x_156 = input_16 = None
        input_17 = torch.nn.functional.relu(x_157, inplace=True)
        x_157 = None
        x_158 = torch.conv2d(
            input_17,
            l_self_modules_stages_modules_1_modules_13_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_13_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_159 = torch.nn.functional.batch_norm(
            x_158,
            l_self_modules_stages_modules_1_modules_13_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_13_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_13_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_13_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_158 = l_self_modules_stages_modules_1_modules_13_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_13_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_13_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_13_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_160 = torch.nn.functional.relu(x_159, inplace=True)
        x_159 = None
        x_161 = torch.conv2d(
            x_160,
            l_self_modules_stages_modules_1_modules_13_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_160 = l_self_modules_stages_modules_1_modules_13_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_162 = torch.nn.functional.batch_norm(
            x_161,
            l_self_modules_stages_modules_1_modules_13_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_13_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_13_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_13_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_161 = l_self_modules_stages_modules_1_modules_13_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_13_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_13_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_13_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_163 = torch.nn.functional.relu(x_162, inplace=True)
        x_162 = None
        x_164 = torch.conv2d(
            x_163,
            l_self_modules_stages_modules_1_modules_13_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_163 = l_self_modules_stages_modules_1_modules_13_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_165 = torch.nn.functional.batch_norm(
            x_164,
            l_self_modules_stages_modules_1_modules_13_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_13_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_13_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_13_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_164 = l_self_modules_stages_modules_1_modules_13_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_13_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_13_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_13_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_166 = x_165 + input_17
        x_165 = input_17 = None
        input_18 = torch.nn.functional.relu(x_166, inplace=True)
        x_166 = None
        x_167 = torch.conv2d(
            input_18,
            l_self_modules_stages_modules_1_modules_14_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_14_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_168 = torch.nn.functional.batch_norm(
            x_167,
            l_self_modules_stages_modules_1_modules_14_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_14_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_14_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_14_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_167 = l_self_modules_stages_modules_1_modules_14_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_14_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_14_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_14_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_169 = torch.nn.functional.relu(x_168, inplace=True)
        x_168 = None
        x_170 = torch.conv2d(
            x_169,
            l_self_modules_stages_modules_1_modules_14_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_169 = l_self_modules_stages_modules_1_modules_14_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_171 = torch.nn.functional.batch_norm(
            x_170,
            l_self_modules_stages_modules_1_modules_14_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_14_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_14_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_14_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_170 = l_self_modules_stages_modules_1_modules_14_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_14_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_14_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_14_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_172 = torch.nn.functional.relu(x_171, inplace=True)
        x_171 = None
        x_173 = torch.conv2d(
            x_172,
            l_self_modules_stages_modules_1_modules_14_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_172 = l_self_modules_stages_modules_1_modules_14_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_174 = torch.nn.functional.batch_norm(
            x_173,
            l_self_modules_stages_modules_1_modules_14_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_14_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_14_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_14_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_173 = l_self_modules_stages_modules_1_modules_14_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_14_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_14_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_14_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_175 = x_174 + input_18
        x_174 = input_18 = None
        input_19 = torch.nn.functional.relu(x_175, inplace=True)
        x_175 = None
        x_176 = torch.conv2d(
            input_19,
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
        x_177 = torch.nn.functional.batch_norm(
            x_176,
            l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_176 = l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_178 = torch.nn.functional.relu(x_177, inplace=True)
        x_177 = None
        x_179 = torch.conv2d(
            x_178,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_178 = l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_180 = torch.nn.functional.batch_norm(
            x_179,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_179 = l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_181 = torch.nn.functional.relu(x_180, inplace=True)
        x_180 = None
        x_182 = torch._C._nn.avg_pool2d(x_181, 2, 2, 0, False, True, None)
        x_181 = None
        x_183 = torch.conv2d(
            x_182,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_182 = l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_184 = torch.nn.functional.batch_norm(
            x_183,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_183 = l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        avg_pool2d_4 = torch._C._nn.avg_pool2d(input_19, 2, 2, 0, True, False, None)
        input_19 = None
        x_185 = torch.conv2d(
            avg_pool2d_4,
            l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        avg_pool2d_4 = l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_conv_parameters_weight_ = (None)
        x_186 = torch.nn.functional.batch_norm(
            x_185,
            l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_185 = l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_bias_ = (None)
        x_187 = x_184 + x_186
        x_184 = x_186 = None
        input_20 = torch.nn.functional.relu(x_187, inplace=True)
        x_187 = None
        x_188 = torch.conv2d(
            input_20,
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
        x_189 = torch.nn.functional.batch_norm(
            x_188,
            l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_188 = l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_190 = torch.nn.functional.relu(x_189, inplace=True)
        x_189 = None
        x_191 = torch.conv2d(
            x_190,
            l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_190 = l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_192 = torch.nn.functional.batch_norm(
            x_191,
            l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_191 = l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_193 = torch.nn.functional.relu(x_192, inplace=True)
        x_192 = None
        x_194 = torch.conv2d(
            x_193,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_193 = l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_195 = torch.nn.functional.batch_norm(
            x_194,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_194 = l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_196 = x_195 + input_20
        x_195 = input_20 = None
        input_21 = torch.nn.functional.relu(x_196, inplace=True)
        x_196 = None
        x_197 = torch.conv2d(
            input_21,
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
        x_198 = torch.nn.functional.batch_norm(
            x_197,
            l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_197 = l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_199 = torch.nn.functional.relu(x_198, inplace=True)
        x_198 = None
        x_200 = torch.conv2d(
            x_199,
            l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_199 = l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_201 = torch.nn.functional.batch_norm(
            x_200,
            l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_200 = l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_202 = torch.nn.functional.relu(x_201, inplace=True)
        x_201 = None
        x_203 = torch.conv2d(
            x_202,
            l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_202 = l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_204 = torch.nn.functional.batch_norm(
            x_203,
            l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_203 = l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_205 = x_204 + input_21
        x_204 = input_21 = None
        input_22 = torch.nn.functional.relu(x_205, inplace=True)
        x_205 = None
        x_206 = torch.conv2d(
            input_22,
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
        x_207 = torch.nn.functional.batch_norm(
            x_206,
            l_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_206 = l_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_3_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_208 = torch.nn.functional.relu(x_207, inplace=True)
        x_207 = None
        x_209 = torch.conv2d(
            x_208,
            l_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_208 = l_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_210 = torch.nn.functional.batch_norm(
            x_209,
            l_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_209 = l_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_3_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_211 = torch.nn.functional.relu(x_210, inplace=True)
        x_210 = None
        x_212 = torch.conv2d(
            x_211,
            l_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_211 = l_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_213 = torch.nn.functional.batch_norm(
            x_212,
            l_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_212 = l_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_3_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_214 = x_213 + input_22
        x_213 = input_22 = None
        input_23 = torch.nn.functional.relu(x_214, inplace=True)
        x_214 = None
        x_215 = torch.conv2d(
            input_23,
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
        x_216 = torch.nn.functional.batch_norm(
            x_215,
            l_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_215 = l_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_217 = torch.nn.functional.relu(x_216, inplace=True)
        x_216 = None
        x_218 = torch.conv2d(
            x_217,
            l_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_217 = l_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_219 = torch.nn.functional.batch_norm(
            x_218,
            l_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_218 = l_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_220 = torch.nn.functional.relu(x_219, inplace=True)
        x_219 = None
        x_221 = torch.conv2d(
            x_220,
            l_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_220 = l_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_222 = torch.nn.functional.batch_norm(
            x_221,
            l_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_221 = l_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_223 = x_222 + input_23
        x_222 = input_23 = None
        input_24 = torch.nn.functional.relu(x_223, inplace=True)
        x_223 = None
        x_224 = torch.conv2d(
            input_24,
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
        x_225 = torch.nn.functional.batch_norm(
            x_224,
            l_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_224 = l_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_5_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_226 = torch.nn.functional.relu(x_225, inplace=True)
        x_225 = None
        x_227 = torch.conv2d(
            x_226,
            l_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_226 = l_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_228 = torch.nn.functional.batch_norm(
            x_227,
            l_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_227 = l_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_5_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_229 = torch.nn.functional.relu(x_228, inplace=True)
        x_228 = None
        x_230 = torch.conv2d(
            x_229,
            l_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_229 = l_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_231 = torch.nn.functional.batch_norm(
            x_230,
            l_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_230 = l_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_5_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_232 = x_231 + input_24
        x_231 = input_24 = None
        input_25 = torch.nn.functional.relu(x_232, inplace=True)
        x_232 = None
        x_233 = torch.conv2d(
            input_25,
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
        x_234 = torch.nn.functional.batch_norm(
            x_233,
            l_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_233 = l_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_6_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_235 = torch.nn.functional.relu(x_234, inplace=True)
        x_234 = None
        x_236 = torch.conv2d(
            x_235,
            l_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_235 = l_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_237 = torch.nn.functional.batch_norm(
            x_236,
            l_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_236 = l_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_6_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_238 = torch.nn.functional.relu(x_237, inplace=True)
        x_237 = None
        x_239 = torch.conv2d(
            x_238,
            l_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_238 = l_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_240 = torch.nn.functional.batch_norm(
            x_239,
            l_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_239 = l_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_6_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_241 = x_240 + input_25
        x_240 = input_25 = None
        input_26 = torch.nn.functional.relu(x_241, inplace=True)
        x_241 = None
        x_242 = torch.conv2d(
            input_26,
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
        x_243 = torch.nn.functional.batch_norm(
            x_242,
            l_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_242 = l_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_7_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_244 = torch.nn.functional.relu(x_243, inplace=True)
        x_243 = None
        x_245 = torch.conv2d(
            x_244,
            l_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_244 = l_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_246 = torch.nn.functional.batch_norm(
            x_245,
            l_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_245 = l_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_7_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_247 = torch.nn.functional.relu(x_246, inplace=True)
        x_246 = None
        x_248 = torch.conv2d(
            x_247,
            l_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_247 = l_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_249 = torch.nn.functional.batch_norm(
            x_248,
            l_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_248 = l_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_7_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_250 = x_249 + input_26
        x_249 = input_26 = None
        input_27 = torch.nn.functional.relu(x_250, inplace=True)
        x_250 = None
        x_251 = torch.conv2d(
            input_27,
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
        x_252 = torch.nn.functional.batch_norm(
            x_251,
            l_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_251 = l_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_8_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_253 = torch.nn.functional.relu(x_252, inplace=True)
        x_252 = None
        x_254 = torch.conv2d(
            x_253,
            l_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_253 = l_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_255 = torch.nn.functional.batch_norm(
            x_254,
            l_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_254 = l_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_8_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_256 = torch.nn.functional.relu(x_255, inplace=True)
        x_255 = None
        x_257 = torch.conv2d(
            x_256,
            l_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_256 = l_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_258 = torch.nn.functional.batch_norm(
            x_257,
            l_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_257 = l_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_8_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_259 = x_258 + input_27
        x_258 = input_27 = None
        input_28 = torch.nn.functional.relu(x_259, inplace=True)
        x_259 = None
        x_260 = torch.conv2d(
            input_28,
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
        x_261 = torch.nn.functional.batch_norm(
            x_260,
            l_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_260 = l_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_9_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_262 = torch.nn.functional.relu(x_261, inplace=True)
        x_261 = None
        x_263 = torch.conv2d(
            x_262,
            l_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_262 = l_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_264 = torch.nn.functional.batch_norm(
            x_263,
            l_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_263 = l_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_9_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_265 = torch.nn.functional.relu(x_264, inplace=True)
        x_264 = None
        x_266 = torch.conv2d(
            x_265,
            l_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_265 = l_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_267 = torch.nn.functional.batch_norm(
            x_266,
            l_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_266 = l_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_9_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_268 = x_267 + input_28
        x_267 = input_28 = None
        input_29 = torch.nn.functional.relu(x_268, inplace=True)
        x_268 = None
        x_269 = torch.conv2d(
            input_29,
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
        x_270 = torch.nn.functional.batch_norm(
            x_269,
            l_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_269 = l_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_10_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_271 = torch.nn.functional.relu(x_270, inplace=True)
        x_270 = None
        x_272 = torch.conv2d(
            x_271,
            l_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_271 = l_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_273 = torch.nn.functional.batch_norm(
            x_272,
            l_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_272 = l_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_10_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_274 = torch.nn.functional.relu(x_273, inplace=True)
        x_273 = None
        x_275 = torch.conv2d(
            x_274,
            l_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_274 = l_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_276 = torch.nn.functional.batch_norm(
            x_275,
            l_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_275 = l_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_10_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_277 = x_276 + input_29
        x_276 = input_29 = None
        input_30 = torch.nn.functional.relu(x_277, inplace=True)
        x_277 = None
        x_278 = torch.conv2d(
            input_30,
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
        x_279 = torch.nn.functional.batch_norm(
            x_278,
            l_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_278 = l_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_11_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_280 = torch.nn.functional.relu(x_279, inplace=True)
        x_279 = None
        x_281 = torch.conv2d(
            x_280,
            l_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_280 = l_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_282 = torch.nn.functional.batch_norm(
            x_281,
            l_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_281 = l_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_11_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_283 = torch.nn.functional.relu(x_282, inplace=True)
        x_282 = None
        x_284 = torch.conv2d(
            x_283,
            l_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_283 = l_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_285 = torch.nn.functional.batch_norm(
            x_284,
            l_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_284 = l_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_11_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_286 = x_285 + input_30
        x_285 = input_30 = None
        input_31 = torch.nn.functional.relu(x_286, inplace=True)
        x_286 = None
        x_287 = torch.conv2d(
            input_31,
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
        x_288 = torch.nn.functional.batch_norm(
            x_287,
            l_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_287 = l_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_12_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_289 = torch.nn.functional.relu(x_288, inplace=True)
        x_288 = None
        x_290 = torch.conv2d(
            x_289,
            l_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_289 = l_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_291 = torch.nn.functional.batch_norm(
            x_290,
            l_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_290 = l_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_12_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_292 = torch.nn.functional.relu(x_291, inplace=True)
        x_291 = None
        x_293 = torch.conv2d(
            x_292,
            l_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_292 = l_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_294 = torch.nn.functional.batch_norm(
            x_293,
            l_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_293 = l_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_12_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_295 = x_294 + input_31
        x_294 = input_31 = None
        input_32 = torch.nn.functional.relu(x_295, inplace=True)
        x_295 = None
        x_296 = torch.conv2d(
            input_32,
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
        x_297 = torch.nn.functional.batch_norm(
            x_296,
            l_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_296 = l_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_13_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_298 = torch.nn.functional.relu(x_297, inplace=True)
        x_297 = None
        x_299 = torch.conv2d(
            x_298,
            l_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_298 = l_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_300 = torch.nn.functional.batch_norm(
            x_299,
            l_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_299 = l_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_13_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_301 = torch.nn.functional.relu(x_300, inplace=True)
        x_300 = None
        x_302 = torch.conv2d(
            x_301,
            l_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_301 = l_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_303 = torch.nn.functional.batch_norm(
            x_302,
            l_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_302 = l_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_13_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_304 = x_303 + input_32
        x_303 = input_32 = None
        input_33 = torch.nn.functional.relu(x_304, inplace=True)
        x_304 = None
        x_305 = torch.conv2d(
            input_33,
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
        x_306 = torch.nn.functional.batch_norm(
            x_305,
            l_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_305 = l_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_14_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_307 = torch.nn.functional.relu(x_306, inplace=True)
        x_306 = None
        x_308 = torch.conv2d(
            x_307,
            l_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_307 = l_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_309 = torch.nn.functional.batch_norm(
            x_308,
            l_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_308 = l_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_14_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_310 = torch.nn.functional.relu(x_309, inplace=True)
        x_309 = None
        x_311 = torch.conv2d(
            x_310,
            l_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_310 = l_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_312 = torch.nn.functional.batch_norm(
            x_311,
            l_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_311 = l_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_14_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_313 = x_312 + input_33
        x_312 = input_33 = None
        input_34 = torch.nn.functional.relu(x_313, inplace=True)
        x_313 = None
        x_314 = torch.conv2d(
            input_34,
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
        x_315 = torch.nn.functional.batch_norm(
            x_314,
            l_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_314 = l_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_15_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_316 = torch.nn.functional.relu(x_315, inplace=True)
        x_315 = None
        x_317 = torch.conv2d(
            x_316,
            l_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_316 = l_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_318 = torch.nn.functional.batch_norm(
            x_317,
            l_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_317 = l_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_15_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_319 = torch.nn.functional.relu(x_318, inplace=True)
        x_318 = None
        x_320 = torch.conv2d(
            x_319,
            l_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_319 = l_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_321 = torch.nn.functional.batch_norm(
            x_320,
            l_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_320 = l_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_15_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_322 = x_321 + input_34
        x_321 = input_34 = None
        input_35 = torch.nn.functional.relu(x_322, inplace=True)
        x_322 = None
        x_323 = torch.conv2d(
            input_35,
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
        x_324 = torch.nn.functional.batch_norm(
            x_323,
            l_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_323 = l_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_16_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_325 = torch.nn.functional.relu(x_324, inplace=True)
        x_324 = None
        x_326 = torch.conv2d(
            x_325,
            l_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_325 = l_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_327 = torch.nn.functional.batch_norm(
            x_326,
            l_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_326 = l_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_16_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_328 = torch.nn.functional.relu(x_327, inplace=True)
        x_327 = None
        x_329 = torch.conv2d(
            x_328,
            l_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_328 = l_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_330 = torch.nn.functional.batch_norm(
            x_329,
            l_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_329 = l_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_16_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_331 = x_330 + input_35
        x_330 = input_35 = None
        input_36 = torch.nn.functional.relu(x_331, inplace=True)
        x_331 = None
        x_332 = torch.conv2d(
            input_36,
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
        x_333 = torch.nn.functional.batch_norm(
            x_332,
            l_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_332 = l_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_17_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_334 = torch.nn.functional.relu(x_333, inplace=True)
        x_333 = None
        x_335 = torch.conv2d(
            x_334,
            l_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_334 = l_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_336 = torch.nn.functional.batch_norm(
            x_335,
            l_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_335 = l_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_17_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_337 = torch.nn.functional.relu(x_336, inplace=True)
        x_336 = None
        x_338 = torch.conv2d(
            x_337,
            l_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_337 = l_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_339 = torch.nn.functional.batch_norm(
            x_338,
            l_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_338 = l_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_17_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_340 = x_339 + input_36
        x_339 = input_36 = None
        input_37 = torch.nn.functional.relu(x_340, inplace=True)
        x_340 = None
        x_341 = torch.conv2d(
            input_37,
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
        x_342 = torch.nn.functional.batch_norm(
            x_341,
            l_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_341 = l_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_18_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_343 = torch.nn.functional.relu(x_342, inplace=True)
        x_342 = None
        x_344 = torch.conv2d(
            x_343,
            l_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_343 = l_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_345 = torch.nn.functional.batch_norm(
            x_344,
            l_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_344 = l_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_18_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_346 = torch.nn.functional.relu(x_345, inplace=True)
        x_345 = None
        x_347 = torch.conv2d(
            x_346,
            l_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_346 = l_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_348 = torch.nn.functional.batch_norm(
            x_347,
            l_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_347 = l_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_18_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_349 = x_348 + input_37
        x_348 = input_37 = None
        input_38 = torch.nn.functional.relu(x_349, inplace=True)
        x_349 = None
        x_350 = torch.conv2d(
            input_38,
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
        x_351 = torch.nn.functional.batch_norm(
            x_350,
            l_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_350 = l_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_19_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_352 = torch.nn.functional.relu(x_351, inplace=True)
        x_351 = None
        x_353 = torch.conv2d(
            x_352,
            l_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_352 = l_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_354 = torch.nn.functional.batch_norm(
            x_353,
            l_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_353 = l_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_19_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_355 = torch.nn.functional.relu(x_354, inplace=True)
        x_354 = None
        x_356 = torch.conv2d(
            x_355,
            l_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_355 = l_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_357 = torch.nn.functional.batch_norm(
            x_356,
            l_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_356 = l_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_19_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_358 = x_357 + input_38
        x_357 = input_38 = None
        input_39 = torch.nn.functional.relu(x_358, inplace=True)
        x_358 = None
        x_359 = torch.conv2d(
            input_39,
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
        x_360 = torch.nn.functional.batch_norm(
            x_359,
            l_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_359 = l_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_20_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_361 = torch.nn.functional.relu(x_360, inplace=True)
        x_360 = None
        x_362 = torch.conv2d(
            x_361,
            l_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_361 = l_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_363 = torch.nn.functional.batch_norm(
            x_362,
            l_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_362 = l_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_20_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_364 = torch.nn.functional.relu(x_363, inplace=True)
        x_363 = None
        x_365 = torch.conv2d(
            x_364,
            l_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_364 = l_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_366 = torch.nn.functional.batch_norm(
            x_365,
            l_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_365 = l_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_20_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_367 = x_366 + input_39
        x_366 = input_39 = None
        input_40 = torch.nn.functional.relu(x_367, inplace=True)
        x_367 = None
        x_368 = torch.conv2d(
            input_40,
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
        x_369 = torch.nn.functional.batch_norm(
            x_368,
            l_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_368 = l_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_21_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_370 = torch.nn.functional.relu(x_369, inplace=True)
        x_369 = None
        x_371 = torch.conv2d(
            x_370,
            l_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_370 = l_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_372 = torch.nn.functional.batch_norm(
            x_371,
            l_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_371 = l_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_21_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_373 = torch.nn.functional.relu(x_372, inplace=True)
        x_372 = None
        x_374 = torch.conv2d(
            x_373,
            l_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_373 = l_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_375 = torch.nn.functional.batch_norm(
            x_374,
            l_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_374 = l_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_21_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_376 = x_375 + input_40
        x_375 = input_40 = None
        input_41 = torch.nn.functional.relu(x_376, inplace=True)
        x_376 = None
        x_377 = torch.conv2d(
            input_41,
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
        x_378 = torch.nn.functional.batch_norm(
            x_377,
            l_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_377 = l_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_22_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_379 = torch.nn.functional.relu(x_378, inplace=True)
        x_378 = None
        x_380 = torch.conv2d(
            x_379,
            l_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_379 = l_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_381 = torch.nn.functional.batch_norm(
            x_380,
            l_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_380 = l_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_22_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_382 = torch.nn.functional.relu(x_381, inplace=True)
        x_381 = None
        x_383 = torch.conv2d(
            x_382,
            l_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_382 = l_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_384 = torch.nn.functional.batch_norm(
            x_383,
            l_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_383 = l_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_22_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_385 = x_384 + input_41
        x_384 = input_41 = None
        input_42 = torch.nn.functional.relu(x_385, inplace=True)
        x_385 = None
        x_386 = torch.conv2d(
            input_42,
            l_self_modules_stages_modules_2_modules_23_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_23_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_387 = torch.nn.functional.batch_norm(
            x_386,
            l_self_modules_stages_modules_2_modules_23_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_23_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_23_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_23_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_386 = l_self_modules_stages_modules_2_modules_23_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_23_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_23_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_23_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_388 = torch.nn.functional.relu(x_387, inplace=True)
        x_387 = None
        x_389 = torch.conv2d(
            x_388,
            l_self_modules_stages_modules_2_modules_23_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_388 = l_self_modules_stages_modules_2_modules_23_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_390 = torch.nn.functional.batch_norm(
            x_389,
            l_self_modules_stages_modules_2_modules_23_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_23_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_23_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_23_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_389 = l_self_modules_stages_modules_2_modules_23_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_23_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_23_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_23_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_391 = torch.nn.functional.relu(x_390, inplace=True)
        x_390 = None
        x_392 = torch.conv2d(
            x_391,
            l_self_modules_stages_modules_2_modules_23_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_391 = l_self_modules_stages_modules_2_modules_23_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_393 = torch.nn.functional.batch_norm(
            x_392,
            l_self_modules_stages_modules_2_modules_23_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_23_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_23_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_23_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_392 = l_self_modules_stages_modules_2_modules_23_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_23_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_23_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_23_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_394 = x_393 + input_42
        x_393 = input_42 = None
        input_43 = torch.nn.functional.relu(x_394, inplace=True)
        x_394 = None
        x_395 = torch.conv2d(
            input_43,
            l_self_modules_stages_modules_2_modules_24_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_24_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_396 = torch.nn.functional.batch_norm(
            x_395,
            l_self_modules_stages_modules_2_modules_24_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_24_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_24_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_24_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_395 = l_self_modules_stages_modules_2_modules_24_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_24_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_24_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_24_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_397 = torch.nn.functional.relu(x_396, inplace=True)
        x_396 = None
        x_398 = torch.conv2d(
            x_397,
            l_self_modules_stages_modules_2_modules_24_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_397 = l_self_modules_stages_modules_2_modules_24_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_399 = torch.nn.functional.batch_norm(
            x_398,
            l_self_modules_stages_modules_2_modules_24_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_24_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_24_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_24_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_398 = l_self_modules_stages_modules_2_modules_24_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_24_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_24_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_24_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_400 = torch.nn.functional.relu(x_399, inplace=True)
        x_399 = None
        x_401 = torch.conv2d(
            x_400,
            l_self_modules_stages_modules_2_modules_24_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_400 = l_self_modules_stages_modules_2_modules_24_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_402 = torch.nn.functional.batch_norm(
            x_401,
            l_self_modules_stages_modules_2_modules_24_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_24_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_24_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_24_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_401 = l_self_modules_stages_modules_2_modules_24_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_24_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_24_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_24_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_403 = x_402 + input_43
        x_402 = input_43 = None
        input_44 = torch.nn.functional.relu(x_403, inplace=True)
        x_403 = None
        x_404 = torch.conv2d(
            input_44,
            l_self_modules_stages_modules_2_modules_25_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_25_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_405 = torch.nn.functional.batch_norm(
            x_404,
            l_self_modules_stages_modules_2_modules_25_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_25_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_25_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_25_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_404 = l_self_modules_stages_modules_2_modules_25_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_25_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_25_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_25_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_406 = torch.nn.functional.relu(x_405, inplace=True)
        x_405 = None
        x_407 = torch.conv2d(
            x_406,
            l_self_modules_stages_modules_2_modules_25_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_406 = l_self_modules_stages_modules_2_modules_25_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_408 = torch.nn.functional.batch_norm(
            x_407,
            l_self_modules_stages_modules_2_modules_25_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_25_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_25_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_25_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_407 = l_self_modules_stages_modules_2_modules_25_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_25_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_25_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_25_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_409 = torch.nn.functional.relu(x_408, inplace=True)
        x_408 = None
        x_410 = torch.conv2d(
            x_409,
            l_self_modules_stages_modules_2_modules_25_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_409 = l_self_modules_stages_modules_2_modules_25_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_411 = torch.nn.functional.batch_norm(
            x_410,
            l_self_modules_stages_modules_2_modules_25_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_25_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_25_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_25_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_410 = l_self_modules_stages_modules_2_modules_25_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_25_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_25_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_25_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_412 = x_411 + input_44
        x_411 = input_44 = None
        input_45 = torch.nn.functional.relu(x_412, inplace=True)
        x_412 = None
        x_413 = torch.conv2d(
            input_45,
            l_self_modules_stages_modules_2_modules_26_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_26_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_414 = torch.nn.functional.batch_norm(
            x_413,
            l_self_modules_stages_modules_2_modules_26_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_26_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_26_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_26_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_413 = l_self_modules_stages_modules_2_modules_26_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_26_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_26_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_26_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_415 = torch.nn.functional.relu(x_414, inplace=True)
        x_414 = None
        x_416 = torch.conv2d(
            x_415,
            l_self_modules_stages_modules_2_modules_26_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_415 = l_self_modules_stages_modules_2_modules_26_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_417 = torch.nn.functional.batch_norm(
            x_416,
            l_self_modules_stages_modules_2_modules_26_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_26_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_26_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_26_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_416 = l_self_modules_stages_modules_2_modules_26_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_26_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_26_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_26_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_418 = torch.nn.functional.relu(x_417, inplace=True)
        x_417 = None
        x_419 = torch.conv2d(
            x_418,
            l_self_modules_stages_modules_2_modules_26_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_418 = l_self_modules_stages_modules_2_modules_26_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_420 = torch.nn.functional.batch_norm(
            x_419,
            l_self_modules_stages_modules_2_modules_26_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_26_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_26_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_26_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_419 = l_self_modules_stages_modules_2_modules_26_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_26_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_26_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_26_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_421 = x_420 + input_45
        x_420 = input_45 = None
        input_46 = torch.nn.functional.relu(x_421, inplace=True)
        x_421 = None
        x_422 = torch.conv2d(
            input_46,
            l_self_modules_stages_modules_2_modules_27_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_27_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_423 = torch.nn.functional.batch_norm(
            x_422,
            l_self_modules_stages_modules_2_modules_27_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_27_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_27_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_27_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_422 = l_self_modules_stages_modules_2_modules_27_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_27_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_27_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_27_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_424 = torch.nn.functional.relu(x_423, inplace=True)
        x_423 = None
        x_425 = torch.conv2d(
            x_424,
            l_self_modules_stages_modules_2_modules_27_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_424 = l_self_modules_stages_modules_2_modules_27_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_426 = torch.nn.functional.batch_norm(
            x_425,
            l_self_modules_stages_modules_2_modules_27_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_27_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_27_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_27_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_425 = l_self_modules_stages_modules_2_modules_27_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_27_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_27_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_27_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_427 = torch.nn.functional.relu(x_426, inplace=True)
        x_426 = None
        x_428 = torch.conv2d(
            x_427,
            l_self_modules_stages_modules_2_modules_27_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_427 = l_self_modules_stages_modules_2_modules_27_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_429 = torch.nn.functional.batch_norm(
            x_428,
            l_self_modules_stages_modules_2_modules_27_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_27_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_27_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_27_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_428 = l_self_modules_stages_modules_2_modules_27_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_27_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_27_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_27_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_430 = x_429 + input_46
        x_429 = input_46 = None
        input_47 = torch.nn.functional.relu(x_430, inplace=True)
        x_430 = None
        x_431 = torch.conv2d(
            input_47,
            l_self_modules_stages_modules_2_modules_28_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_28_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_432 = torch.nn.functional.batch_norm(
            x_431,
            l_self_modules_stages_modules_2_modules_28_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_28_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_28_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_28_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_431 = l_self_modules_stages_modules_2_modules_28_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_28_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_28_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_28_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_433 = torch.nn.functional.relu(x_432, inplace=True)
        x_432 = None
        x_434 = torch.conv2d(
            x_433,
            l_self_modules_stages_modules_2_modules_28_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_433 = l_self_modules_stages_modules_2_modules_28_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_435 = torch.nn.functional.batch_norm(
            x_434,
            l_self_modules_stages_modules_2_modules_28_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_28_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_28_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_28_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_434 = l_self_modules_stages_modules_2_modules_28_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_28_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_28_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_28_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_436 = torch.nn.functional.relu(x_435, inplace=True)
        x_435 = None
        x_437 = torch.conv2d(
            x_436,
            l_self_modules_stages_modules_2_modules_28_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_436 = l_self_modules_stages_modules_2_modules_28_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_438 = torch.nn.functional.batch_norm(
            x_437,
            l_self_modules_stages_modules_2_modules_28_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_28_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_28_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_28_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_437 = l_self_modules_stages_modules_2_modules_28_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_28_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_28_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_28_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_439 = x_438 + input_47
        x_438 = input_47 = None
        input_48 = torch.nn.functional.relu(x_439, inplace=True)
        x_439 = None
        x_440 = torch.conv2d(
            input_48,
            l_self_modules_stages_modules_2_modules_29_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_29_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_441 = torch.nn.functional.batch_norm(
            x_440,
            l_self_modules_stages_modules_2_modules_29_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_29_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_29_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_29_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_440 = l_self_modules_stages_modules_2_modules_29_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_29_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_29_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_29_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_442 = torch.nn.functional.relu(x_441, inplace=True)
        x_441 = None
        x_443 = torch.conv2d(
            x_442,
            l_self_modules_stages_modules_2_modules_29_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_442 = l_self_modules_stages_modules_2_modules_29_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_444 = torch.nn.functional.batch_norm(
            x_443,
            l_self_modules_stages_modules_2_modules_29_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_29_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_29_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_29_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_443 = l_self_modules_stages_modules_2_modules_29_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_29_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_29_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_29_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_445 = torch.nn.functional.relu(x_444, inplace=True)
        x_444 = None
        x_446 = torch.conv2d(
            x_445,
            l_self_modules_stages_modules_2_modules_29_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_445 = l_self_modules_stages_modules_2_modules_29_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_447 = torch.nn.functional.batch_norm(
            x_446,
            l_self_modules_stages_modules_2_modules_29_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_29_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_29_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_29_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_446 = l_self_modules_stages_modules_2_modules_29_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_29_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_29_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_29_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_448 = x_447 + input_48
        x_447 = input_48 = None
        input_49 = torch.nn.functional.relu(x_448, inplace=True)
        x_448 = None
        x_449 = torch.conv2d(
            input_49,
            l_self_modules_stages_modules_2_modules_30_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_30_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_450 = torch.nn.functional.batch_norm(
            x_449,
            l_self_modules_stages_modules_2_modules_30_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_30_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_30_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_30_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_449 = l_self_modules_stages_modules_2_modules_30_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_30_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_30_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_30_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_451 = torch.nn.functional.relu(x_450, inplace=True)
        x_450 = None
        x_452 = torch.conv2d(
            x_451,
            l_self_modules_stages_modules_2_modules_30_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_451 = l_self_modules_stages_modules_2_modules_30_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_453 = torch.nn.functional.batch_norm(
            x_452,
            l_self_modules_stages_modules_2_modules_30_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_30_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_30_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_30_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_452 = l_self_modules_stages_modules_2_modules_30_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_30_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_30_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_30_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_454 = torch.nn.functional.relu(x_453, inplace=True)
        x_453 = None
        x_455 = torch.conv2d(
            x_454,
            l_self_modules_stages_modules_2_modules_30_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_454 = l_self_modules_stages_modules_2_modules_30_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_456 = torch.nn.functional.batch_norm(
            x_455,
            l_self_modules_stages_modules_2_modules_30_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_30_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_30_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_30_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_455 = l_self_modules_stages_modules_2_modules_30_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_30_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_30_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_30_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_457 = x_456 + input_49
        x_456 = input_49 = None
        input_50 = torch.nn.functional.relu(x_457, inplace=True)
        x_457 = None
        x_458 = torch.conv2d(
            input_50,
            l_self_modules_stages_modules_2_modules_31_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_31_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_459 = torch.nn.functional.batch_norm(
            x_458,
            l_self_modules_stages_modules_2_modules_31_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_31_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_31_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_31_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_458 = l_self_modules_stages_modules_2_modules_31_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_31_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_31_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_31_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_460 = torch.nn.functional.relu(x_459, inplace=True)
        x_459 = None
        x_461 = torch.conv2d(
            x_460,
            l_self_modules_stages_modules_2_modules_31_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_460 = l_self_modules_stages_modules_2_modules_31_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_462 = torch.nn.functional.batch_norm(
            x_461,
            l_self_modules_stages_modules_2_modules_31_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_31_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_31_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_31_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_461 = l_self_modules_stages_modules_2_modules_31_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_31_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_31_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_31_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_463 = torch.nn.functional.relu(x_462, inplace=True)
        x_462 = None
        x_464 = torch.conv2d(
            x_463,
            l_self_modules_stages_modules_2_modules_31_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_463 = l_self_modules_stages_modules_2_modules_31_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_465 = torch.nn.functional.batch_norm(
            x_464,
            l_self_modules_stages_modules_2_modules_31_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_31_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_31_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_31_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_464 = l_self_modules_stages_modules_2_modules_31_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_31_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_31_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_31_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_466 = x_465 + input_50
        x_465 = input_50 = None
        input_51 = torch.nn.functional.relu(x_466, inplace=True)
        x_466 = None
        x_467 = torch.conv2d(
            input_51,
            l_self_modules_stages_modules_2_modules_32_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_32_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_468 = torch.nn.functional.batch_norm(
            x_467,
            l_self_modules_stages_modules_2_modules_32_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_32_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_32_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_32_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_467 = l_self_modules_stages_modules_2_modules_32_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_32_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_32_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_32_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_469 = torch.nn.functional.relu(x_468, inplace=True)
        x_468 = None
        x_470 = torch.conv2d(
            x_469,
            l_self_modules_stages_modules_2_modules_32_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_469 = l_self_modules_stages_modules_2_modules_32_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_471 = torch.nn.functional.batch_norm(
            x_470,
            l_self_modules_stages_modules_2_modules_32_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_32_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_32_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_32_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_470 = l_self_modules_stages_modules_2_modules_32_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_32_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_32_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_32_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_472 = torch.nn.functional.relu(x_471, inplace=True)
        x_471 = None
        x_473 = torch.conv2d(
            x_472,
            l_self_modules_stages_modules_2_modules_32_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_472 = l_self_modules_stages_modules_2_modules_32_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_474 = torch.nn.functional.batch_norm(
            x_473,
            l_self_modules_stages_modules_2_modules_32_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_32_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_32_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_32_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_473 = l_self_modules_stages_modules_2_modules_32_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_32_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_32_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_32_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_475 = x_474 + input_51
        x_474 = input_51 = None
        input_52 = torch.nn.functional.relu(x_475, inplace=True)
        x_475 = None
        x_476 = torch.conv2d(
            input_52,
            l_self_modules_stages_modules_2_modules_33_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_33_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_477 = torch.nn.functional.batch_norm(
            x_476,
            l_self_modules_stages_modules_2_modules_33_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_33_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_33_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_33_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_476 = l_self_modules_stages_modules_2_modules_33_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_33_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_33_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_33_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_478 = torch.nn.functional.relu(x_477, inplace=True)
        x_477 = None
        x_479 = torch.conv2d(
            x_478,
            l_self_modules_stages_modules_2_modules_33_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_478 = l_self_modules_stages_modules_2_modules_33_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_480 = torch.nn.functional.batch_norm(
            x_479,
            l_self_modules_stages_modules_2_modules_33_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_33_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_33_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_33_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_479 = l_self_modules_stages_modules_2_modules_33_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_33_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_33_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_33_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_481 = torch.nn.functional.relu(x_480, inplace=True)
        x_480 = None
        x_482 = torch.conv2d(
            x_481,
            l_self_modules_stages_modules_2_modules_33_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_481 = l_self_modules_stages_modules_2_modules_33_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_483 = torch.nn.functional.batch_norm(
            x_482,
            l_self_modules_stages_modules_2_modules_33_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_33_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_33_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_33_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_482 = l_self_modules_stages_modules_2_modules_33_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_33_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_33_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_33_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_484 = x_483 + input_52
        x_483 = input_52 = None
        input_53 = torch.nn.functional.relu(x_484, inplace=True)
        x_484 = None
        x_485 = torch.conv2d(
            input_53,
            l_self_modules_stages_modules_2_modules_34_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_34_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_486 = torch.nn.functional.batch_norm(
            x_485,
            l_self_modules_stages_modules_2_modules_34_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_34_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_34_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_34_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_485 = l_self_modules_stages_modules_2_modules_34_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_34_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_34_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_34_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_487 = torch.nn.functional.relu(x_486, inplace=True)
        x_486 = None
        x_488 = torch.conv2d(
            x_487,
            l_self_modules_stages_modules_2_modules_34_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_487 = l_self_modules_stages_modules_2_modules_34_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_489 = torch.nn.functional.batch_norm(
            x_488,
            l_self_modules_stages_modules_2_modules_34_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_34_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_34_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_34_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_488 = l_self_modules_stages_modules_2_modules_34_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_34_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_34_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_34_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_490 = torch.nn.functional.relu(x_489, inplace=True)
        x_489 = None
        x_491 = torch.conv2d(
            x_490,
            l_self_modules_stages_modules_2_modules_34_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_490 = l_self_modules_stages_modules_2_modules_34_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_492 = torch.nn.functional.batch_norm(
            x_491,
            l_self_modules_stages_modules_2_modules_34_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_34_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_34_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_34_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_491 = l_self_modules_stages_modules_2_modules_34_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_34_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_34_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_34_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_493 = x_492 + input_53
        x_492 = input_53 = None
        input_54 = torch.nn.functional.relu(x_493, inplace=True)
        x_493 = None
        x_494 = torch.conv2d(
            input_54,
            l_self_modules_stages_modules_2_modules_35_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_35_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_495 = torch.nn.functional.batch_norm(
            x_494,
            l_self_modules_stages_modules_2_modules_35_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_35_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_35_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_35_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_494 = l_self_modules_stages_modules_2_modules_35_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_35_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_35_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_35_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_496 = torch.nn.functional.relu(x_495, inplace=True)
        x_495 = None
        x_497 = torch.conv2d(
            x_496,
            l_self_modules_stages_modules_2_modules_35_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_496 = l_self_modules_stages_modules_2_modules_35_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_498 = torch.nn.functional.batch_norm(
            x_497,
            l_self_modules_stages_modules_2_modules_35_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_35_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_35_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_35_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_497 = l_self_modules_stages_modules_2_modules_35_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_35_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_35_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_35_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_499 = torch.nn.functional.relu(x_498, inplace=True)
        x_498 = None
        x_500 = torch.conv2d(
            x_499,
            l_self_modules_stages_modules_2_modules_35_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_499 = l_self_modules_stages_modules_2_modules_35_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_501 = torch.nn.functional.batch_norm(
            x_500,
            l_self_modules_stages_modules_2_modules_35_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_35_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_35_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_35_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_500 = l_self_modules_stages_modules_2_modules_35_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_35_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_35_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_35_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_502 = x_501 + input_54
        x_501 = input_54 = None
        input_55 = torch.nn.functional.relu(x_502, inplace=True)
        x_502 = None
        x_503 = torch.conv2d(
            input_55,
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
        x_504 = torch.nn.functional.batch_norm(
            x_503,
            l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_503 = l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_505 = torch.nn.functional.relu(x_504, inplace=True)
        x_504 = None
        x_506 = torch.conv2d(
            x_505,
            l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_505 = l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_507 = torch.nn.functional.batch_norm(
            x_506,
            l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_506 = l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_508 = torch.nn.functional.relu(x_507, inplace=True)
        x_507 = None
        x_509 = torch._C._nn.avg_pool2d(x_508, 2, 2, 0, False, True, None)
        x_508 = None
        x_510 = torch.conv2d(
            x_509,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_509 = l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_511 = torch.nn.functional.batch_norm(
            x_510,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_510 = l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        avg_pool2d_6 = torch._C._nn.avg_pool2d(input_55, 2, 2, 0, True, False, None)
        input_55 = None
        x_512 = torch.conv2d(
            avg_pool2d_6,
            l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        avg_pool2d_6 = l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_conv_parameters_weight_ = (None)
        x_513 = torch.nn.functional.batch_norm(
            x_512,
            l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_512 = l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_modules_bn_parameters_bias_ = (None)
        x_514 = x_511 + x_513
        x_511 = x_513 = None
        input_56 = torch.nn.functional.relu(x_514, inplace=True)
        x_514 = None
        x_515 = torch.conv2d(
            input_56,
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
        x_516 = torch.nn.functional.batch_norm(
            x_515,
            l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_515 = l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_517 = torch.nn.functional.relu(x_516, inplace=True)
        x_516 = None
        x_518 = torch.conv2d(
            x_517,
            l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_517 = l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_519 = torch.nn.functional.batch_norm(
            x_518,
            l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_518 = l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_520 = torch.nn.functional.relu(x_519, inplace=True)
        x_519 = None
        x_521 = torch.conv2d(
            x_520,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_520 = l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_522 = torch.nn.functional.batch_norm(
            x_521,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_521 = l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_523 = x_522 + input_56
        x_522 = input_56 = None
        input_57 = torch.nn.functional.relu(x_523, inplace=True)
        x_523 = None
        x_524 = torch.conv2d(
            input_57,
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
        x_525 = torch.nn.functional.batch_norm(
            x_524,
            l_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_524 = l_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_2_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_526 = torch.nn.functional.relu(x_525, inplace=True)
        x_525 = None
        x_527 = torch.conv2d(
            x_526,
            l_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_526 = l_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_528 = torch.nn.functional.batch_norm(
            x_527,
            l_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_527 = l_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_2_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_529 = torch.nn.functional.relu(x_528, inplace=True)
        x_528 = None
        x_530 = torch.conv2d(
            x_529,
            l_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_529 = l_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_531 = torch.nn.functional.batch_norm(
            x_530,
            l_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_530 = l_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_2_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_532 = x_531 + input_57
        x_531 = input_57 = None
        input_58 = torch.nn.functional.relu(x_532, inplace=True)
        x_532 = None
        x_533 = torch.conv2d(
            input_58,
            l_self_modules_stages_modules_3_modules_3_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_3_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_534 = torch.nn.functional.batch_norm(
            x_533,
            l_self_modules_stages_modules_3_modules_3_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_3_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_3_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_3_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_533 = l_self_modules_stages_modules_3_modules_3_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_3_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_3_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_3_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_535 = torch.nn.functional.relu(x_534, inplace=True)
        x_534 = None
        x_536 = torch.conv2d(
            x_535,
            l_self_modules_stages_modules_3_modules_3_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_535 = l_self_modules_stages_modules_3_modules_3_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_537 = torch.nn.functional.batch_norm(
            x_536,
            l_self_modules_stages_modules_3_modules_3_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_3_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_3_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_3_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_536 = l_self_modules_stages_modules_3_modules_3_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_3_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_3_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_3_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_538 = torch.nn.functional.relu(x_537, inplace=True)
        x_537 = None
        x_539 = torch.conv2d(
            x_538,
            l_self_modules_stages_modules_3_modules_3_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_538 = l_self_modules_stages_modules_3_modules_3_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_540 = torch.nn.functional.batch_norm(
            x_539,
            l_self_modules_stages_modules_3_modules_3_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_3_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_3_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_3_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_539 = l_self_modules_stages_modules_3_modules_3_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_3_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_3_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_3_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_541 = x_540 + input_58
        x_540 = input_58 = None
        input_59 = torch.nn.functional.relu(x_541, inplace=True)
        x_541 = None
        x_542 = torch.conv2d(
            input_59,
            l_self_modules_stages_modules_3_modules_4_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_4_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_543 = torch.nn.functional.batch_norm(
            x_542,
            l_self_modules_stages_modules_3_modules_4_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_4_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_4_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_4_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_542 = l_self_modules_stages_modules_3_modules_4_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_4_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_4_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_4_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_544 = torch.nn.functional.relu(x_543, inplace=True)
        x_543 = None
        x_545 = torch.conv2d(
            x_544,
            l_self_modules_stages_modules_3_modules_4_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_544 = l_self_modules_stages_modules_3_modules_4_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_546 = torch.nn.functional.batch_norm(
            x_545,
            l_self_modules_stages_modules_3_modules_4_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_4_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_4_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_4_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_545 = l_self_modules_stages_modules_3_modules_4_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_4_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_4_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_4_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_547 = torch.nn.functional.relu(x_546, inplace=True)
        x_546 = None
        x_548 = torch.conv2d(
            x_547,
            l_self_modules_stages_modules_3_modules_4_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_547 = l_self_modules_stages_modules_3_modules_4_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_549 = torch.nn.functional.batch_norm(
            x_548,
            l_self_modules_stages_modules_3_modules_4_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_4_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_4_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_4_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_548 = l_self_modules_stages_modules_3_modules_4_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_4_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_4_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_4_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_550 = x_549 + input_59
        x_549 = input_59 = None
        input_60 = torch.nn.functional.relu(x_550, inplace=True)
        x_550 = None
        x_551 = torch.conv2d(
            input_60,
            l_self_modules_stages_modules_3_modules_5_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_5_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_552 = torch.nn.functional.batch_norm(
            x_551,
            l_self_modules_stages_modules_3_modules_5_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_5_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_5_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_5_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_551 = l_self_modules_stages_modules_3_modules_5_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_5_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_5_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_5_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_553 = torch.nn.functional.relu(x_552, inplace=True)
        x_552 = None
        x_554 = torch.conv2d(
            x_553,
            l_self_modules_stages_modules_3_modules_5_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_553 = l_self_modules_stages_modules_3_modules_5_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_555 = torch.nn.functional.batch_norm(
            x_554,
            l_self_modules_stages_modules_3_modules_5_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_5_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_5_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_5_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_554 = l_self_modules_stages_modules_3_modules_5_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_5_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_5_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_5_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_556 = torch.nn.functional.relu(x_555, inplace=True)
        x_555 = None
        x_557 = torch.conv2d(
            x_556,
            l_self_modules_stages_modules_3_modules_5_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_556 = l_self_modules_stages_modules_3_modules_5_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_558 = torch.nn.functional.batch_norm(
            x_557,
            l_self_modules_stages_modules_3_modules_5_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_5_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_5_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_5_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_557 = l_self_modules_stages_modules_3_modules_5_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_5_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_5_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_5_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_559 = x_558 + input_60
        x_558 = input_60 = None
        input_61 = torch.nn.functional.relu(x_559, inplace=True)
        x_559 = None
        x_560 = torch.conv2d(
            input_61,
            l_self_modules_stages_modules_3_modules_6_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_6_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_561 = torch.nn.functional.batch_norm(
            x_560,
            l_self_modules_stages_modules_3_modules_6_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_6_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_6_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_6_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_560 = l_self_modules_stages_modules_3_modules_6_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_6_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_6_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_6_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_562 = torch.nn.functional.relu(x_561, inplace=True)
        x_561 = None
        x_563 = torch.conv2d(
            x_562,
            l_self_modules_stages_modules_3_modules_6_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_562 = l_self_modules_stages_modules_3_modules_6_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_564 = torch.nn.functional.batch_norm(
            x_563,
            l_self_modules_stages_modules_3_modules_6_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_6_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_6_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_6_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_563 = l_self_modules_stages_modules_3_modules_6_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_6_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_6_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_6_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_565 = torch.nn.functional.relu(x_564, inplace=True)
        x_564 = None
        x_566 = torch.conv2d(
            x_565,
            l_self_modules_stages_modules_3_modules_6_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_565 = l_self_modules_stages_modules_3_modules_6_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_567 = torch.nn.functional.batch_norm(
            x_566,
            l_self_modules_stages_modules_3_modules_6_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_6_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_6_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_6_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_566 = l_self_modules_stages_modules_3_modules_6_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_6_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_6_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_6_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_568 = x_567 + input_61
        x_567 = input_61 = None
        input_62 = torch.nn.functional.relu(x_568, inplace=True)
        x_568 = None
        x_569 = torch.conv2d(
            input_62,
            l_self_modules_stages_modules_3_modules_7_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_7_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_570 = torch.nn.functional.batch_norm(
            x_569,
            l_self_modules_stages_modules_3_modules_7_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_7_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_7_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_7_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_569 = l_self_modules_stages_modules_3_modules_7_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_7_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_7_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_7_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_571 = torch.nn.functional.relu(x_570, inplace=True)
        x_570 = None
        x_572 = torch.conv2d(
            x_571,
            l_self_modules_stages_modules_3_modules_7_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_571 = l_self_modules_stages_modules_3_modules_7_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_573 = torch.nn.functional.batch_norm(
            x_572,
            l_self_modules_stages_modules_3_modules_7_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_7_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_7_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_7_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_572 = l_self_modules_stages_modules_3_modules_7_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_7_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_7_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_7_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_574 = torch.nn.functional.relu(x_573, inplace=True)
        x_573 = None
        x_575 = torch.conv2d(
            x_574,
            l_self_modules_stages_modules_3_modules_7_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_574 = l_self_modules_stages_modules_3_modules_7_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_576 = torch.nn.functional.batch_norm(
            x_575,
            l_self_modules_stages_modules_3_modules_7_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_7_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_7_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_7_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_575 = l_self_modules_stages_modules_3_modules_7_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_7_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_7_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_7_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_577 = x_576 + input_62
        x_576 = input_62 = None
        input_63 = torch.nn.functional.relu(x_577, inplace=True)
        x_577 = None
        x_578 = torch.conv2d(
            input_63,
            l_self_modules_stages_modules_3_modules_8_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_8_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_579 = torch.nn.functional.batch_norm(
            x_578,
            l_self_modules_stages_modules_3_modules_8_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_8_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_8_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_8_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_578 = l_self_modules_stages_modules_3_modules_8_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_8_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_8_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_8_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_580 = torch.nn.functional.relu(x_579, inplace=True)
        x_579 = None
        x_581 = torch.conv2d(
            x_580,
            l_self_modules_stages_modules_3_modules_8_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_580 = l_self_modules_stages_modules_3_modules_8_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_582 = torch.nn.functional.batch_norm(
            x_581,
            l_self_modules_stages_modules_3_modules_8_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_8_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_8_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_8_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_581 = l_self_modules_stages_modules_3_modules_8_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_8_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_8_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_8_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_583 = torch.nn.functional.relu(x_582, inplace=True)
        x_582 = None
        x_584 = torch.conv2d(
            x_583,
            l_self_modules_stages_modules_3_modules_8_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_583 = l_self_modules_stages_modules_3_modules_8_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_585 = torch.nn.functional.batch_norm(
            x_584,
            l_self_modules_stages_modules_3_modules_8_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_8_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_8_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_8_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_584 = l_self_modules_stages_modules_3_modules_8_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_8_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_8_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_8_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_586 = x_585 + input_63
        x_585 = input_63 = None
        input_64 = torch.nn.functional.relu(x_586, inplace=True)
        x_586 = None
        x_587 = torch.conv2d(
            input_64,
            l_self_modules_stages_modules_3_modules_9_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_9_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_588 = torch.nn.functional.batch_norm(
            x_587,
            l_self_modules_stages_modules_3_modules_9_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_9_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_9_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_9_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_587 = l_self_modules_stages_modules_3_modules_9_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_9_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_9_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_9_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_589 = torch.nn.functional.relu(x_588, inplace=True)
        x_588 = None
        x_590 = torch.conv2d(
            x_589,
            l_self_modules_stages_modules_3_modules_9_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_589 = l_self_modules_stages_modules_3_modules_9_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_591 = torch.nn.functional.batch_norm(
            x_590,
            l_self_modules_stages_modules_3_modules_9_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_9_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_9_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_9_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_590 = l_self_modules_stages_modules_3_modules_9_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_9_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_9_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_9_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_592 = torch.nn.functional.relu(x_591, inplace=True)
        x_591 = None
        x_593 = torch.conv2d(
            x_592,
            l_self_modules_stages_modules_3_modules_9_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_592 = l_self_modules_stages_modules_3_modules_9_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_594 = torch.nn.functional.batch_norm(
            x_593,
            l_self_modules_stages_modules_3_modules_9_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_9_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_9_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_9_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_593 = l_self_modules_stages_modules_3_modules_9_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_9_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_9_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_9_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_595 = x_594 + input_64
        x_594 = input_64 = None
        input_65 = torch.nn.functional.relu(x_595, inplace=True)
        x_595 = None
        x_596 = torch.nn.functional.adaptive_avg_pool2d(input_65, 1)
        input_65 = None
        x_597 = x_596.flatten(1, -1)
        x_596 = None
        x_598 = torch.nn.functional.dropout(x_597, 0.0, False, False)
        x_597 = None
        return (x_598,)
