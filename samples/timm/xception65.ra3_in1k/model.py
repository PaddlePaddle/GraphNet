import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_shortcut_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_shortcut_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_shortcut_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_shortcut_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_shortcut_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_shortcut_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_shortcut_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_shortcut_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_shortcut_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_shortcut_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_shortcut_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_shortcut_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_shortcut_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_shortcut_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_shortcut_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_shortcut_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_shortcut_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_19_modules_shortcut_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_19_modules_shortcut_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_shortcut_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_0_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_0_modules_conv_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_stem_modules_0_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_0_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_0_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_0_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_0_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_0_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_stem_modules_1_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_1_modules_conv_parameters_weight_
        )
        l_self_modules_stem_modules_1_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_1_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_1_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_1_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_1_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_1_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_1_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_1_modules_bn_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_0_modules_shortcut_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_0_modules_shortcut_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_0_modules_shortcut_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_0_modules_shortcut_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_0_modules_shortcut_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_0_modules_shortcut_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_0_modules_shortcut_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_0_modules_shortcut_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_0_modules_shortcut_modules_bn_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_shortcut_modules_bn_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_1_modules_shortcut_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_1_modules_shortcut_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_1_modules_shortcut_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_shortcut_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_shortcut_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_shortcut_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_shortcut_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_1_modules_shortcut_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_1_modules_shortcut_modules_bn_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_shortcut_modules_bn_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_2_modules_shortcut_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_shortcut_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_shortcut_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_shortcut_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_shortcut_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_shortcut_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_shortcut_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_shortcut_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_shortcut_modules_bn_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_shortcut_modules_bn_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_19_modules_shortcut_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_19_modules_shortcut_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_19_modules_shortcut_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_19_modules_shortcut_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_19_modules_shortcut_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_19_modules_shortcut_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_19_modules_shortcut_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_19_modules_shortcut_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_19_modules_shortcut_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_19_modules_shortcut_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_
        l_self_modules_head_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_fc_parameters_bias_
        )
        x = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_stem_modules_0_modules_conv_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_stem_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x = (
            l_self_modules_stem_modules_0_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_0_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_0_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_0_modules_bn_parameters_bias_ = None
        x_2 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_stem_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_stem_modules_1_modules_conv_parameters_weight_ = None
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_stem_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_3 = (
            l_self_modules_stem_modules_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_1_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_1_modules_bn_parameters_bias_ = None
        x_5 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        input_1 = torch.nn.functional.relu(x_5, inplace=False)
        x_6 = torch.conv2d(
            input_1,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
        input_1 = l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (None)
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_6 = l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = (None)
        x_8 = torch.conv2d(
            x_7,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_7 = l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_9 = torch.nn.functional.batch_norm(
            x_8,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_8 = l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = (None)
        input_2 = torch.nn.functional.relu(x_9, inplace=True)
        x_9 = None
        x_10 = torch.conv2d(
            input_2,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        input_2 = l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_11 = torch.nn.functional.batch_norm(
            x_10,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_10 = l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = (None)
        x_12 = torch.conv2d(
            x_11,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_11 = l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_13 = torch.nn.functional.batch_norm(
            x_12,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_12 = l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = (None)
        input_3 = torch.nn.functional.relu(x_13, inplace=True)
        x_13 = None
        x_14 = torch.conv2d(
            input_3,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            128,
        )
        input_3 = l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_15 = torch.nn.functional.batch_norm(
            x_14,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_14 = l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = (None)
        x_16 = torch.conv2d(
            x_15,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_15 = l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_16 = l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = (None)
        x_18 = torch.conv2d(
            x_5,
            l_self_modules_blocks_modules_0_modules_shortcut_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_5 = l_self_modules_blocks_modules_0_modules_shortcut_modules_conv_parameters_weight_ = (None)
        x_19 = torch.nn.functional.batch_norm(
            x_18,
            l_self_modules_blocks_modules_0_modules_shortcut_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_shortcut_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_shortcut_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_shortcut_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_18 = l_self_modules_blocks_modules_0_modules_shortcut_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_0_modules_shortcut_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_0_modules_shortcut_modules_bn_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_shortcut_modules_bn_parameters_bias_
        ) = None
        x_20 = x_17 + x_19
        x_17 = x_19 = None
        input_4 = torch.nn.functional.relu(x_20, inplace=False)
        x_21 = torch.conv2d(
            input_4,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        input_4 = l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (None)
        x_22 = torch.nn.functional.batch_norm(
            x_21,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_21 = l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = (None)
        x_23 = torch.conv2d(
            x_22,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_22 = l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_24 = torch.nn.functional.batch_norm(
            x_23,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_23 = l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = (None)
        input_5 = torch.nn.functional.relu(x_24, inplace=True)
        x_24 = None
        x_25 = torch.conv2d(
            input_5,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        input_5 = l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_26 = torch.nn.functional.batch_norm(
            x_25,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_25 = l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = (None)
        x_27 = torch.conv2d(
            x_26,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_26 = l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_28 = torch.nn.functional.batch_norm(
            x_27,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_27 = l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = (None)
        input_6 = torch.nn.functional.relu(x_28, inplace=True)
        x_28 = None
        x_29 = torch.conv2d(
            input_6,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            256,
        )
        input_6 = l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_30 = torch.nn.functional.batch_norm(
            x_29,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_29 = l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = (None)
        x_31 = torch.conv2d(
            x_30,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_30 = l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_32 = torch.nn.functional.batch_norm(
            x_31,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_31 = l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = (None)
        x_33 = torch.conv2d(
            x_20,
            l_self_modules_blocks_modules_1_modules_shortcut_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_20 = l_self_modules_blocks_modules_1_modules_shortcut_modules_conv_parameters_weight_ = (None)
        x_34 = torch.nn.functional.batch_norm(
            x_33,
            l_self_modules_blocks_modules_1_modules_shortcut_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_shortcut_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_shortcut_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_shortcut_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_33 = l_self_modules_blocks_modules_1_modules_shortcut_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_shortcut_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_shortcut_modules_bn_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_shortcut_modules_bn_parameters_bias_
        ) = None
        x_35 = x_32 + x_34
        x_32 = x_34 = None
        input_7 = torch.nn.functional.relu(x_35, inplace=False)
        x_36 = torch.conv2d(
            input_7,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        input_7 = l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (None)
        x_37 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_36 = l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = (None)
        x_38 = torch.conv2d(
            x_37,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_37 = l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_39 = torch.nn.functional.batch_norm(
            x_38,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_38 = l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = (None)
        input_8 = torch.nn.functional.relu(x_39, inplace=True)
        x_39 = None
        x_40 = torch.conv2d(
            input_8,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_8 = l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_40 = l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = (None)
        x_42 = torch.conv2d(
            x_41,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_41 = l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_43 = torch.nn.functional.batch_norm(
            x_42,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_42 = l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = (None)
        input_9 = torch.nn.functional.relu(x_43, inplace=True)
        x_43 = None
        x_44 = torch.conv2d(
            input_9,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            728,
        )
        input_9 = l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_45 = torch.nn.functional.batch_norm(
            x_44,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_44 = l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = (None)
        x_46 = torch.conv2d(
            x_45,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_45 = l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_47 = torch.nn.functional.batch_norm(
            x_46,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_46 = l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = (None)
        x_48 = torch.conv2d(
            x_35,
            l_self_modules_blocks_modules_2_modules_shortcut_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_35 = l_self_modules_blocks_modules_2_modules_shortcut_modules_conv_parameters_weight_ = (None)
        x_49 = torch.nn.functional.batch_norm(
            x_48,
            l_self_modules_blocks_modules_2_modules_shortcut_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_shortcut_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_shortcut_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_shortcut_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_48 = l_self_modules_blocks_modules_2_modules_shortcut_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_shortcut_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_shortcut_modules_bn_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_shortcut_modules_bn_parameters_bias_
        ) = None
        x_50 = x_47 + x_49
        x_47 = x_49 = None
        input_10 = torch.nn.functional.relu(x_50, inplace=False)
        x_51 = torch.conv2d(
            input_10,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_10 = l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (None)
        x_52 = torch.nn.functional.batch_norm(
            x_51,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_51 = l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = (None)
        x_53 = torch.conv2d(
            x_52,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_52 = l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_54 = torch.nn.functional.batch_norm(
            x_53,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_53 = l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = (None)
        input_11 = torch.nn.functional.relu(x_54, inplace=True)
        x_54 = None
        x_55 = torch.conv2d(
            input_11,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_11 = l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_56 = torch.nn.functional.batch_norm(
            x_55,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_55 = l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = (None)
        x_57 = torch.conv2d(
            x_56,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_56 = l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_58 = torch.nn.functional.batch_norm(
            x_57,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_57 = l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = (None)
        input_12 = torch.nn.functional.relu(x_58, inplace=True)
        x_58 = None
        x_59 = torch.conv2d(
            input_12,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_12 = l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_60 = torch.nn.functional.batch_norm(
            x_59,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_59 = l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = (None)
        x_61 = torch.conv2d(
            x_60,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_60 = l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_62 = torch.nn.functional.batch_norm(
            x_61,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_61 = l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = (None)
        x_63 = x_62 + x_50
        x_62 = x_50 = None
        input_13 = torch.nn.functional.relu(x_63, inplace=False)
        x_64 = torch.conv2d(
            input_13,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_13 = l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (None)
        x_65 = torch.nn.functional.batch_norm(
            x_64,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_64 = l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = (None)
        x_66 = torch.conv2d(
            x_65,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_65 = l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_67 = torch.nn.functional.batch_norm(
            x_66,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_66 = l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = (None)
        input_14 = torch.nn.functional.relu(x_67, inplace=True)
        x_67 = None
        x_68 = torch.conv2d(
            input_14,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_14 = l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_69 = torch.nn.functional.batch_norm(
            x_68,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_68 = l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = (None)
        x_70 = torch.conv2d(
            x_69,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_69 = l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_71 = torch.nn.functional.batch_norm(
            x_70,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_70 = l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = (None)
        input_15 = torch.nn.functional.relu(x_71, inplace=True)
        x_71 = None
        x_72 = torch.conv2d(
            input_15,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_15 = l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_72 = l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = (None)
        x_74 = torch.conv2d(
            x_73,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_73 = l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_75 = torch.nn.functional.batch_norm(
            x_74,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_74 = l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = (None)
        x_76 = x_75 + x_63
        x_75 = x_63 = None
        input_16 = torch.nn.functional.relu(x_76, inplace=False)
        x_77 = torch.conv2d(
            input_16,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_16 = l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (None)
        x_78 = torch.nn.functional.batch_norm(
            x_77,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_77 = l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = (None)
        x_79 = torch.conv2d(
            x_78,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_78 = l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_80 = torch.nn.functional.batch_norm(
            x_79,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_79 = l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = (None)
        input_17 = torch.nn.functional.relu(x_80, inplace=True)
        x_80 = None
        x_81 = torch.conv2d(
            input_17,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_17 = l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_82 = torch.nn.functional.batch_norm(
            x_81,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_81 = l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = (None)
        x_83 = torch.conv2d(
            x_82,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_82 = l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_84 = torch.nn.functional.batch_norm(
            x_83,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_83 = l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = (None)
        input_18 = torch.nn.functional.relu(x_84, inplace=True)
        x_84 = None
        x_85 = torch.conv2d(
            input_18,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_18 = l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_86 = torch.nn.functional.batch_norm(
            x_85,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_85 = l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = (None)
        x_87 = torch.conv2d(
            x_86,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_86 = l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_88 = torch.nn.functional.batch_norm(
            x_87,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_87 = l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = (None)
        x_89 = x_88 + x_76
        x_88 = x_76 = None
        input_19 = torch.nn.functional.relu(x_89, inplace=False)
        x_90 = torch.conv2d(
            input_19,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_19 = l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (None)
        x_91 = torch.nn.functional.batch_norm(
            x_90,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_90 = l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = (None)
        x_92 = torch.conv2d(
            x_91,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_91 = l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_93 = torch.nn.functional.batch_norm(
            x_92,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_92 = l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = (None)
        input_20 = torch.nn.functional.relu(x_93, inplace=True)
        x_93 = None
        x_94 = torch.conv2d(
            input_20,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_20 = l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_95 = torch.nn.functional.batch_norm(
            x_94,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_94 = l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = (None)
        x_96 = torch.conv2d(
            x_95,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_95 = l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_97 = torch.nn.functional.batch_norm(
            x_96,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_96 = l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = (None)
        input_21 = torch.nn.functional.relu(x_97, inplace=True)
        x_97 = None
        x_98 = torch.conv2d(
            input_21,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_21 = l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_99 = torch.nn.functional.batch_norm(
            x_98,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_98 = l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = (None)
        x_100 = torch.conv2d(
            x_99,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_99 = l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_101 = torch.nn.functional.batch_norm(
            x_100,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_100 = l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = (None)
        x_102 = x_101 + x_89
        x_101 = x_89 = None
        input_22 = torch.nn.functional.relu(x_102, inplace=False)
        x_103 = torch.conv2d(
            input_22,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_22 = l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (None)
        x_104 = torch.nn.functional.batch_norm(
            x_103,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_103 = l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = (None)
        x_105 = torch.conv2d(
            x_104,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_104 = l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_106 = torch.nn.functional.batch_norm(
            x_105,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_105 = l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = (None)
        input_23 = torch.nn.functional.relu(x_106, inplace=True)
        x_106 = None
        x_107 = torch.conv2d(
            input_23,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_23 = l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_108 = torch.nn.functional.batch_norm(
            x_107,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_107 = l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = (None)
        x_109 = torch.conv2d(
            x_108,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_108 = l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_110 = torch.nn.functional.batch_norm(
            x_109,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_109 = l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = (None)
        input_24 = torch.nn.functional.relu(x_110, inplace=True)
        x_110 = None
        x_111 = torch.conv2d(
            input_24,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_24 = l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_112 = torch.nn.functional.batch_norm(
            x_111,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_111 = l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = (None)
        x_113 = torch.conv2d(
            x_112,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_112 = l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_114 = torch.nn.functional.batch_norm(
            x_113,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_113 = l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = (None)
        x_115 = x_114 + x_102
        x_114 = x_102 = None
        input_25 = torch.nn.functional.relu(x_115, inplace=False)
        x_116 = torch.conv2d(
            input_25,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_25 = l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (None)
        x_117 = torch.nn.functional.batch_norm(
            x_116,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_116 = l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = (None)
        x_118 = torch.conv2d(
            x_117,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_117 = l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_119 = torch.nn.functional.batch_norm(
            x_118,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_118 = l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = (None)
        input_26 = torch.nn.functional.relu(x_119, inplace=True)
        x_119 = None
        x_120 = torch.conv2d(
            input_26,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_26 = l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_121 = torch.nn.functional.batch_norm(
            x_120,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_120 = l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = (None)
        x_122 = torch.conv2d(
            x_121,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_121 = l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_123 = torch.nn.functional.batch_norm(
            x_122,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_122 = l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = (None)
        input_27 = torch.nn.functional.relu(x_123, inplace=True)
        x_123 = None
        x_124 = torch.conv2d(
            input_27,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_27 = l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_125 = torch.nn.functional.batch_norm(
            x_124,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_124 = l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = (None)
        x_126 = torch.conv2d(
            x_125,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_125 = l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_127 = torch.nn.functional.batch_norm(
            x_126,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_126 = l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = (None)
        x_128 = x_127 + x_115
        x_127 = x_115 = None
        input_28 = torch.nn.functional.relu(x_128, inplace=False)
        x_129 = torch.conv2d(
            input_28,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_28 = l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (None)
        x_130 = torch.nn.functional.batch_norm(
            x_129,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_129 = l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = (None)
        x_131 = torch.conv2d(
            x_130,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_130 = l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_132 = torch.nn.functional.batch_norm(
            x_131,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_131 = l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = (None)
        input_29 = torch.nn.functional.relu(x_132, inplace=True)
        x_132 = None
        x_133 = torch.conv2d(
            input_29,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_29 = l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_134 = torch.nn.functional.batch_norm(
            x_133,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_133 = l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = (None)
        x_135 = torch.conv2d(
            x_134,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_134 = l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_136 = torch.nn.functional.batch_norm(
            x_135,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_135 = l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = (None)
        input_30 = torch.nn.functional.relu(x_136, inplace=True)
        x_136 = None
        x_137 = torch.conv2d(
            input_30,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_30 = l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_138 = torch.nn.functional.batch_norm(
            x_137,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_137 = l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = (None)
        x_139 = torch.conv2d(
            x_138,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_138 = l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_140 = torch.nn.functional.batch_norm(
            x_139,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_139 = l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = (None)
        x_141 = x_140 + x_128
        x_140 = x_128 = None
        input_31 = torch.nn.functional.relu(x_141, inplace=False)
        x_142 = torch.conv2d(
            input_31,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_31 = l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (None)
        x_143 = torch.nn.functional.batch_norm(
            x_142,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_142 = l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = (None)
        x_144 = torch.conv2d(
            x_143,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_143 = l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_145 = torch.nn.functional.batch_norm(
            x_144,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_144 = l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = (None)
        input_32 = torch.nn.functional.relu(x_145, inplace=True)
        x_145 = None
        x_146 = torch.conv2d(
            input_32,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_32 = l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_147 = torch.nn.functional.batch_norm(
            x_146,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_146 = l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = (None)
        x_148 = torch.conv2d(
            x_147,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_147 = l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_149 = torch.nn.functional.batch_norm(
            x_148,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_148 = l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = (None)
        input_33 = torch.nn.functional.relu(x_149, inplace=True)
        x_149 = None
        x_150 = torch.conv2d(
            input_33,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_33 = l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_151 = torch.nn.functional.batch_norm(
            x_150,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_150 = l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = (None)
        x_152 = torch.conv2d(
            x_151,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_151 = l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_153 = torch.nn.functional.batch_norm(
            x_152,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_152 = l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = (None)
        x_154 = x_153 + x_141
        x_153 = x_141 = None
        input_34 = torch.nn.functional.relu(x_154, inplace=False)
        x_155 = torch.conv2d(
            input_34,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_34 = l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (None)
        x_156 = torch.nn.functional.batch_norm(
            x_155,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_155 = l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = (None)
        x_157 = torch.conv2d(
            x_156,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_156 = l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_158 = torch.nn.functional.batch_norm(
            x_157,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_157 = l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = (None)
        input_35 = torch.nn.functional.relu(x_158, inplace=True)
        x_158 = None
        x_159 = torch.conv2d(
            input_35,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_35 = l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_160 = torch.nn.functional.batch_norm(
            x_159,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_159 = l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = (None)
        x_161 = torch.conv2d(
            x_160,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_160 = l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_162 = torch.nn.functional.batch_norm(
            x_161,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_161 = l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = (None)
        input_36 = torch.nn.functional.relu(x_162, inplace=True)
        x_162 = None
        x_163 = torch.conv2d(
            input_36,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_36 = l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_164 = torch.nn.functional.batch_norm(
            x_163,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_163 = l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = (None)
        x_165 = torch.conv2d(
            x_164,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_164 = l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_166 = torch.nn.functional.batch_norm(
            x_165,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_165 = l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = (None)
        x_167 = x_166 + x_154
        x_166 = x_154 = None
        input_37 = torch.nn.functional.relu(x_167, inplace=False)
        x_168 = torch.conv2d(
            input_37,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_37 = l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (None)
        x_169 = torch.nn.functional.batch_norm(
            x_168,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_168 = l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = (None)
        x_170 = torch.conv2d(
            x_169,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_169 = l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_171 = torch.nn.functional.batch_norm(
            x_170,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_170 = l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = (None)
        input_38 = torch.nn.functional.relu(x_171, inplace=True)
        x_171 = None
        x_172 = torch.conv2d(
            input_38,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_38 = l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_173 = torch.nn.functional.batch_norm(
            x_172,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_172 = l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = (None)
        x_174 = torch.conv2d(
            x_173,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_173 = l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_175 = torch.nn.functional.batch_norm(
            x_174,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_174 = l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = (None)
        input_39 = torch.nn.functional.relu(x_175, inplace=True)
        x_175 = None
        x_176 = torch.conv2d(
            input_39,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_39 = l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_177 = torch.nn.functional.batch_norm(
            x_176,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_176 = l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = (None)
        x_178 = torch.conv2d(
            x_177,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_177 = l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_179 = torch.nn.functional.batch_norm(
            x_178,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_178 = l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = (None)
        x_180 = x_179 + x_167
        x_179 = x_167 = None
        input_40 = torch.nn.functional.relu(x_180, inplace=False)
        x_181 = torch.conv2d(
            input_40,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_40 = l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (None)
        x_182 = torch.nn.functional.batch_norm(
            x_181,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_181 = l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = (None)
        x_183 = torch.conv2d(
            x_182,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_182 = l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_184 = torch.nn.functional.batch_norm(
            x_183,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_183 = l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = (None)
        input_41 = torch.nn.functional.relu(x_184, inplace=True)
        x_184 = None
        x_185 = torch.conv2d(
            input_41,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_41 = l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_186 = torch.nn.functional.batch_norm(
            x_185,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_185 = l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = (None)
        x_187 = torch.conv2d(
            x_186,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_186 = l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_188 = torch.nn.functional.batch_norm(
            x_187,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_187 = l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = (None)
        input_42 = torch.nn.functional.relu(x_188, inplace=True)
        x_188 = None
        x_189 = torch.conv2d(
            input_42,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_42 = l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_190 = torch.nn.functional.batch_norm(
            x_189,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_189 = l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = (None)
        x_191 = torch.conv2d(
            x_190,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_190 = l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_192 = torch.nn.functional.batch_norm(
            x_191,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_191 = l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = (None)
        x_193 = x_192 + x_180
        x_192 = x_180 = None
        input_43 = torch.nn.functional.relu(x_193, inplace=False)
        x_194 = torch.conv2d(
            input_43,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_43 = l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (None)
        x_195 = torch.nn.functional.batch_norm(
            x_194,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_194 = l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = (None)
        x_196 = torch.conv2d(
            x_195,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_195 = l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_197 = torch.nn.functional.batch_norm(
            x_196,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_196 = l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = (None)
        input_44 = torch.nn.functional.relu(x_197, inplace=True)
        x_197 = None
        x_198 = torch.conv2d(
            input_44,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_44 = l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_199 = torch.nn.functional.batch_norm(
            x_198,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_198 = l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = (None)
        x_200 = torch.conv2d(
            x_199,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_199 = l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_201 = torch.nn.functional.batch_norm(
            x_200,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_200 = l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = (None)
        input_45 = torch.nn.functional.relu(x_201, inplace=True)
        x_201 = None
        x_202 = torch.conv2d(
            input_45,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_45 = l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_203 = torch.nn.functional.batch_norm(
            x_202,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_202 = l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = (None)
        x_204 = torch.conv2d(
            x_203,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_203 = l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_205 = torch.nn.functional.batch_norm(
            x_204,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_204 = l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = (None)
        x_206 = x_205 + x_193
        x_205 = x_193 = None
        input_46 = torch.nn.functional.relu(x_206, inplace=False)
        x_207 = torch.conv2d(
            input_46,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_46 = l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (None)
        x_208 = torch.nn.functional.batch_norm(
            x_207,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_207 = l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = (None)
        x_209 = torch.conv2d(
            x_208,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_208 = l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_210 = torch.nn.functional.batch_norm(
            x_209,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_209 = l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = (None)
        input_47 = torch.nn.functional.relu(x_210, inplace=True)
        x_210 = None
        x_211 = torch.conv2d(
            input_47,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_47 = l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_212 = torch.nn.functional.batch_norm(
            x_211,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_211 = l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = (None)
        x_213 = torch.conv2d(
            x_212,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_212 = l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_214 = torch.nn.functional.batch_norm(
            x_213,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_213 = l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = (None)
        input_48 = torch.nn.functional.relu(x_214, inplace=True)
        x_214 = None
        x_215 = torch.conv2d(
            input_48,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_48 = l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_216 = torch.nn.functional.batch_norm(
            x_215,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_215 = l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = (None)
        x_217 = torch.conv2d(
            x_216,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_216 = l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_218 = torch.nn.functional.batch_norm(
            x_217,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_217 = l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = (None)
        x_219 = x_218 + x_206
        x_218 = x_206 = None
        input_49 = torch.nn.functional.relu(x_219, inplace=False)
        x_220 = torch.conv2d(
            input_49,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_49 = l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (None)
        x_221 = torch.nn.functional.batch_norm(
            x_220,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_220 = l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = (None)
        x_222 = torch.conv2d(
            x_221,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_221 = l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_223 = torch.nn.functional.batch_norm(
            x_222,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_222 = l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = (None)
        input_50 = torch.nn.functional.relu(x_223, inplace=True)
        x_223 = None
        x_224 = torch.conv2d(
            input_50,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_50 = l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_225 = torch.nn.functional.batch_norm(
            x_224,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_224 = l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = (None)
        x_226 = torch.conv2d(
            x_225,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_225 = l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_227 = torch.nn.functional.batch_norm(
            x_226,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_226 = l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = (None)
        input_51 = torch.nn.functional.relu(x_227, inplace=True)
        x_227 = None
        x_228 = torch.conv2d(
            input_51,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_51 = l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_229 = torch.nn.functional.batch_norm(
            x_228,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_228 = l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = (None)
        x_230 = torch.conv2d(
            x_229,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_229 = l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_231 = torch.nn.functional.batch_norm(
            x_230,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_230 = l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = (None)
        x_232 = x_231 + x_219
        x_231 = x_219 = None
        input_52 = torch.nn.functional.relu(x_232, inplace=False)
        x_233 = torch.conv2d(
            input_52,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_52 = l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (None)
        x_234 = torch.nn.functional.batch_norm(
            x_233,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_233 = l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = (None)
        x_235 = torch.conv2d(
            x_234,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_234 = l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_236 = torch.nn.functional.batch_norm(
            x_235,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_235 = l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = (None)
        input_53 = torch.nn.functional.relu(x_236, inplace=True)
        x_236 = None
        x_237 = torch.conv2d(
            input_53,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_53 = l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_238 = torch.nn.functional.batch_norm(
            x_237,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_237 = l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = (None)
        x_239 = torch.conv2d(
            x_238,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_238 = l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_240 = torch.nn.functional.batch_norm(
            x_239,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_239 = l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = (None)
        input_54 = torch.nn.functional.relu(x_240, inplace=True)
        x_240 = None
        x_241 = torch.conv2d(
            input_54,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_54 = l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_242 = torch.nn.functional.batch_norm(
            x_241,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_241 = l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = (None)
        x_243 = torch.conv2d(
            x_242,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_242 = l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_244 = torch.nn.functional.batch_norm(
            x_243,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_243 = l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = (None)
        x_245 = x_244 + x_232
        x_244 = x_232 = None
        input_55 = torch.nn.functional.relu(x_245, inplace=False)
        x_246 = torch.conv2d(
            input_55,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_55 = l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (None)
        x_247 = torch.nn.functional.batch_norm(
            x_246,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_246 = l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = (None)
        x_248 = torch.conv2d(
            x_247,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_247 = l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_249 = torch.nn.functional.batch_norm(
            x_248,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_248 = l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = (None)
        input_56 = torch.nn.functional.relu(x_249, inplace=True)
        x_249 = None
        x_250 = torch.conv2d(
            input_56,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_56 = l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_251 = torch.nn.functional.batch_norm(
            x_250,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_250 = l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = (None)
        x_252 = torch.conv2d(
            x_251,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_251 = l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_253 = torch.nn.functional.batch_norm(
            x_252,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_252 = l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = (None)
        input_57 = torch.nn.functional.relu(x_253, inplace=True)
        x_253 = None
        x_254 = torch.conv2d(
            input_57,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_57 = l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_255 = torch.nn.functional.batch_norm(
            x_254,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_254 = l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = (None)
        x_256 = torch.conv2d(
            x_255,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_255 = l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_257 = torch.nn.functional.batch_norm(
            x_256,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_256 = l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = (None)
        x_258 = x_257 + x_245
        x_257 = x_245 = None
        input_58 = torch.nn.functional.relu(x_258, inplace=False)
        x_259 = torch.conv2d(
            input_58,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_58 = l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (None)
        x_260 = torch.nn.functional.batch_norm(
            x_259,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_259 = l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = (None)
        x_261 = torch.conv2d(
            x_260,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_260 = l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_262 = torch.nn.functional.batch_norm(
            x_261,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_261 = l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = (None)
        input_59 = torch.nn.functional.relu(x_262, inplace=True)
        x_262 = None
        x_263 = torch.conv2d(
            input_59,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_59 = l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_264 = torch.nn.functional.batch_norm(
            x_263,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_263 = l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = (None)
        x_265 = torch.conv2d(
            x_264,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_264 = l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_266 = torch.nn.functional.batch_norm(
            x_265,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_265 = l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = (None)
        input_60 = torch.nn.functional.relu(x_266, inplace=True)
        x_266 = None
        x_267 = torch.conv2d(
            input_60,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1024,
        )
        input_60 = l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_268 = torch.nn.functional.batch_norm(
            x_267,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_267 = l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = (None)
        x_269 = torch.conv2d(
            x_268,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_268 = l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_270 = torch.nn.functional.batch_norm(
            x_269,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_269 = l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = (None)
        x_271 = torch.conv2d(
            x_258,
            l_self_modules_blocks_modules_19_modules_shortcut_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_258 = l_self_modules_blocks_modules_19_modules_shortcut_modules_conv_parameters_weight_ = (None)
        x_272 = torch.nn.functional.batch_norm(
            x_271,
            l_self_modules_blocks_modules_19_modules_shortcut_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_19_modules_shortcut_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_19_modules_shortcut_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_shortcut_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_271 = l_self_modules_blocks_modules_19_modules_shortcut_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_19_modules_shortcut_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_19_modules_shortcut_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_19_modules_shortcut_modules_bn_parameters_bias_ = (None)
        x_273 = x_270 + x_272
        x_270 = x_272 = None
        x_274 = torch.conv2d(
            x_273,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1024,
        )
        x_273 = l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (None)
        x_275 = torch.nn.functional.batch_norm(
            x_274,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_274 = l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_dw_parameters_bias_ = (None)
        x_276 = torch.nn.functional.relu(x_275, inplace=True)
        x_275 = None
        x_277 = torch.conv2d(
            x_276,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_276 = l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_278 = torch.nn.functional.batch_norm(
            x_277,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_277 = l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_bn_pw_parameters_bias_ = (None)
        x_279 = torch.nn.functional.relu(x_278, inplace=True)
        x_278 = None
        x_280 = torch.conv2d(
            x_279,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        x_279 = l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_281 = torch.nn.functional.batch_norm(
            x_280,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_280 = l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_dw_parameters_bias_ = (None)
        x_282 = torch.nn.functional.relu(x_281, inplace=True)
        x_281 = None
        x_283 = torch.conv2d(
            x_282,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_282 = l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_284 = torch.nn.functional.batch_norm(
            x_283,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_283 = l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_bn_pw_parameters_bias_ = (None)
        x_285 = torch.nn.functional.relu(x_284, inplace=True)
        x_284 = None
        x_286 = torch.conv2d(
            x_285,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        x_285 = l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_287 = torch.nn.functional.batch_norm(
            x_286,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_286 = l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_dw_buffers_running_mean_ = l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_dw_buffers_running_var_ = l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_dw_parameters_weight_ = l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_dw_parameters_bias_ = (None)
        x_288 = torch.nn.functional.relu(x_287, inplace=True)
        x_287 = None
        x_289 = torch.conv2d(
            x_288,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_288 = l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_290 = torch.nn.functional.batch_norm(
            x_289,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_289 = l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_pw_buffers_running_mean_ = l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_pw_buffers_running_var_ = l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_pw_parameters_weight_ = l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_bn_pw_parameters_bias_ = (None)
        x_291 = torch.nn.functional.relu(x_290, inplace=True)
        x_290 = None
        x_292 = torch.nn.functional.adaptive_avg_pool2d(x_291, 1)
        x_291 = None
        x_293 = x_292.flatten(1, -1)
        x_292 = None
        x_294 = torch.nn.functional.dropout(x_293, 0.0, False, False)
        x_293 = None
        x_295 = torch._C._nn.linear(
            x_294,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_294 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_295,)
